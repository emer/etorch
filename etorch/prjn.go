// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package etorch

import (
	"errors"
	"fmt"
	"io"
	"log"
	"sort"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
)

// Prjn contains the basic structural information for specifying a projection of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The exact same struct object is added to the Recv and Send layers, and it manages everything
// about the connectivity, and methods on the Prjn handle all the relevant computation.
type Prjn struct {
	Off         bool                        `desc:"inactivate this projection -- allows for easy experimentation"`
	Cls         string                      `desc:"Class is for applying parameter styles, can be space separated multple tags"`
	Notes       string                      `desc:"can record notes about this projection here"`
	Send        emer.Layer                  `desc:"sending layer for this projection"`
	Recv        emer.Layer                  `desc:"receiving layer for this projection -- the emer.Layer interface can be converted to the specific Layer type you are using, e.g., rlay := prjn.Recv.(*leabra.Layer)"`
	Pat         prjn.Pattern                `desc:"pattern of connectivity"`
	Typ         emer.PrjnType               `desc:"type of projection -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)"`
	RConN       []int32                     `view:"-" desc:"number of recv connections for each neuron in the receiving layer, as a flat list"`
	RConNAvgMax minmax.AvgMax32             `inactive:"+" desc:"average and maximum number of recv connections in the receiving layer"`
	RConIdxSt   []int32                     `view:"-" desc:"starting index into ConIdx list for each neuron in receiving layer -- just a list incremented by ConN"`
	RConIdx     []int32                     `view:"-" desc:"index of other neuron on sending side of projection, ordered by the receiving layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`
	SSynIdx     []int32                     `view:"-" desc:"index of synaptic state values for each send unit x connection, for the sending projection which does not own the synapses, and instead indexes into recv-ordered list"`
	SConN       []int32                     `view:"-" desc:"number of sending connections for each neuron in the sending layer, as a flat list"`
	SConNAvgMax minmax.AvgMax32             `inactive:"+" desc:"average and maximum number of sending connections in the sending layer"`
	SConIdxSt   []int32                     `view:"-" desc:"starting index into ConIdx list for each neuron in sending layer -- just a list incremented by ConN"`
	SConIdx     []int32                     `view:"-" desc:"index of other neuron on receiving side of projection, ordered by the sending layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`
	States      map[string]*etensor.Float32 `desc:"map of states of the projection (weights, etc) -- name is variable name, tensor holds the data"`
	VarNamesMap map[string]int              `view:"-" desc:"map of variable names with index into the VarNames list"`
	VarNames    []string                    `view:"-" desc:"list of variable names alpha order"`
}

// emer.Prjn interface

// Init MUST be called to initialize the prjn's pointer to itself as an emer.Prjn
// which enables the proper interface methods to be called.
func (pj *Prjn) Init(prjn emer.Prjn) {
	pj.AddWtDWtVars()
}

func (pj *Prjn) TypeName() string              { return "Prjn" } // always, for params..
func (pj *Prjn) Class() string                 { return pj.Cls }
func (pj *Prjn) SetClass(cls string) emer.Prjn { pj.Cls = cls; return pj }
func (pj *Prjn) Name() string {
	return pj.Send.Name() + "To" + pj.Recv.Name()
}
func (pj *Prjn) Label() string                         { return pj.Name() }
func (pj *Prjn) RecvLay() emer.Layer                   { return pj.Recv }
func (pj *Prjn) SendLay() emer.Layer                   { return pj.Send }
func (pj *Prjn) Pattern() prjn.Pattern                 { return pj.Pat }
func (pj *Prjn) SetPattern(pat prjn.Pattern) emer.Prjn { pj.Pat = pat; return pj }
func (pj *Prjn) Type() emer.PrjnType                   { return pj.Typ }
func (pj *Prjn) SetType(typ emer.PrjnType) emer.Prjn   { pj.Typ = typ; return pj }
func (pj *Prjn) PrjnTypeName() string                  { return pj.Typ.String() }

func (pj *Prjn) IsOff() bool {
	return pj.Off || pj.Recv.IsOff() || pj.Send.IsOff()
}
func (pj *Prjn) SetOff(off bool) { pj.Off = off }

func (pj *Prjn) Defaults() {
}

// UpdateParams updates all params given any changes that might have been made to individual values
func (pj *Prjn) UpdateParams() {
}

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pj *Prjn) Connect(slay, rlay emer.Layer, pat prjn.Pattern, typ emer.PrjnType) {
	pj.Send = slay
	pj.Recv = rlay
	pj.Pat = pat
	pj.Typ = typ
}

// Validate tests for non-nil settings for the projection -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (pj *Prjn) Validate(logmsg bool) error {
	emsg := ""
	if pj.Pat == nil {
		emsg += "Pat is nil; "
	}
	if pj.Recv == nil {
		emsg += "Recv is nil; "
	}
	if pj.Send == nil {
		emsg += "Send is nil; "
	}
	if emsg != "" {
		err := errors.New(emsg)
		if logmsg {
			log.Println(emsg)
		}
		return err
	}
	return nil
}

// AddVar adds a variable to record in this layer.  Each variable is recorded in
// a separate etensor.Float32, which is updated in python from torch tensors.
func (pj *Prjn) AddVar(varNm string) {
	if pj.VarNamesMap == nil {
		pj.VarNamesMap = make(map[string]int)
	}
	pj.VarNamesMap[varNm] = 0
}

// AddVars adds variables to record in this layer.  Each variable is recorded in
// a separate etensor.Float32, which is updated in python from torch tensors.
func (pj *Prjn) AddVars(varNms []string) {
	for _, nm := range varNms {
		pj.AddVar(nm)
	}
}

// AddWtDWtVars adds standard Wt, DWt variables
func (pj *Prjn) AddWtDWtVars() {
	pj.AddVars([]string{"Wt", "DWt"})
}

// BuildStru constructs the full connectivity among the layers as specified in this projection.
// Calls Validate and returns false if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
func (pj *Prjn) BuildStru() error {
	if pj.Off {
		return nil
	}
	err := pj.Validate(true)
	if err != nil {
		return err
	}
	ssh := pj.Send.Shape()
	rsh := pj.Recv.Shape()
	sendn, recvn, cons := pj.Pat.Connect(ssh, rsh, pj.Recv == pj.Send)
	slen := ssh.Len()
	rlen := rsh.Len()
	tcons := pj.SetNIdxSt(&pj.SConN, &pj.SConNAvgMax, &pj.SConIdxSt, sendn)
	tconr := pj.SetNIdxSt(&pj.RConN, &pj.RConNAvgMax, &pj.RConIdxSt, recvn)
	if tconr != tcons {
		log.Printf("%v programmer error: total recv cons %v != total send cons %v\n", pj.String(), tconr, tcons)
	}
	pj.RConIdx = make([]int32, tconr)
	pj.SSynIdx = make([]int32, tconr)
	pj.SConIdx = make([]int32, tcons)
	sconN := make([]int32, slen) // temporary mem needed to tracks cur n of sending cons

	cbits := cons.Values
	for ri := 0; ri < rlen; ri++ {
		rbi := ri * slen     // recv bit index
		rtcn := pj.RConN[ri] // number of cons
		rst := pj.RConIdxSt[ri]
		rci := int32(0)
		for si := 0; si < slen; si++ {
			if !cbits.Index(rbi + si) { // no connection
				continue
			}
			sst := pj.SConIdxSt[si]
			if rci >= rtcn {
				log.Printf("%v programmer error: recv target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), rtcn, ri, si)
				break
			}
			pj.RConIdx[rst+rci] = int32(si)

			sci := sconN[si]
			stcn := pj.SConN[si]
			if sci >= stcn {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pj.String(), stcn, ri, si)
				break
			}
			pj.SConIdx[sst+sci] = int32(ri)
			pj.SSynIdx[sst+sci] = rst + rci
			(sconN[si])++
			rci++
		}
	}
	return nil
}

// SetNIdxSt sets the *ConN and *ConIdxSt values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pj *Prjn) SetNIdxSt(n *[]int32, avgmax *minmax.AvgMax32, idxst *[]int32, tn *etensor.Int32) int32 {
	ln := tn.Len()
	tnv := tn.Values
	*n = make([]int32, ln)
	*idxst = make([]int32, ln)
	idx := int32(0)
	avgmax.Init()
	for i := 0; i < ln; i++ {
		nv := tnv[i]
		(*n)[i] = nv
		(*idxst)[i] = idx
		idx += nv
		avgmax.UpdateVal(float32(nv), i)
	}
	avgmax.CalcAvg()
	return idx
}

// String satisfies fmt.Stringer for prjn
func (pj *Prjn) String() string {
	str := ""
	if pj.Recv == nil {
		str += "recv=nil; "
	} else {
		str += pj.Recv.Name() + " <- "
	}
	if pj.Send == nil {
		str += "send=nil"
	} else {
		str += pj.Send.Name()
	}
	if pj.Pat == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + pj.Pat.Name()
	}
	return str
}

// ApplyParams applies given parameter style Sheet to this projection.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (pj *Prjn) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(pj, setMsg)
	return app, err
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (pj *Prjn) NonDefaultParams() string {
	pth := pj.Recv.Name() + "." + pj.Name() // redundant but clearer..
	nds := giv.StructNonDefFieldsStr(pj, pth)
	return nds
}

// AllParams returns a listing of all parameters in the Layer
func (pj *Prjn) AllParams() string {
	return ""
}

// BuildVarNames makes the var names from VarNamesMap added previously
func (pj *Prjn) BuildVarNames() {
	pj.VarNames = make([]string, len(pj.VarNamesMap))
	i := 0
	for nm := range pj.VarNamesMap {
		pj.VarNames[i] = nm
		i++
	}
	sort.Strings(pj.VarNames)
	for i := range pj.VarNames {
		pj.VarNamesMap[pj.VarNames[i]] = i
	}
}

func (pj *Prjn) SynVarNames() []string {
	return pj.VarNames
}

// SynVarProps returns properties for variables
func (pj *Prjn) SynVarProps() map[string]string {
	return nil
}

// SynIdx returns the index of the synapse between given send, recv unit indexes
// (1D, flat indexes). Returns -1 if synapse not found between these two neurons.
// Requires searching within connections for sending unit.
func (pj *Prjn) SynIdx(sidx, ridx int) int {
	nc := int(pj.RConN[ridx])
	st := int(pj.RConIdxSt[ridx])
	for ci := 0; ci < nc; ci++ {
		si := int(pj.RConIdx[st+ci])
		if si != sidx {
			continue
		}
		return int(st + ci)
	}
	return -1
}

// SynVarIdx returns the index of given variable within the synapse,
// according to *this prjn's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *Prjn) SynVarIdx(varNm string) (int, error) {
	vi, ok := pj.VarNamesMap[varNm]
	if !ok {
		return -1, fmt.Errorf("variable name not found: %s in Prjn: %s", varNm, pj.Name())
	}
	return vi, nil
}

// SynVarNum returns the number of synapse-level variables
// for this prjn.  This is needed for extending indexes in derived types.
func (pj *Prjn) SynVarNum() int {
	return len(pj.VarNames)
}

// Syn1DNum returns the number of synapses for this prjn as a 1D array.
// This is the max idx for SynVal1D and the number of vals set by SynVals.
func (pj *Prjn) Syn1DNum() int {
	vnm := pj.VarNames[0]
	st := pj.States[vnm]
	return st.Len()
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *Prjn) SynVal1D(varIdx int, synIdx int) float32 {
	if varIdx < 0 || varIdx >= pj.SynVarNum() {
		return mat32.NaN()
	}
	vnm := pj.VarNames[varIdx]
	st := pj.States[vnm]
	if synIdx < 0 || synIdx >= st.Len() {
		return mat32.NaN()
	}
	return st.Value1D(synIdx)
}

// SynVals sets values of given variable name for each synapse, using the natural ordering
// of the synapses (recv based)
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pj *Prjn) SynVals(vals *[]float32, varNm string) error {
	_, err := pj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	st := pj.States[varNm]
	ns := st.Len()
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[0:ns]
	}
	for i := range st.Values {
		(*vals)[i] = st.Values[i]
	}
	return nil
}

// SynVal returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns mat32.NaN() for access errors (see SynValTry for error message)
func (pj *Prjn) SynVal(varNm string, sidx, ridx int) float32 {
	vidx, err := pj.SynVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	synIdx := pj.SynIdx(sidx, ridx)
	return pj.SynVal1D(vidx, synIdx)
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pj *Prjn) SetSynVal(varNm string, sidx, ridx int, val float32) error {
	_, err := pj.SynVarIdx(varNm)
	if err != nil {
		return err
	}
	st := pj.States[varNm]
	ns := st.Len()
	synIdx := pj.SynIdx(sidx, ridx)
	if synIdx < 0 || synIdx >= ns {
		return err
	}
	st.Values[synIdx] = val
	return nil
}

// Build constructs the full connectivity among the layers as specified in this projection.
// Calls PrjnStru.BuildStru and then allocates the synaptic values in Syns accordingly.
func (pj *Prjn) Build() error {
	if err := pj.BuildStru(); err != nil {
		return err
	}
	pj.BuildVarNames()
	pj.States = make(map[string]*etensor.Float32, len(pj.VarNamesMap))
	ncons := len(pj.RConIdx)
	for vn := range pj.VarNamesMap {
		st := etensor.NewFloat32([]int{ncons}, nil, nil)
		pj.States[vn] = st
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWtsJSON writes the weights from this projection from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pj *Prjn) WriteWtsJSON(w io.Writer, depth int) {
}

// ReadWtsJSON reads the weights from this projection from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one prjn only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (pj *Prjn) ReadWtsJSON(r io.Reader) error {
	return nil
}

// SetWts sets the weights for this projection from weights.Prjn decoded values
func (pj *Prjn) SetWts(pw *weights.Prjn) error {
	return nil
}
