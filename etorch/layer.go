// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package etorch

import (
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"sort"
	"strings"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
)

// etorch.Layer manages the structural elements of the layer, which are common
// to any Layer type
type Layer struct {
	Network     emer.Network                `copy:"-" json:"-" xml:"-" view:"-" desc:"our parent network, in case we need to use it to find other layers etc -- set when added by network"`
	Nm          string                      `desc:"Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name"`
	Cls         string                      `desc:"Class is for applying parameter styles, can be space separated multple tags"`
	Off         bool                        `desc:"inactivate this layer -- allows for easy experimentation"`
	Shp         etensor.Shape               `desc:"shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools"`
	Typ         emer.LayerType              `desc:"type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)"`
	Thr         int                         `desc:"the thread number (go routine) to use in updating this layer. The user is responsible for allocating layers to threads, trying to maintain an even distribution across layers and establishing good break-points."`
	Rel         relpos.Rel                  `view:"inline" desc:"Spatial relationship to other layer, determines positioning"`
	Ps          mat32.Vec3                  `desc:"position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis."`
	Idx         int                         `desc:"a 0..n-1 index of the position of the layer within list of layers in the network."`
	RcvPrjns    emer.Prjns                  `desc:"list of receiving projections into this layer from other layers"`
	SndPrjns    emer.Prjns                  `desc:"list of sending projections from this layer to other layers"`
	States      map[string]*etensor.Float32 `desc:"map of states of the layer (activation, etc) -- name is variable name, tensor holds the data"`
	VarNamesMap map[string]int              `view:"-" desc:"map of variable names with index into the VarNames list"`
	VarNames    []string                    `view:"-" desc:"list of variable names alpha order"`
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ly *Layer) InitName(lay emer.Layer, name string, net emer.Network) {
	ly.Nm = name
	ly.Network = net
}

func (ly *Layer) Name() string               { return ly.Nm }
func (ly *Layer) SetName(nm string)          { ly.Nm = nm }
func (ly *Layer) Label() string              { return ly.Nm }
func (ly *Layer) Class() string              { return ly.Typ.String() + " " + ly.Cls }
func (ly *Layer) SetClass(cls string)        { ly.Cls = cls }
func (ly *Layer) TypeName() string           { return "Layer" } // type category, for params..
func (ly *Layer) Type() emer.LayerType       { return ly.Typ }
func (ly *Layer) SetType(typ emer.LayerType) { ly.Typ = typ }
func (ly *Layer) IsOff() bool                { return ly.Off }
func (ly *Layer) SetOff(off bool)            { ly.Off = off }
func (ly *Layer) Shape() *etensor.Shape      { return &ly.Shp }
func (ly *Layer) Is2D() bool                 { return ly.Shp.NumDims() == 2 }
func (ly *Layer) Is4D() bool                 { return ly.Shp.NumDims() == 4 }
func (ly *Layer) Thread() int                { return ly.Thr }
func (ly *Layer) SetThread(thr int)          { ly.Thr = thr }
func (ly *Layer) RelPos() relpos.Rel         { return ly.Rel }
func (ly *Layer) Pos() mat32.Vec3            { return ly.Ps }
func (ly *Layer) SetPos(pos mat32.Vec3)      { ly.Ps = pos }
func (ly *Layer) Index() int                 { return ly.Idx }
func (ly *Layer) SetIndex(idx int)           { ly.Idx = idx }
func (ly *Layer) RecvPrjns() *emer.Prjns     { return &ly.RcvPrjns }
func (ly *Layer) NRecvPrjns() int            { return len(ly.RcvPrjns) }
func (ly *Layer) RecvPrjn(idx int) emer.Prjn { return ly.RcvPrjns[idx] }
func (ly *Layer) SendPrjns() *emer.Prjns     { return &ly.SndPrjns }
func (ly *Layer) NSendPrjns() int            { return len(ly.SndPrjns) }
func (ly *Layer) SendPrjn(idx int) emer.Prjn { return ly.SndPrjns[idx] }

func (ly *Layer) Idx4DFrom2D(x, y int) ([]int, bool) {
	lshp := ly.Shape()
	nux := lshp.Dim(3)
	nuy := lshp.Dim(2)
	ux := x % nux
	uy := y % nuy
	px := x / nux
	py := y / nuy
	idx := []int{py, px, uy, ux}
	if !lshp.IdxIsValid(idx) {
		return nil, false
	}
	return idx, true
}

func (ly *Layer) Defaults() {

}

func (ly *Layer) SetRelPos(rel relpos.Rel) {
	ly.Rel = rel
	if ly.Rel.Scale == 0 {
		ly.Rel.Defaults()
	}
}

func (ly *Layer) Size() mat32.Vec2 {
	if ly.Rel.Scale == 0 {
		ly.Rel.Defaults()
	}
	var sz mat32.Vec2
	switch {
	case ly.Is2D():
		sz = mat32.Vec2{float32(ly.Shp.Dim(1)), float32(ly.Shp.Dim(0))} // Y, X
	case ly.Is4D():
		// note: pool spacing is handled internally in display and does not affect overall size
		sz = mat32.Vec2{float32(ly.Shp.Dim(1) * ly.Shp.Dim(3)), float32(ly.Shp.Dim(0) * ly.Shp.Dim(2))} // Y, X
	default:
		sz = mat32.Vec2{float32(ly.Shp.Len()), 1}
	}
	return sz.MulScalar(ly.Rel.Scale)
}

// SetShape sets the layer shape and also uses default dim names
func (ly *Layer) SetShape(shape []int) {
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ly.Shp.SetShape(shape, nil, dnms) // row major default
}

// AddVar adds a variable to record in this layer.  Each variable is recorded in
// a separate etensor.Float32, which is updated in python from torch tensors.
func (ly *Layer) AddVar(varNm string) {
	if ly.VarNamesMap == nil {
		ly.VarNamesMap = make(map[string]int)
	}
	ly.VarNamesMap[varNm] = 0
}

// AddVars adds variables to record in this layer.  Each variable is recorded in
// a separate etensor.Float32, which is updated in python from torch tensors.
func (ly *Layer) AddVars(varNms []string) {
	for _, nm := range varNms {
		ly.AddVar(nm)
	}
}

// AddNetinActBiasVars adds standard Netin, Act, Bias variables
func (ly *Layer) AddNetinActBiasVars() {
	ly.AddVars([]string{"Net", "Act", "Bias"})
}

// NPools returns the number of unit sub-pools according to the shape parameters.
// Currently supported for a 4D shape, where the unit pools are the first 2 Y,X dims
// and then the units within the pools are the 2nd 2 Y,X dims
func (ly *Layer) NPools() int {
	if ly.Shp.NumDims() != 4 {
		return 0
	}
	return ly.Shp.Dim(0) * ly.Shp.Dim(1)
}

// RecipToSendPrjn finds the reciprocal projection relative to the given sending projection
// found within the SendPrjns of this layer.  This is then a recv prjn within this layer:
//  S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
// returns false if not found.
func (ly *Layer) RecipToSendPrjn(spj emer.Prjn) (emer.Prjn, bool) {
	for _, rpj := range ly.RcvPrjns {
		if rpj.SendLay() == spj.RecvLay() {
			return rpj, true
		}
	}
	return nil, false
}

// Config configures the basic properties of the layer
func (ly *Layer) Config(shape []int, typ emer.LayerType) {
	ly.SetShape(shape)
	ly.Typ = typ
	ly.AddNetinActBiasVars() // default
}

// ApplyParams applies given parameter style Sheet to this layer and its recv projections.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ly *Layer) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	app, err := pars.Apply(ly, setMsg)
	if app {
		applied = true
	}
	if err != nil {
		rerr = err
	}
	for _, pj := range ly.RcvPrjns {
		app, err = pj.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ly *Layer) NonDefaultParams() string {
	nds := giv.StructNonDefFieldsStr(ly, ly.Nm)
	for _, pj := range ly.RcvPrjns {
		pnd := pj.NonDefaultParams()
		nds += pnd
	}
	return nds
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	for _, pj := range ly.RcvPrjns {
		pj.UpdateParams()
	}
}

// JsonToParams reformates json output to suitable params display output
func JsonToParams(b []byte) string {
	br := strings.Replace(string(b), `"`, ``, -1)
	br = strings.Replace(br, ",\n", "", -1)
	br = strings.Replace(br, "{\n", "{", -1)
	br = strings.Replace(br, "} ", "}\n  ", -1)
	br = strings.Replace(br, "\n }", " }", -1)
	br = strings.Replace(br, "\n  }\n", " }", -1)
	return br[1:] + "\n"
}

// AllParams returns a listing of all parameters in the Layer
func (ly *Layer) AllParams() string {
	return ""
}

// BuildVarNames makes the var names from VarNamesMap added previously
func (ly *Layer) BuildVarNames() {
	ly.VarNames = make([]string, len(ly.VarNamesMap))
	i := 0
	for nm := range ly.VarNamesMap {
		ly.VarNames[i] = nm
		i++
	}
	sort.Strings(ly.VarNames)
	for i := range ly.VarNames {
		ly.VarNamesMap[ly.VarNames[i]] = i
	}
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return ly.VarNames
}

// UnitVarProps returns properties for variables
func (ly *Layer) UnitVarProps() map[string]string {
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vi, ok := ly.VarNamesMap[varNm]
	if !ok {
		return -1, fmt.Errorf("variable name not found: %s in layer: %s", varNm, ly.Nm)
	}
	return vi, nil
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return len(ly.VarNames)
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= ly.Shp.Len() {
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= ly.UnitVarNum() {
		return mat32.NaN()
	}
	vnm := ly.VarNames[varIdx]
	st := ly.States[vnm]
	return float32(st.FloatVal1D(idx))
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitVals(vals *[]float32, varNm string) error {
	nn := ly.Shp.Len()
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	_, err := ly.UnitVarIdx(varNm)
	if err != nil {
		nan := mat32.NaN()
		for i := 0; i < nn; i++ {
			(*vals)[i] = nan
		}
		return err
	}
	st := ly.States[varNm]
	for i := 0; i < nn; i++ {
		(*vals)[i] = st.Value1D(i)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	nn := ly.Shp.Len()
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	_, err := ly.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for i := 0; i < nn; i++ {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	st := ly.States[varNm]
	for i := 0; i < nn; i++ {
		v := st.Value1D(i)
		tsr.SetFloat1D(i, float64(v))
	}
	return nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitVal(varNm string, idx []int) float32 {
	_, err := ly.UnitVarIdx(varNm)
	if err != nil {
		return mat32.NaN()
	}
	st := ly.States[varNm]
	return st.Value(idx)
}

// RecvPrjnVals fills in values of given synapse variable name,
// for projection into given sending layer and neuron 1D index,
// for all receiving neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the receiving neuron is not connected to the given sending layer or neuron
// then the value is set to mat32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) RecvPrjnVals(vals *[]float32, varNm string, sendLay emer.Layer, sendIdx1D int, prjnType string) error {
	var err error
	nn := ly.Shp.Len()
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := mat32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("sending layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = sendLay.SendPrjns().RecvNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
		}
	} else {
		pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pj.SynVal(varNm, sendIdx1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

// SendPrjnVals fills in values of given synapse variable name,
// for projection into given receiving layer and neuron 1D index,
// for all sending neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the sending neuron is not connected to the given receiving layer or neuron
// then the value is set to mat32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) SendPrjnVals(vals *[]float32, varNm string, recvLay emer.Layer, recvIdx1D int, prjnType string) error {
	var err error
	nn := ly.Shp.Len()
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := mat32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("receiving layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = recvLay.RecvPrjns().SendNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
		}
	} else {
		pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pj.SynVal(varNm, si, recvIdx1D)
	}
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildPrjns builds the projections, recv-side
func (ly *Layer) BuildPrjns() error {
	emsg := ""
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		err := pj.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// Build constructs the layer state, including calling Build on the projections
func (ly *Layer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.BuildVarNames()
	ly.States = make(map[string]*etensor.Float32, len(ly.VarNamesMap))
	for vn := range ly.VarNamesMap {
		st := etensor.NewFloat32Shape(&ly.Shp, nil)
		ly.States[vn] = st
	}
	err := ly.BuildPrjns()
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
	sz := ly.Shp.Len()
	if sz == 0 {
		return
	}
	_, err = ly.UnitVarIdx(varNm)
	if err != nil {
		return
	}
	st := ly.States[varNm]
	v0 := st.Value1D(0)
	min = v0
	max = v0
	for i := 1; i < sz; i++ {
		vl := st.Value1D(i)
		if vl < min {
			min = vl
		}
		if vl > max {
			max = vl
		}
	}
	return
}

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWtsJSON(w io.Writer, depth int) {
}

// ReadWtsJSON reads the weights from this layer from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one layer only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (ly *Layer) ReadWtsJSON(r io.Reader) error {
	return nil
}

// SetWts sets the weights for this layer from weights.Layer decoded values
func (ly *Layer) SetWts(lw *weights.Layer) error {
	return nil

}
