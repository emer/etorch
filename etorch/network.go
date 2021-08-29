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
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/weights"
	"github.com/goki/gi/gi"
	"github.com/goki/mat32"
)

// etorch.Network holds the layers of the network
type Network struct {
	EmerNet         emer.Network          `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Nm              string                `desc:"overall name of network -- helps discriminate if there are multiple"`
	Layers          emer.Layers           `desc:"list of layers"`
	LayMap          map[string]emer.Layer `view:"-" desc:"map of name to layers -- layer names must be unique"`
	MinPos          mat32.Vec3            `view:"-" desc:"minimum display position in network"`
	MaxPos          mat32.Vec3            `view:"-" desc:"maximum display position in network"`
	MetaData        map[string]string     `desc:"optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save"`
	LayVarNamesMap  map[string]int        `view:"-" desc:"map of variable names accumulated across layers, with index into the LayVarNames list"`
	LayVarNames     []string              `view:"-" desc:"list of variable names accumulated across layers, alpha order"`
	PrjnVarNamesMap map[string]int        `view:"-" desc:"map of variable names accumulated across prjns, with index into the LayVarNames list"`
	PrjnVarNames    []string              `view:"-" desc:"list of variable names accumulated across prjns, alpha order"`
}

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name.
func (nt *Network) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
}

// emer.Network interface methods:
func (nt *Network) Name() string                  { return nt.Nm }
func (nt *Network) Label() string                 { return nt.Nm }
func (nt *Network) NLayers() int                  { return len(nt.Layers) }
func (nt *Network) Layer(idx int) emer.Layer      { return nt.Layers[idx] }
func (nt *Network) Bounds() (min, max mat32.Vec3) { min = nt.MinPos; max = nt.MaxPos; return }

// LayerByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *Network) LayerByName(name string) emer.Layer {
	if nt.LayMap == nil || len(nt.LayMap) != len(nt.Layers) {
		nt.MakeLayMap()
	}
	ly := nt.LayMap[name]
	return ly
}

// LayerByNameTry returns a layer by looking it up by name -- emits a log error message
// if layer is not found
func (nt *Network) LayerByNameTry(name string) (emer.Layer, error) {
	ly := nt.LayerByName(name)
	if ly == nil {
		err := fmt.Errorf("Layer named: %v not found in Network: %v\n", name, nt.Nm)
		log.Println(err)
		return ly, err
	}
	return ly, nil
}

// MakeLayMap updates layer map based on current layers
func (nt *Network) MakeLayMap() {
	nt.LayMap = make(map[string]emer.Layer, len(nt.Layers))
	for _, ly := range nt.Layers {
		nt.LayMap[ly.Name()] = ly
	}
}

// StdVertLayout arranges layers in a standard vertical (z axis stack) layout, by setting
// the Rel settings
func (nt *Network) StdVertLayout() {
	lstnm := ""
	for li, ly := range nt.Layers {
		if li == 0 {
			ly.SetRelPos(relpos.Rel{Rel: relpos.NoRel})
			lstnm = ly.Name()
		} else {
			ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstnm, XAlign: relpos.Middle, YAlign: relpos.Front})
		}
	}
}

// Layout computes the 3D layout of layers based on their relative position settings
func (nt *Network) Layout() {
	for itr := 0; itr < 5; itr++ {
		var lstly emer.Layer
		for _, ly := range nt.Layers {
			rp := ly.RelPos()
			var oly emer.Layer
			if lstly != nil && rp.Rel == relpos.NoRel {
				oly = lstly
				ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
			} else {
				if rp.Other != "" {
					var err error
					oly, err = nt.LayerByNameTry(rp.Other)
					if err != nil {
						log.Println(err)
						continue
					}
				} else if lstly != nil {
					oly = lstly
					ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
				}
			}
			if oly != nil {
				ly.SetPos(rp.Pos(oly.Pos(), oly.Size(), ly.Size()))
			}
			lstly = ly
		}
	}
	nt.BoundsUpdt()
}

// BoundsUpdt updates the Min / Max display bounds for 3D display
func (nt *Network) BoundsUpdt() {
	mn := mat32.NewVec3Scalar(mat32.Infinity)
	mx := mat32.Vec3Zero
	for _, ly := range nt.Layers {
		ps := ly.Pos()
		sz := ly.Size()
		ru := ps
		ru.X += sz.X
		ru.Y += sz.Y
		mn.SetMax(ps)
		mx.SetMax(ru)
	}
	nt.MaxPos = mn
	nt.MaxPos = mx
}

// ApplyParams applies given parameter style Sheet to layers and prjns in this network.
// Calls UpdateParams to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (nt *Network) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	for _, ly := range nt.Layers {
		app, err := ly.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Network that
// are not at their default values -- useful for setting param styles etc.
func (nt *Network) NonDefaultParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.NonDefaultParams()
		nds += nd
	}
	return nds
}

// AllParams returns a listing of all parameters in the Network.
func (nt *Network) AllParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.AllParams()
		nds += nd
	}
	return nds
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *Network) AddLayerInit(ly emer.Layer, name string, shape []int, typ emer.LayerType) {
	if nt.EmerNet == nil {
		log.Printf("Network EmerNet is nil -- you MUST call InitName on network, passing a pointer to the network to initialize properly!")
		return
	}
	ly.InitName(ly, name, nt.EmerNet)
	ly.Config(shape, typ)
	nt.Layers = append(nt.Layers, ly)
	nt.MakeLayMap()
}

// AddLayer adds a new layer with given name and shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential -- see
// AddLayer2D and 4D for convenience methods for those.  4D layers enable
// pool (unit-group) level inhibition in Etorch networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each unit
// group having 4 rows (Y) of 5 (X) units.
func (nt *Network) AddLayer(name string, shape []int, typ emer.LayerType) emer.Layer {
	ly := nt.EmerNet.NewLayer() // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *Network) AddLayer2D(name string, shapeY, shapeX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Etorch networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *Network) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a projection between two layers, referenced by name
// adding to the recv and send projection lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) ConnectLayerNames(send, recv string, pat prjn.Pattern, typ emer.PrjnType) (rlay, slay emer.Layer, pj emer.Prjn, err error) {
	rlay, err = nt.LayerByNameTry(recv)
	if err != nil {
		return
	}
	slay, err = nt.LayerByNameTry(send)
	if err != nil {
		return
	}
	pj = nt.ConnectLayers(slay, rlay, pat, typ)
	return
}

// ConnectLayers establishes a projection between two layers,
// adding to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) ConnectLayers(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.ConnectLayersPrjn(send, recv, pat, typ, pj)
}

// ConnectLayersPrjn makes connection using given projection between two layers,
// adding given prjn to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) ConnectLayersPrjn(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(send, recv, pat, typ)
	recv.RecvPrjns().Add(pj)
	send.SendPrjns().Add(pj)
	return pj
}

// BidirConnectLayerNames establishes bidirectional projections between two layers,
// referenced by name, with low = the lower layer that sends a Forward projection
// to the high layer, and receives a Back projection in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) BidirConnectLayerNames(low, high string, pat prjn.Pattern) (lowlay, highlay emer.Layer, fwdpj, backpj emer.Prjn, err error) {
	lowlay, err = nt.LayerByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayerByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, emer.Forward)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, emer.Back)
	return
}

// BidirConnectLayers establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) BidirConnectLayers(low, high emer.Layer, pat prjn.Pattern) (fwdpj, backpj emer.Prjn) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, emer.Back)
	return
}

// BidirConnectLayersPy establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *Network) BidirConnectLayersPy(low, high emer.Layer, pat prjn.Pattern) {
	nt.ConnectLayers(low, high, pat, emer.Forward)
	nt.ConnectLayers(high, low, pat, emer.Back)
}

// LateralConnectLayer establishes a self-projection within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayer(lay emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.LateralConnectLayerPrjn(lay, pat, pj)
}

// LateralConnectLayerPrjn makes lateral self-projection using given projection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayerPrjn(lay emer.Layer, pat prjn.Pattern, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(lay, lay, pat, emer.Lateral)
	lay.RecvPrjns().Add(pj)
	lay.SendPrjns().Add(pj)
	return pj
}

// Build constructs the layer and projection state based on the layer shapes
// and patterns of interconnectivity
func (nt *Network) Build() error {
	emsg := ""
	for li, ly := range nt.Layers {
		ly.SetIndex(li)
		if ly.IsOff() {
			continue
		}
		err := ly.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	nt.Layout()
	nt.BuildVarNames()
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (nt *Network) VarRange(varNm string) (min, max float32, err error) {
	first := true
	for _, ly := range nt.Layers {
		lmin, lmax, lerr := ly.VarRange(varNm)
		if lerr != nil {
			err = lerr
			return
		}
		if first {
			min = lmin
			max = lmax
			continue
		}
		if lmin < min {
			min = lmin
		}
		if lmax > max {
			max = lmax
		}
	}
	return
}

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	for li, ly := range nt.Layers {
		ly.Defaults()
		ly.SetIndex(li)
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// BuildVarNames makes the var names from states of network
func (nt *Network) BuildVarNames() {
	nt.LayVarNamesMap = make(map[string]int)
	nt.PrjnVarNamesMap = make(map[string]int)
	for _, lyi := range nt.Layers {
		ly := lyi.(*Layer)
		for nm := range ly.States {
			nt.LayVarNamesMap[nm] = 0
		}
		for _, pji := range ly.RcvPrjns {
			pj := pji.(*Prjn)
			for nm := range pj.States {
				nt.PrjnVarNamesMap[nm] = 0
			}
		}
	}
	nt.LayVarNames = make([]string, len(nt.LayVarNamesMap))
	i := 0
	for nm := range nt.LayVarNamesMap {
		nt.LayVarNames[i] = nm
		i++
	}
	sort.Strings(nt.LayVarNames)
	for i := range nt.LayVarNames {
		nt.LayVarNamesMap[nt.LayVarNames[i]] = i
	}

	nt.PrjnVarNames = make([]string, len(nt.PrjnVarNamesMap))
	i = 0
	for nm := range nt.PrjnVarNamesMap {
		nt.PrjnVarNames[i] = nm
		i++
	}
	sort.Strings(nt.PrjnVarNames)
	for i := range nt.PrjnVarNames {
		nt.PrjnVarNamesMap[nt.PrjnVarNames[i]] = i
	}
}

// UnitVarNames returns a list of variable names available on the units in this network.
// Not all layers need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) UnitVarNames() []string {
	return nt.LayVarNames
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return nil
}

// SynVarNames returns the names of all the variables on the synapses in this network.
// Not all projections need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) SynVarNames() []string {
	return nt.PrjnVarNames
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return nil
}

// WriteWtsJSON writes network weights (and any other state that adapts with learning)
// to JSON-formatted output.
func (nt *Network) WriteWtsJSON(w io.Writer) error {
	return nil
}

// ReadWtsJSON reads network weights (and any other state that adapts with learning)
// from JSON-formatted input.  Reads into a temporary weights.Network structure that
// is then passed to SetWts to actually set the weights.
func (nt *Network) ReadWtsJSON(r io.Reader) error {
	return nil
}

// SetWts sets the weights for this network from weights.Network decoded values
func (nt *Network) SetWts(nw *weights.Network) error {
	return nil
}

// SaveWtsJSON saves network weights (and any other state that adapts with learning)
// to a JSON-formatted file.  If filename has .gz extension, then file is gzip compressed.
func (nt *Network) SaveWtsJSON(filename gi.FileName) error {
	return nil
}

// OpenWtsJSON opens network weights (and any other state that adapts with learning)
// from a JSON-formatted file.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *Network) OpenWtsJSON(filename gi.FileName) error {
	return nil
}
