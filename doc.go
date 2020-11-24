// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
eTorch provides the emergent GUI and other support for PyTorch https://pytorch.org
networks, including an interactive 3D NetView for visualizing network dynamics,
and other GUI elements for controlling the model and plotting training
and testing performance, etc.

The key idea for the NetView is that each `etorch.Layer` stores the state variables
as a `etensor.Float32`, which are just copied via Python code from the
torch.FloatTensor` state values recorded from running the network.

The `etor` python-side library provides a `State` object that handles
the recording of state during the `forward` pass through a torch model.
You just need to call the `rec` method for each step that you want to record.
The `set_net` method is called with the `torch.Network` to record state to.
*/
package etorch
