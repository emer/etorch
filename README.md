# eTorch

eTorch is the emergent GUI and other support for PyTorch networks: provides a NetView for torch networks.

Each `etorch.Layer` stores the state variables as etensor.Float32, which are just copied via python code from the torch FloatTensor state values recorded from running the network.

The most explicit way to get these state values from the torch network is to record them in tensors on the network object, during the `forward` function pass.  For other standard networks where you do not directly write the forward function, we can presumably find some way to extract state from internal torch representations (state dict or something?)

See `examples/etra25` for a torch version of the `leabra/examples/ra25` random associator, using all of the same overall program infrastructure.

For now, use the `pyleabra` executable, which includes `etorch` to facilitate interoperability between leabra and torch models.  A separate `etorch` executable will also be configured soon.


