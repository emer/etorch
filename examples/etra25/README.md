# etra25

etra25 is the eTorch version of the leabra ra25 standard demo, using all of the emergent infrastructure, with only a pytorch model in the middle actually doing the computation.  This is not a "normal" way that people would use pytorch, but it makes an easy testing platform, and a good bridge between pytorch and the emergent model framework.

The start of etra25.py defines a simple `Feedforward` torch network, and ConfigNet initializes a parallel `etorch.Network` for holding the state values, which then drives the NetView display.  The `UpdateNetState` method calls `UpdateLayerState` to copy from the torch network into the etorch network.

All of the other infrastructure for the model is based on the Go framework, with buttons for stepping over the different levels, testing the network, etc.  The patterns are generated using the Go etable, and then copied to torch tensors, which are used for driving inputs to the torch network.  The specific input pattern is still controlled by the Go FixedTable environment -- only a single input pattern is presented at a time, in contrast to typical torch usage of presenting the entire epoch at a time in one big batch.  This allows full control over training vs. testing orders, etc.  All of that could presumably be done using more standard torch mechanisms.

A more typical use-case would be to only use the gui for testing, with training being the normal batch mode and not recorded or visualized.

