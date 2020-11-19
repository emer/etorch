# eTorch

eTorch is the emergent GUI and other support for PyTorch networks: provides a NetView for torch networks.

Each `etorch.Layer` stores the state variables as etensor.Float32, which are just copied via python code from the `torch.FloatTensor` state values recorded from running the network.

The `etor` python-side library provides a `State` object that handles the recording of state during the `forward` pass through a torch model.  You just need to call the `rec` method for each step that you want to record.  Then, when you want to update the `NetView`, you call `update` -- the `init_net` method called with the `torch.Network` configures everything so this update is then fully self-contained.

Here's the `forward` code for the `alexnet` example:

```Python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.est.rec(x, "Image.Act")
        for i, f in enumerate(self.features):
            x = f(x)
            self.est.rec(x, self.fnames[i])
        x = self.avgpool(x)
        self.est.rec(x, "AP.Act")
        x = torch.flatten(x, 1)
        cnames = ["", "Cl1.Net", "Cl1.Act", "", "Cl2.Net", "Cl2.Act", "Out.Net"]
        for i, f in enumerate(self.classifier):
            x = f(x)
            cnm = cnames[i]
            if len(cnm) > 0:
                self.est.rec(x, cnm)
        return x
```

Where `self.fnames` are the layer.variable names to save for each step in updating the `features` sequence, while `self.cnames` is the equivalent for the `classifier`.

As usual, the best way to see everything is to check out the `examples`:

* `etra25` is a torch version of the `leabra/examples/ra25` random associator, using all of the same overall program infrastructure.  This provides full training and testing control, weight visualization, etc, and is a good reference for various code examples.  It is very much Go-based in design, having been translated from Go source originally, so it may not be as familiar for typical Python users, but it does show what kind of overall complete GUI you can create.

* `alexnet` is the standard `torchvision` `alexnet` example, showing how large convolutional neural networks look with the visualization.  Because these models are so ... visual, you can really see what each step is doing.  This setup is only for testing, and doesn't show weights.

For now, use the `pyleabra` executable, which includes `etorch` to facilitate interoperability between leabra and torch models.  A separate `etorch` executable will also be configured soon.


