# eTorch

eTorch provides the emergent GUI and other support for [PyTorch](https://pytorch.org) networks, including an interactive 3D NetView for visualizing network dynamics, and other GUI elements for controlling the model and plotting training and testing performance, etc.

The key idea for the NetView is that each `etorch.Layer` stores the state variables as a `etensor.Float32`, which are just copied via Python code from the `torch.FloatTensor` state values recorded from running the network.

The `etor` python-side library provides a `State` object that handles the recording of state during the `forward` pass through a torch model.  You just need to call the `rec` method for each step that you want to record. The `set_net` method is called with the `torch.Network` to record state to.

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
        for i, f in enumerate(self.classifier):
            x = f(x)
            cnm = self.cnames[i]
            if len(cnm) > 0:
                self.est.rec(x, cnm)
        return x
```

Where `self.fnames` are the layer.variable names to save for each step in updating the `features` sequence, while `self.cnames` is the equivalent for the `classifier`.

As usual, the best way to see everything is to check out the `examples`:

* `etra25` is a torch version of the `leabra/examples/ra25` random associator, using all of the same overall program infrastructure.  This provides full training and testing control, weight visualization, etc, and is a good reference for various code examples.  It is very much Go-based in design, having been translated from Go source originally, so it may not be as familiar for typical Python users, but it does show what kind of overall complete GUI you can create.

* `alexnet` is the standard `torchvision` `alexnet` example, showing how large convolutional neural networks look with the visualization.  Because these models are so ... visual, you can really see what each step is doing.  This setup is only for testing, and doesn't show weights.

![Screenshot of AlexNet example](alexnet_screen.png?raw=true "Screenshot of AlexNet example")

# Installation

See the https://github.com/emer/etorch/tree/main/python directory for instructions on building an `etorch` program that is just like the `python3` executable, but also includes all of the Go-based infrastructure that enables etorch to work.

You can also use the `pyleabra` executable from https://github.com/emer/leabra/tree/master/python, which includes `etorch` to facilitate interoperability between leabra and torch models.

# Interoperating between Go and Python

See [etable pyet](https://github.com/emer/etable/tree/master/examples/pyet) for example code for converting between the Go `etable.Table` and `numpy`, `torch`, and `pandas` table structures, using the `pyet` Python library that is installed with etorch.


