# Python executable

To run `eTorch`, you need to build a version of Python that has the Go GUI and other infrastructure compiled into it.  Unfortunately, due to the demands for the GUI to run on the main thread, and the limited ability to coordinate that between Python and Go, we have to build a separate Python executable, instead of just loading libraries.  This Python executable works exactly like the official one, which itself is just a very thin wrapper around the same core libraries that we build with -- its the same thing, just with a different name.

The new executable you build here is called `etorch`, and it will be installed into your `/usr/local/bin` path, so you can just run it, just like you would run `python3`.

The tool that makes this all possible is [gopy](https://github.com/go-python/gopy), which automatically creates Python bindings for Go packages. 

See the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) for more details on how the python wrapper works and how to use it for GUI-level functionality.

Note: **You must follow the installation instructions in the above GoGi Python README** to install the `gopy` program prior to running the further installation instructions below.  Given that etorch depends fully on GoGi, doing this first ensures that everything is all working prior to moving on to etorch itself.

# Installation

First, you have to install the Go version of emergent: [Wiki Install](https://github.com/emer/emergent/wiki/Install), and follow the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) installation instructions, and make sure everything works with the standard GoGi `widgets` example.

Python version 3 (3.6, 3.8 have been well tested) is recommended.

This assumes that you are using go modules, as discussed in the wiki install page, and *that you are in the `leabra` directory where you installed leabra* (e.g., `git clone https://github.com/emer/leabra` and then `cd leabra`)

```sh
$ cd python    # should be in leabra/python now -- i.e., the dir where this README.md is..
$ make
$ make install  # may need to do sudo make install -- installs into /usr/local/bin and python site-packages
$ cd ../examples/ra25
$ pyleabra -i ra25.py   # pyleabra was installed during make install into /usr/local/bin
```

* The `pyleabra` executable combines standard python and the full Go emergent and GoGi gui packages -- see the information in the GoGi python readme for more technical information about this.

