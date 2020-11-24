# Python executable

To run `eTorch`, you need to build a version of Python that has the Go GUI and other infrastructure compiled into it.  Unfortunately, due to the demands for the GUI to run on the main thread, and the limited ability to coordinate that between Python and Go, we have to build a separate Python executable, instead of just loading libraries.  This Python executable works exactly like the official one, which itself is just a very thin wrapper around the same core libraries that we build with -- its the same thing, just with a different name.

The new executable you build here is called `etorch`, and it will be installed into your `/usr/local/bin` path, so you can just run it, just like you would run `python3`.

The tool that makes this all possible is [gopy](https://github.com/go-python/gopy), which automatically creates Python bindings for Go packages. 

See the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) for more details on how the python wrapper works and how to use it for GUI-level functionality.  **If you encounter any difficulties with this install, then try doing the install in GoGi first**, and read more about the different install issues there.

# Installation

First, you have to install the Go version of etorch:
```sh
$ cd <anywhere you want to install> # you can put the code anywhere
$ git clone https://github.com/emer/etorch   # makes an etorch directory 
$ cd etorch/examples/gotest
$ go build    # this installs all the dependencies for you!  do *not* type go get ./...
$ ./gotest &  # note: doesn't really do anything, but make sure it runs!
```

See also the emergent [Wiki Install](https://github.com/emer/emergent/wiki/Install) info.

Python version 3 (3.6, 3.8 have been well tested) is recommended.

This assumes that you are using go modules, as discussed in the wiki install page, and *that you are in the `etorch` directory where you installed etorch* (e.g., `git clone https://github.com/emer/etorch` and then `cd etorch`)

```sh
$ cd python    # should be in etorch/python now -- i.e., the dir where this README.md is..
$ make
$ make install  # may need to do: sudo make install -- installs into /usr/local/bin and python site-packages
$ cd ../examples/etra25
$ ./etra25.py   # runs using magic code on first line of file -- alternatively:
$ etorch -i etra25.py   # etorch was installed during make install into /usr/local/bin
```

The `etorch` executable combines standard python and the full Go emergent and GoGi gui packages -- see the information in the GoGi python readme for more technical information about this.

