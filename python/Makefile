# Makefile for gopy pkg generation of python bindings to emergent

PYTHON=python3
PIP=$(PYTHON) -m pip

PBGV=`$(PIP) list | grep PyBindGen`

all: prereq gen

.PHONY: prereq gen all build install install-pkg install-exe clean

prereq:
	@echo "Installing go prerequisites:"
	- go get golang.org/x/tools/cmd/goimports  # this installs into ~/go/bin
	- go get github.com/go-python/gopy@0.4.0
	@echo "Installing python prerequisites -- ignore err if already installed:"
	- $(PIP) install -r requirements.txt
	@echo
	@echo "if this fails, you may see errors like this:"
	@echo "    Undefined symbols for architecture x86_64:"
	@echo "    _PyInit__gi, referenced from:..."
	@echo

install: install-pkg install-exe

# note: it is important that leabra come before deep otherwise deep captures all the common types
# unfortunately this means that all sub-packages need to be explicitly listed.
gen:
	gopy exe -name=etorch -vm=python3 -no-warn -exclude=driver,oswin,draw,example,examples,gif,jpeg,png,draw -main="runtime.LockOSThread(); gimain.Main(func() {  GoPyMainRun() })" math/rand image github.com/anthonynsimon/bild/transform github.com/goki/ki/ki github.com/goki/ki/kit github.com/goki/mat32  github.com/goki/gi/units github.com/goki/gi/gist github.com/goki/gi/girl github.com/goki/gi/gi github.com/goki/gi/svg github.com/goki/gi/giv github.com/goki/gi/gi3d github.com/goki/gi/gimain github.com/emer/etable github.com/emer/emergent github.com/emer/vision github.com/emer/etorch
	
build:
	$(MAKE) -C etorch build

install-pkg:
	# this does a local install of the package, building the sdist and then directly installing it
	# copy pyside/*.py etc to etorch so it will be installed along with rest
	cp pyside/*.py etorch/
	rm -rf dist build */*.egg-info *.egg-info
	$(PYTHON) setup.py sdist
	$(PIP) install dist/*.tar.gz

install-exe:
	# install executable into /usr/local/bin
	cp etorch/pyetorch /usr/local/bin/etorch

clean:
	rm -rf etorch dist build */*.egg-info *.egg-info
	
