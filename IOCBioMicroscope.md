# Introduction #

This software package allows to deconvolve microscope images. In addition to the deconvolution program, the package includes of the set of tools that are required for processing the images, estimation of point spread function (PSF) and visualizing the results. This software is written in Python and is released with an open-source license (see below).

# Package tools #

The following tools are provided in the package iocbio.microscope:

  * [iocbio\_deconvolve](http://sysbio.ioc.ee/download/software/iocbio/generated/iocbio-deconvolve.html), [iocbio\_deconvolve\_with\_sphere](http://sysbio.ioc.ee/download/software/iocbio/generated/iocbio-deconvolve_with_sphere.html) --- these are scripts to deconvolve 3D microscope images with specified kernel (e.g. PSF that is estimated from the measurements of microspheres) or sphere (to improve the estimated PSF by taking into account the finite dimensions of microspheres).
  * [iocbio\_estimate\_psf](http://sysbio.ioc.ee/download/software/iocbio/generated/iocbio-estimate_psf.html) --- this is a script to compute estimated PSF from the measurements of microspheres.

For more information, see [IOCBio Documentation](http://sysbio.ioc.ee/download/software/iocbio/).

# Examples #

See [DeconvolutionTutorial](DeconvolutionTutorial.md)

# Installation #

Since _iocbio.microscope_ is written in Python language and uses various scientific computational modules and provides basic graphical interfaces, then the following software (with all their dependencies) must be installed to use IOCBio Microscope Software:

  * [Python](http://www.python.org/) - an easy to learn, powerful programming language.
  * [WX Python](http://www.wxpython.org/) - a GUI toolkit for the Python programming language.
  * [Numpy](http://www.numpy.org/) - the fundamental package needed for scientific computing with Python.
  * [Scipy](http://www.scipy.org/) - an open-source software for mathematics, science, and engineering.
  * [libtiff](http://www.libtiff.org/) - provides support for the Tag Image File Format (TIFF).
  * [pyfftw](http://www.launchpad.net/pyfftw) - Python bindings to the FFTW3 C-library.
  * [Matplotlib](http://matplotlib.sourceforge.net/) - a python 2D plotting library (optional).

The sources of iocbio.microscope is available via [SVN](http://code.google.com/p/iocbio/source/list) and HTTP. Note that iocbio.microscope requires Python 2.5 or newer. Python 2.5 users must also install the backport of the [multiprocessing](http://code.google.com/p/python-multiprocessing/) package.

## Windows users ##

To ease the process of installing required software and iocbio package for Windows users, we provide installers for various Python versions (if you don't have any installed, just pick one from the list):

  * [iocbio\_installer\_py26.exe](http://iocbio.googlecode.com/files/iocbio_installer_py26.exe)
  * [for other Python versions](http://sysbio.ioc.ee/download/software/binaries/latest/)

The installer should be run as Administrator. So, you must download the installer, save it to disk, and then right-click it and choose `"Run as administrator"` menu item.

When some required software component is not found then the iocbio installer can download and run the corresponding installers, just be ready to click Next, Next, ... and accept all default installation options. The following selection of software has prebuilt binaries available:
```
Python 2.6, Numpy 1.5, Scipy 0.9, Matplotlib 1.0, wx 2.9.
```

Finally, computer restart is required to finalise any PATH updates.

You will find a new folder `IOCBio Software` in your computers Desktop that contains links to iocbio.microscope scripts. Double-click on the links to start GUI of the corresponding program.

To get updates for IOCBio Software, rerun the iocbio installer and at the Iocbio configuration tab select `"Update and install iocbio from svn"`.

## Linux users ##

The following commands install all prerequisites as well as iocbio.microscope software for various Linux distributions:

Ubuntu/Debian:
```
sudo apt-get install python-dev python-numpy python-scipy python-wxgtk2.8 \
                     python-matplotlib libtiff4-dev libfftw3-dev gfortran
```

Other:
```
# Install python, numpy, scipy, wx Python bindings, matplotlib, tiff library,
# and fftw library according to the installation instructions of these software.
```

Finally, all Linux distributions should execute the following commands:
```
# Install pyfftw:
bzr branch lp:pyfftw
cd pyfftw
python setup.py install --prefix=/usr/local/
cd ..

# Only for Python 2.5 users, install multiprocessing:
svn checkout http://python-multiprocessing.googlecode.com/svn/trunk/ multiprocessing
cd multiprocessing
sudo python setup.py install
cd ..

# Install iocbio package:
svn checkout http://iocbio.googlecode.com/svn/trunk/ iocbio-read-only
cd iocbio-read-only
sudo python setup.py install
cd ..
```

To test iocbio.microscope installation, run, for example,
```
iocbio.deconvolve
```

## License and reference ##

The iocbio package is distributed using an open-source license (BSD). In addition to the terms of the license, we ask to acknowledge the use of the package in scientific articles by citing the following paper:

  * Laasmaa, M, Vendelin, M, Peterson, P (2011). Application of regularized Richardson-Lucy algorithm for deconvolution of confocal microscopy images. J. Microscopy. Volume 243,  [Issue 2](https://code.google.com/p/iocbio/issues/detail?id=2) , pages 124–140, August 2011: http://onlinelibrary.wiley.com/doi/10.1111/j.1365-2818.2011.03486.x/full
# Acknowledgments #

The iocbio software uses the following third-party software (included in the package):
  * [tifffile.py](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) - Read TIFF, STK, LSM and FluoView files and access image data as NumPy array.
  * [pylibtiff](http://code.google.com/p/pylibtiff/) - Wraps [libtiff](http://www.libtiff.org/) library to Python using [ctypes](http://docs.python.org/library/ctypes.html).
  * [killableprocess](http://benjamin.smedbergs.us/blog/2006-12-11/killableprocesspy/) - Extends [subprocess](http://docs.python.org/library/subprocess.html) such that the process and all of its sub-subprocesses are killed correctly, on Windows, Mac, and Linux.
  * Windows installers are build using [MinGW](http://www.mingw.org/) under [Ubuntu Linux](http://www.ubuntu.com/) ([howto](http://code.google.com/p/sympycore/wiki/BuildWindowsInstallerOnLinux)) and using library packages from the [GnuWin32](http://gnuwin32.sourceforge.net/) project. See iocbio/installer directory for the source code.