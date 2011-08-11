
Installer tools
===============

Overview
--------

This directory contains Python modules for building a installer for
iocbio software and all of its dependencies (Python, Numpy, Scipy,
Matplotlib, Wx, etc).  The installer is designed for Windows users to
ease the iocbio software installation process. However, the model of
installer is general and it should be easy to support other OS-s as
well.

The installer consists of a Python script that is turned to an EXE
file using the pyinstaller program (http://www.pyinstaller.org/).  The
installer script depends on Python (obviously) and wx
(http://www.wxpython.org/) that is used for GUI. The installer can be
built under Wine (http://www.winehq.org/) using the Mingw
(http://www.mingw.org/) tools.

Quick reference
---------------

To build iocbio installer, run

  ./wineit.sh make_iocbio_installer.bat

that will create a file iocbio_installer_py26.exe, for
instance. Before running the script, you have to install wine. The
make_iocbio_installer.bat will download and install other components
automatically. So, be ready to step through installer wizards of
Python, wx, and pyinstaller programs. Note that the iocbio installer
file depends on the Python version. 

To build iocbio installer for other Python versions, edit the header
of the make_iocbio_installer.sh file accordingly.

To test the iocbio installer within wine, run

  ./wineit2.sh iocbio_installer_py26.exe

The installer can be also tested under Linux environment by running

  python iocbio_installer.py

that will fireup a GUI window and you can step through all the
software components.

The ultimate installer test should be carried out under Windows.

Notes
-----

For using mingw-light, one must run

  make_iocbio_mingw.sh

and copy the resulting file (say, mingw-20110802-light.zip) to

  /net/cens/home/www/sysbio/download/software/binaries/latest
