# Introduction #

This software is a wrapper to [Strathkelvin 929 Oxygen System](http://www.strathkelvin.com/biomedical/928.asp) software that can measure the oxygen concentration in respiration cells.
While experiment is running, the IOCBio.StrathKelvin software captures oxygen measurements, runs analysis procedures, and displays the results in its GUI program.
The software is written in Python and can be easily extended with custom analysis procedures for live data. As an example, the software computes
and displays the respiration rates while acquiring oxygen concentration data.

# GUI program #

The IOCBio.StrathKelvin software provides a GUI program:
```
  iocbio.strathkelvin929
```
which can be accessed via `Strathkelvin929` shortcut in the `Desktop/IOCBio Software` folder.

# Installation #

Since _iocbio.strathkelvin_ is written in Python and uses various scientific computational modules and provides basic graphical interfaces, then the following software (with all their dependencies) must be installed to use IOCBio StrathKelvin software:

  * [Python](http://www.python.org/) - an easy to learn, powerful programming language.
  * [WX Python](http://www.wxpython.org/) - a GUI toolkit for the Python programming language.
  * [Numpy](http://www.numpy.org/) - the fundamental package needed for scientific computing with Python.
  * [Matplotlib](http://matplotlib.sourceforge.net/) - a python 2D plotting library.

The sources of iocbio.strathkelvin is available via [SVN](http://code.google.com/p/iocbio/source/list) and HTTP. Note that iocbio.strathkelvin requires Python 2.5 or newer.

To ease the process of installing required software for Windows users, below we provide direct links to the corresponding installers. The installers should be executed in the given order:

  * Python: http://iocbio.googlecode.com/files/python-2.6.2.msi
  * WX Python: http://iocbio.googlecode.com/files/wxPython2.8-win32-unicode-2.8.9.2-py26.exe
  * Numpy: http://iocbio.googlecode.com/files/numpy-1.4.0.dev6882.win32-py2.6.exe
  * Matplotlib: http://iocbio.googlecode.com/files/matplotlib-1.0.svn.win32-py2.6.exe
  * iocbio: http://iocbio.googlecode.com/files/iocbio-1.2.0.dev139.win32-py2.6.exe (run this installer program as an Administrator, otherwise you might need to edit PATH environment variable manually)
  * and finally, restart your computer.

You will find a new folder "IOCBio Software" in your computers Desktop that contains a shortcut to iocbio.strathkelvin929 script. Double-click on the Strathkelvin929 to start the GUI program.