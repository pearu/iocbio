Original document: http://code.google.com/p/sympycore/wiki/BuildWindowsInstallerOnLinux

# Introduction #

This document describes how to build windows installers of Python packages on Linux
using [Wine](http://www.winehq.org/) and [MinGW](http://www.mingw.org/).

The instructions are developed under Ubuntu Linux (Lucid) should work with minor modifications for other Linux systems as well.

# Prerequisites #

Install wine, Python, MinGW, and Make:
```
sudo apt-get install wine
mkdir wine && cd wine
wget http://www.python.org/ftp/python/2.6.5/python-2.6.5.msi
msiexec /i python-2.6.5.msi
wget http://downloads.sourceforge.net/project/mingw/Automated%20MinGW%20Installer/MinGW%205.1.6/MinGW-5.1.6.exe
wine MinGW-5.1.6.exe
#IMPORTANT: Make sure that g++, g77 compilers are checked for install.
wget http://downloads.sourceforge.net/project/gnuwin32/make/3.81/make-3.81.exe
wine make-3.81.exe
```

Fix wine PATH environment variable:
```
regedit
# Lookup "PATH" environment variable and append ";C:\Python26;C:\MinGW\bin;C:\gtk\bin;C:\Program Files\GnuWin32\bin" to it.
# The path to "PATH" is "HKEY_LOCAL_MACHINE/System/CurrentControlSet/Control/Session Manager/Environment/PATH".
```

# wxPython #

```
wget http://downloads.sourceforge.net/wxpython/wxPython2.8-win32-unicode-2.8.11.0-py26.exe
wine wxPython2.8-win32-unicode-2.8.11.0-py26.exe
```

# Numpy #

```
svn co http://svn.scipy.org/svn/numpy/trunk numpy
cd numpy
PYTHONPATH= wine python setup.py build --compiler=mingw32 bdist_wininst
wine dist/numpy-2.0.0.dev8445.win32-py2.6.exe
```

# LAPACK #

```
mkdir -p ~/src && cd ~/src
# lapack-3.2 and newer require a Fortran 90 compiler.
# we are using g77 since gfortran is not easily available in mingw currently
wget http://www.netlib.org/lapack/lapack-3.1.1.tgz
tar xzf lapack-3.1.1.tgz
cd lapack-3.1.1
wget http://iocbio.googlecode.com/files/make.inc.MINGW-g77
cp make.inc.MINGW-g77 make.inc
cd SRC && wine make -j 5
mv lapack_MINGW.a libflapack.a
```

# Scipy #

```
export LAPACK=~/src/lapack-3.1.1
export BLAS_SRC=/path/to/blas
svn co http://svn.scipy.org/svn/scipy/trunk scipy
cd scipy
PYTHONPATH= wine python setup.py build --compiler=mingw32 bdist_wininst
# You might need to rerun the build command several times when
# it fails with the following error:
#   g77.exe: C:/MinGW/bin/../lib/gcc/mingw32/3.4.5/specs: No such file or directory
```

# GTK/pygtk/pycairo/pygobject #

```
cd ~/.wine/drive_c
mkdir gtk
cd gtk
wget http://ftp.gnome.org/pub/gnome/binaries/win32/gtk+/2.16/gtk+-bundle_2.16.6-20100207_win32.zip
unzip gtk+-bundle_2.16.6-20100207_win32.zip

cd ~/wine
wget http://ftp.gnome.org/pub/GNOME/binaries/win32/pygtk/2.16/pygtk-2.16.0+glade.win32-py2.6.exe
wget http://ftp.gnome.org/pub/GNOME/binaries/win32/pygobject/2.20/pygobject-2.20.0.win32-py2.6.exe
wget http://ftp.gnome.org/pub/GNOME/binaries/win32/pycairo/1.8/pycairo-1.8.6.win32-py2.6.exe
wine pygtk-2.16.0+glade.win32-py2.6.exe
wine pycairo-1.8.6.win32-py2.6.exe
wine pygobject-2.20.0.win32-py2.6.exe
```

# Matplotlib #

```
cd ~/svn
svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib
cd matplotlib
# Apply matplotlib-wine.patch that adds wine-win32 support to setupext.py:
wget http://iocbio.googlecode.com/files/matplotlib-wine.patch
# ...
PYTHONPATH= wine python setup.py build --compiler=mingw32 bdist_wininst
```

# PyFFTW3 #

```
mkdir ~/bzr
cd bzr && bzr branch lp:pyfftw
cd pyfftw
wget ftp://ftp.fftw.org/pub/fftw/fftw-3.2.2.pl1-dll32.zip
unzip fftw-3.2.2.pl1-dll32.zip libfftw3-3.dll libfftw3f-3.dll libfftw3l-3.dll
cp *.dll src/templates/
PYTHONPATH= wine python setup.py bdist_wininst
```

# IOCBio #

```
cd ~/svn
svn checkout http://iocbio.googlecode.com/svn/trunk/ iocbio
cd iocbio
./mk_wininstaller.sh 
```

# Prebuilt binaries #

Binaries that are built according to these instructions are available in
http://sysbio.ioc.ee/download/software/binaries/