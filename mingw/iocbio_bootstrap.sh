#!/c/MinGW/msys/1.0/bin/bash.exe
#
# A self-contained script for Wine or Windows to download, build and
# install Python, numpy, scipy, matplotlib and other packages.
#
# Usage for Linux/wine:
#
#   sudo apt-get install wine
#   wget http://sourceforge.net/projects/mingw/files/Automated%20MinGW%20Installer/mingw-get-inst/mingw-get-inst-20100909/mingw-get-inst-20100909.exe
#   wine mingw-get-inst-20100909.exe                   # takes about 1 min 20 sec depending on Internet connection speed
#
#   wine c:/MinGW/bin/mingw-get.exe install msys-bash  # takes about 30 sec
#   wineconsole c:/MinGW/msys/1.0/bin/bash
#   ./iocbio_bootstrap.sh                              # takes about 1.5h and 300MB
#
# Usage for Windows (XP):
#
#   Install the following libraries and software:
#     http://download.microsoft.com/download/vc60pro/update/1/w9xnt4/en-us/vc6redistsetup_enu.exe
#     http://download.microsoft.com/download/d/d/9/dd9a82d0-52ef-40db-8dab-795376989c03/vcredist_x86.exe
#     http://sourceforge.net/projects/mingw/files/Automated%20MinGW%20Installer/mingw-get-inst/mingw-get-inst-20100909/mingw-get-inst-20100909.exe
#   Save iocbio_bootstrap.sh script to c:\
#   For a clean install, make sure that the following directories are (re)moved:
#     c:\gtk
#     c:\Python26
#     c:\MinGW
#   Run from command propmt:
#      c:\MinGW\msys\1.0\bin\bash
#   Run from bash shell:
#      /c/iocbio_bootstrap.sh
#
# Be ready to click Next, Next.. and pay attention to IMPORTANT output messages.
# Building some libraries can take some time, so be patient.
# The script can be re-run after fixing any occuring problems.
# Finally, fix the PATH environment variable according to the finishing instructions.
#
# Author: Pearu Peterson
# Created: October 2010
#
# TODO: pyfftw3 installer, test iocbio.deconvolve using fftw3

export NUMPY_VERSION=1.5.1 # 1.5.1, 1.4.1, 1.3.0, py2.5: 1.2.1, 1.1.1, 1.0.4
export SCIPY_VERSION=0.9.0 # 0.9.0, 0.8.0, 0.7.2
export LAPACK_VERSION=3.2.2
export MATPLOTLIB_VERSION=1.0.0 # 1.0.1, 1.0.0
export MATPLOTLIB_DIR_VERSION=1.0 # 1.0.1, 1.0
export PYTHON_VERSION=2.6.6 # [3.2], 3.1.3, 3.0.1, [2.7.1], 2.6.6, 2.5.5, 2.4.6, 2.3.7
export IOCBIO_VERSION=svn # svn, 1.2.0.dev139

NPVER=${NUMPY_VERSION::3}
PYVER=${PYTHON_VERSION:0:3}
PYVR=${PYTHON_VERSION:0:1}${PYTHON_VERSION:2:1}

echo "NPVER=$NPVER, PYVER=$PYVER, PYVR=$PYVR"

ROOT=tmp/iocbio_bootstrap_cache
GTK_ROOT=/c/gtk
export FFTW3="c:\fftw3"
REQUIREDPATH=/c/MinGW/bin:/c/MinGW/msys/1.0/bin:/c/Python$PYVR:/c/Python$PYVR/Scripts:$GTK_ROOT/bin:/c/Program\ Files/Subversion/bin:/c/Program\ Files/GnuWin32/bin:/c/fftw3
WINREQUIREDPATH="c:\MinGW\bin;c:\MinGW\msys\1.0\bin;c:\Python$PYVR;c:\Python$PYVR\Scripts;c:\gtk\bin;c:\Program Files\Subversion\bin;c:\Program Files\GnuWin32\bin:c:\fftw3"
export PATH=/c/windows/command:$PATH:$REQUIREDPATH
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/c/Python$PYVR/Lib/pkgconfig:/c/gtk/lib/pkgconfig

if ( which mkdir && exit 1 || exit 0 ) ; then

which mkdir || mingw-get install msys-core msys-coreutils msys-wget msys-unzip gfortran  g++ binutils msys-patch msys-tar msys-make
echo "Checking the availability of commands.."
which mkdir || mingw-get install msys-core msys-coreutils
which start || exit 1
which wget || mingw-get install msys-wget
which unzip || mingw-get install msys-unzip
which make || mingw-get install msys-make
which gfortran || mingw-get install gfortran
which g++ || mingw-get install g++
which ar || mingw-get install binutils
which patch || mingw-get install msys-patch
which tar || mingw-get install msys-tar
echo "done"

fi

echo "PATH=$PATH"
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"


mkdir -p $ROOT
cd $ROOT
echo "PWD=`pwd`"

VCREDIST_INSTALLER=http://download.microsoft.com/download/d/d/9/dd9a82d0-52ef-40db-8dab-795376989c03/vcredist_x86.exe
MSVCP90_DLL=/c/windows/system32/msvcp90.dll
MSVCR90_DLL=/c/windows/system32/msvcr90.dll

VC6REDIST_INSTALLER=http://download.microsoft.com/download/vc60pro/update/1/w9xnt4/en-us/vc6redistsetup_enu.exe
MSVCP60_DLL=/c/windows/system32/msvcp60.dll

PYTHON_INSTALLER=http://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION.msi

BLAS_TARBALL=http://www.netlib.org/blas/blas.tgz

LAPACK_TARBALL=http://www.netlib.org/lapack/lapack-$LAPACK_VERSION.tgz

export BLAS=`pwd`/BLAS/libfblas.a
export LAPACK=`pwd`/lapack-$LAPACK_VERSION/libflapack.a


FFTW_ZIP=ftp://ftp.fftw.org/pub/fftw/fftw-3.2.2.pl1-dll32.zip

NUMPY_INSTALLER=http://sourceforge.net/projects/numpy/files/NumPy/$NUMPY_VERSION/numpy-$NUMPY_VERSION-win32-superpack-python$PYVER.exe

GTK_ZIP=http://ftp.acc.umu.se/pub/gnome/binaries/win32/gtk+/2.22/gtk+-bundle_2.22.0-20101016_win32.zip

PYGTK_INSTALLER=http://ftp.gnome.org/pub/GNOME/binaries/win32/pygtk/2.16/pygtk-2.16.0+glade.win32-py$PYVER.exe

PYGOBJECT_INSTALLER=http://ftp.gnome.org/pub/GNOME/binaries/win32/pygobject/2.20/pygobject-2.20.0.win32-py$PYVER.exe

PYCAIRO_INSTALLER=http://ftp.gnome.org/pub/GNOME/binaries/win32/pycairo/1.8/pycairo-1.8.6.win32-py$PYVER.exe

SUBVERSION_INSTALLER=http://downloads.sourceforge.net/project/win32svn/1.6.13/Setup-Subversion-1.6.13.msi

WXPYTHON_INSTALLER=http://sourceforge.net/projects/wxpython/files/wxPython/2.9.1.1/wxPython2.9-win32-2.9.1.1-py$PYVR.exe

SETUPTOOLS_INSTALLER=http://pypi.python.org/packages/$PYVER/s/setuptools/setuptools-0.6c11.win32-py$PYVER.exe

MATPLOTLIB_PATCH=http://iocbio.googlecode.com/files/matplotlib-$MATPLOTLIB_VERSION.patch
MATPLOTLIB_TARBALL=http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-$MATPLOTLIB_DIR_VERSION/matplotlib-$MATPLOTLIB_VERSION.tar.gz
MATPLOTLIB_INSTALLER=matplotlib-$MATPLOTLIB_VERSION/dist/matplotlib-$MATPLOTLIB_VERSION.win32-py$PYVER.exe
NEW_MATPLOTLIB_INSTALLER=matplotlib-$MATPLOTLIB_VERSION/dist/matplotlib-$MATPLOTLIB_VERSION.win32-py$PYVER-numpy$NPVER.exe

SCIPY_TARBALL=http://sourceforge.net/projects/scipy/files/scipy/$SCIPY_VERSION/scipy-$SCIPY_VERSION.tar.gz
SCIPY_INSTALLER=scipy-$SCIPY_VERSION/dist/scipy-$SCIPY_VERSION.win32-py$PYVER.exe
NEW_SCIPY_INSTALLER=scipy-$SCIPY_VERSION/dist/scipy-$SCIPY_VERSION.win32-py$PYVER-numpy$NPVER.exe

IOCBIO_SVN_PATH=http://iocbio.googlecode.com/svn/trunk/
IOCBIO_TARBALL=http://iocbio.googlecode.com/files/iocbio-$IOCBIO_VERSION.tar.gz

if [ "$IOCBIO_VERSION" == "svn" ] ; then
  export IOCBIO_SRC_PATH=iocbio
else
  export IOCBIO_SRC_PATH=iocbio-$IOCBIO_VERSION
fi

TIFF_INSTALLER=http://pylibtiff.googlecode.com/files/tiff-3.8.2-1.exe
PYLIBTIFF_SVN_PATH=http://pylibtiff.googlecode.com/svn/trunk/
LIBTIFF_DLL=/c/Program\ Files/GnuWin32/bin/libtiff3.dll

SYMPYCORE_SVN_PATH=http://sympycore.googlecode.com/svn/trunk/

# create paths
test -d $GTK_ROOT || (mkdir -p $GTK_ROOT)

# download files
test -f `basename $PYTHON_INSTALLER` || (wget $PYTHON_INSTALLER) || exit 1
test -f `basename $NUMPY_INSTALLER` || (wget $NUMPY_INSTALLER) || exit 1
test -f `basename $VC6REDIST_INSTALLER` || (wget $VC6REDIST_INSTALLER) || exit 1
test -f `basename $VCREDIST_INSTALLER` || (wget $VCREDIST_INSTALLER) || exit 1
test -f `basename $WXPYTHON_INSTALLER` || (wget $WXPYTHON_INSTALLER) || exit 1
test -f `basename $PYGTK_INSTALLER` || (wget $PYGTK_INSTALLER) || exit 1
test -f `basename $PYGOBJECT_INSTALLER` || (wget $PYGOBJECT_INSTALLER) || exit 1
test -f `basename $PYCAIRO_INSTALLER` || (wget $PYCAIRO_INSTALLER) || exit 1
test -f `basename $SETUPTOOLS_INSTALLER` || (wget $SETUPTOOLS_INSTALLER) || exit 1
test -f `basename $SUBVERSION_INSTALLER` || wget $SUBVERSION_INSTALLER || exit 1
test -f `basename $TIFF_INSTALLER` || wget $TIFF_INSTALLER || exit 1

test -f `basename $BLAS_TARBALL` || wget $BLAS_TARBALL || exit 1
test -f `basename $LAPACK_TARBALL` || wget $LAPACK_TARBALL || exit 1
test -f `basename $SCIPY_TARBALL` || wget $SCIPY_TARBALL || exit 1

if [ "$IOCBIO_VERSION" != "svn" ] ; then
test -f `basename $IOCBIO_TARBALL` || wget $IOCBIO_TARBALL || exit 1
fi
test -f `basename $MATPLOTLIB_TARBALL` || wget $MATPLOTLIB_TARBALL || exit 1
test -f $GTK_ROOT/`basename $GTK_ZIP` || (cd $GTK_ROOT && wget $GTK_ZIP) || exit 1

test -d $FFTW3 || mkdir $FFTW3 || exit 1
test -f $FFTW3/`basename $FFTW_ZIP` || (cd $FFTW3 && wget $FFTW_ZIP) || exit 1
test -f $FFTW3/fftw3.h || (cd $FFTW3 && unzip `basename $FFTW_ZIP`) || exit 1
test -f $FFTW3/libfftw3.lib || (cd $FFTW3 && cp libfftw3-3.dll libfftw3.dll && touch libfftw3.lib) || exit 1

# unpack files
test -d BLAS || tar xzf `basename $BLAS_TARBALL`  || exit 1
test -d lapack-$LAPACK_VERSION || tar xzf `basename $LAPACK_TARBALL` || exit 1
test -d scipy-$SCIPY_VERSION || tar xzf `basename $SCIPY_TARBALL` || exit 1
if [ "$IOCBIO_VERSION" != "svn" ] ; then
  test -d $IOCBIO_SRC_PATH || tar xzf `basename $IOCBIO_TARBALL` || exit 1
fi

test -d matplotlib-$MATPLOTLIB_VERSION || tar xzf `basename $MATPLOTLIB_TARBALL` || exit 1
test -f matplotlib-$MATPLOTLIB_VERSION/`basename $MATPLOTLIB_PATCH` || (cd matplotlib-$MATPLOTLIB_VERSION && wget $MATPLOTLIB_PATCH && patch -p0 < `basename $MATPLOTLIB_PATCH`) || exit 1

test -d $GTK_ROOT/bin && echo $GTK_ROOT || (cd $GTK_ROOT && unzip `basename $GTK_ZIP`)  || exit 1

# build libraries
test -f $BLAS && echo $BLAS || (echo "Building $BLAS" && cd BLAS && gfortran -fno-second-underscore -O2 -c *.f && ar r libfblas.a *.o && ranlib libfblas.a && exit 0)  || exit 1

test -f lapack-$LAPACK_VERSION/make.inc.MINGW || (cd lapack-$LAPACK_VERSION && wget  http://iocbio.googlecode.com/files/make.inc.MINGW && cp make.inc.MINGW make.inc) || exit 1
test -f $LAPACK && echo $LAPACK || (echo "Building $LAPACK" && cd lapack-$LAPACK_VERSION/SRC && make && mv ../lapack_MINGW.a ../libflapack.a)  || exit 1


# check for MSVC DLL files:
test -f "$MSVCP60_DLL" && echo "$MSVCP60_DLL" || (echo "IMPORTANT: Specify c:\ when asked for the location of extracted files." && ./`basename $VC6REDIST_INSTALLER` && (test -f /c/vcredist.exe &&  (/c/vcredist.exe || (echo "Ignoring crash" ) || exit 1)) && ./`basename $VCREDIST_INSTALLER`) || exit 1
test -f "$MSVCP60_DLL" || (echo "$MSVCP60_DLL does not exist" && exit 1) || exit 1

# install Python and numpy and others
which svn || start ./`basename $SUBVERSION_INSTALLER` || exit 1

while (sleep 5 && (which svn || exit 0 && exit 1)); do echo -e "Waiting another 5 secs for svn to become available.\nPlease, finish Subversion Setup..."; done
svn --version || (echo "Something wrong with svn installation") || exit 1

which python || (start `basename $PYTHON_INSTALLER`) || exit 1

while (sleep 5 && (python -c "import sys; print sys.version" || exit 0 && exit 1)); do echo -e "Waiting another 5 secs for Python to become available.\nPlease, finish Python Setup..."; done

test -f "$LIBTIFF_DLL" && echo "$LIBTIFF_DLL" || ./`basename $TIFF_INSTALLER` || exit 1

if [ "$IOCBIO_VERSION" == "svn" ] ; then
  test -d $IOCBIO_SRC_PATH || svn checkout $IOCBIO_SVN_PATH iocbio
  IOCBIO_VERSION=`cd $IOCBIO_SRC_PATH && python -c 'import setup; print "%(MAJOR)s.%(MINOR)s.%(MICRO)s" % (setup.__dict__)'`
  IOCBIO_SVNVERSION=`cd $IOCBIO_SRC_PATH && svnversion`
  IOCBIO_SVNVERSION=${IOCBIO_SVNVERSION%M}
  IOCBIO_SVNVERSION=${IOCBIO_SVNVERSION%:}
  echo "IOCBIO_VERSION=$IOCBIO_VERSION, IOCBIO_SVNVERSION=$IOCBIO_SVNVERSION"
  IOCBIO_INSTALLER=iocbio/dist/iocbio-$IOCBIO_VERSION.dev$IOCBIO_SVNVERSION.win32-py$PYVER.exe
  NEW_IOCBIO_INSTALLER=iocbio/dist/iocbio-$IOCBIO_VERSION.dev$IOCBIO_SVNVERSION.win32-py$PYVER-numpy$NPVER.exe
else
  IOCBIO_INSTALLER=$IOCBIO_SRC_PATH/dist/iocbio-$IOCBIO_VERSION.win32-py$PYVER.exe
  NEW_IOCBIO_INSTALLER=$IOCBIO_SRC_PATH/dist/iocbio-$IOCBIO_VERSION.win32-py$PYVER-numpy$NPVER.exe
fi

test -d pylibtiff || svn checkout $PYLIBTIFF_SVN_PATH pylibtiff
PYLIBTIFF_VERSION=`cd pylibtiff/libtiff && python -c 'import version'`
PYLIBTIFF_VERSION=${PYLIBTIFF_VERSION%M}
echo "PYLIBTIFF_VERSION=$PYLIBTIFF_VERSION"
PYLIBTIFF_INSTALLER=pylibtiff/dist/pylibtiff-$PYLIBTIFF_VERSION.win32-py$PYVER.exe
NEW_PYLIBTIFF_INSTALLER=pylibtiff/dist/pylibtiff-$PYLIBTIFF_VERSION.win32-py$PYVER-numpy$NPVER.exe

test -d sympycore || svn checkout $SYMPYCORE_SVN_PATH sympycore
SYMPYCORE_VERSION=0.2-svn
SYMPYCORE_SVNVERSION=`cd sympycore && svnversion`
SYMPYCORE_SVNVERSION=${SYMPYCORE_SVNVERSION%M}
echo "SYMPYCORE_VERSION=$SYMPYCORE_VERSION, SYMPYCORE_SVNVERSION=$SYMPYCORE_SVNVERSION"
SYMPYCORE_INSTALLER=SYMPYCORE/dist/sympycore-$SYMPYCORE_VERSION.win32-py$PYVER.exe
NEW_SYMPYCORE_INSTALLER=sympycore/dist/sympycore-$SYMPYCORE_VERSION$SYMPYCORE_SVNVERSION.win32-py$PYVER.exe

python -c 'import wx' || ./`basename $WXPYTHON_INSTALLER` || exit 1
python -c 'import wx; print "wx:",wx.__version__' || exit 1

python -c 'import pygtk' || ./`basename $PYGTK_INSTALLER` || exit 1
python -c 'import gobject' || ./`basename $PYGOBJECT_INSTALLER` || exit 1
python -c 'import cairo' || ./`basename $PYCAIRO_INSTALLER` || exit 1
python -c 'import gtk; print "gtk:",".".join(map(str,gtk.gtk_version))' || exit 1

which easy_install || ./`basename $SETUPTOOLS_INSTALLER` || exit 1
python -c 'import nose' || easy_install nose || exit 1
python -c 'import nose; print nose' || exit 1

python -c 'import numpy, os; v=os.environ["NUMPY_VERSION"]; assert numpy.__version__==v,repr((numpy.__version__,v))' || (rm -rf /c/Python$PYVT/Lib/site-packages/numpy; ./`basename $NUMPY_INSTALLER`)  || exit 1
python -c 'import numpy; print "numpy:",numpy.__version__' || exit 1

# build installers
BUILD_OPTS="--compiler=mingw32 --build-base=build-numpy-$NPVER"

test -f $NEW_SCIPY_INSTALLER || (cd scipy-$SCIPY_VERSION && python setup.py build $BUILD_OPTS bdist_wininst)  || exit 1
test -f $NEW_SCIPY_INSTALLER || mv $SCIPY_INSTALLER $NEW_SCIPY_INSTALLER || exit 1
test -f $NEW_MATPLOTLIB_INSTALLER || (cd matplotlib-$MATPLOTLIB_VERSION && python setup.py build $BUILD_OPTS bdist_wininst)  || exit 1
test -f $NEW_MATPLOTLIB_INSTALLER || mv $MATPLOTLIB_INSTALLER $NEW_MATPLOTLIB_INSTALLER || exit 1
test -f $NEW_IOCBIO_INSTALLER || (cd $IOCBIO_SRC_PATH && python setup.py build $BUILD_OPTS bdist_wininst)  || exit 1
test -f $NEW_IOCBIO_INSTALLER || mv $IOCBIO_INSTALLER $NEW_IOCBIO_INSTALLER || exit 1
test -f $NEW_PYLIBTIFF_INSTALLER || (cd pylibtiff && python setup.py build $BUILD_OPTS bdist_wininst)  || exit 1
test -f $NEW_PYLIBTIFF_INSTALLER || mv $PYLIBTIFF_INSTALLER $NEW_PYLIBTIFF_INSTALLER || exit 1

test -f $NEW_SYMPYCORE_INSTALLER || (cd sympycore && python setup.py build $BUILD_OPTS bdist_wininst)  || exit 1
test -f $NEW_SYMPYCORE_INSTALLER || mv $SYMPYCORE_INSTALLER $NEW_SYMPYCORE_INSTALLER || exit 1

# execute installers
python -c 'import scipy' || ./$NEW_SCIPY_INSTALLER  || exit 1
python -c 'import scipy; print "scipy:",scipy.__version__' || exit 1
python -c 'import matplotlib' || ./$NEW_MATPLOTLIB_INSTALLER || exit 1
python -c 'import matplotlib; print "matplotlib:", matplotlib.__version__' || exit 1
python -c 'import iocbio' || ./$NEW_IOCBIO_INSTALLER  || exit 1
python -c 'import iocbio.version as m; print "iocbio:",m.version' || exit 1

python -c 'import sympycore' || ./$NEW_SYMPYCORE_INSTALLER  || exit 1
python -c 'import sympycore as m; print "sympycore:",m.__version__' || exit 1

python -c 'import libtiff' || ./$NEW_PYLIBTIFF_INSTALLER  || exit 1
python -c 'import libtiff.version as m; print "libtiff:",m.version' || echo "Ignoring failure"

echo
echo "CONGRATULATIONS!!"
echo "You have succesfully installed Python, numpy, scipy, matplotlib, and friends!"
echo
echo "To finalize, make sure that the Windows environment variable PATH contains the following paths:"
echo
echo "  $WINREQUIREDPATH"
echo
echo "To update the environment PATH, open regedit (for instance) and go to"
echo
echo "  HKEY_LOCAL_MACHINE/System/CurrentControlSet/Control/Session Manager/Environment/PATH"
echo
echo "Note for wine&matplotlib users: matplotlib.use('WX')"
echo
echo "To save space, you might want to remove installers and sources from the directory $ROOT"
echo "Bye"