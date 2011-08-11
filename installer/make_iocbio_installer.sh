
# Author: Pearu Peterson
# Created: Apr 2011

export ARCH=32 # 32, 64
export PYTHON_VERSION=2.6.6 # [3.2], 3.1.3, 3.0.1, [2.7.1], 2.6.6, 2.5.4, 2.4.4, 2.3.7
export PYINSTALLER_VERSION=1.5 # 1.5, 1.5-rc1, 1.4
export WXPYTHON_VERSION=2.9.1.1 # 2.9.1.1, 2.8.10.1 (for py2.4)

PYVER=${PYTHON_VERSION:0:3}
PYVR=${PYTHON_VERSION:0:1}${PYTHON_VERSION:2:1}
WXVER=${WXPYTHON_VERSION:0:3}

PYINSTALLER_ZIP=http://www.pyinstaller.org/static/source/$PYINSTALLER_VERSION/pyinstaller-$PYINSTALLER_VERSION.zip

if [ "$ARCH" == "64" ] ; then
    PYTHON_INSTALLER=http://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION.amd64.msi
    PYWIN32_INSTALLER=http://sourceforge.net/projects/pywin32/files/pywin32/Build216/pywin32-216.win-amd64-py$PYVER.exe
else
    PYTHON_INSTALLER=http://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION.msi
    PYWIN32_INSTALLER=http://sourceforge.net/projects/pywin32/files/pywin32/Build216/pywin32-216.win32-py$PYVER.exe
fi
if [ "$WXPYTHON_VERSION" == "2.8.10.1" ] ;  then
    WXPYTHON_INSTALLER=http://sourceforge.net/projects/wxpython/files/wxPython/2.8.10.1/wxPython2.8-win32-unicode-2.8.10.1-py$PYVR.exe
else
    WXPYTHON_INSTALLER=http://sourceforge.net/projects/wxpython/files/wxPython/$WXPYTHON_VERSION/wxPython$WXVER-win32-$WXPYTHON_VERSION-py$PYVR.exe
fi

START_EXE=/c/windows/command/start.exe
START_EXE="c:\\windows\\command\\start.exe"
test -f $START_EXE || START_EXE="start"
echo "Using START_EXE=$START_EXE"

export PATH=$PATH:/c/Python$PYVR
PYTHON_EXE=/c/Python$PYVR/python.exe

test -d work-py$PYVR || mkdir  work-py$PYVR
cd work-py$PYVR

function download {
    test -f `basename $1` || wget $1
    test -f `basename $1` || (echo "Failed to download $1" && exit 1)
}

function run_command
{
    echo $PATH
    cmd /c "echo path=%path%"
    echo
    cmd /c "dir"
    echo "Running command: $*"
    cmd /c "$*"
}

function start_installer {
    run_command $START_EXE /W $1
}

function check_python {
    if [ -f $PYTHON_EXE ]; then
	echo "\"$PYTHON_EXE\" exists"
    else
	download $PYTHON_INSTALLER || exit 1
	start_installer `basename $PYTHON_INSTALLER` || exit 1
    fi
    $PYTHON_EXE -c "import sys; print 'Python',sys.version"
}

function check_win32 {
    SUCCESS="YES"
    $PYTHON_EXE -c "import win32api" || SUCCESS="NO"
    if [ "$SUCCESS" == "NO" ]; then
	download $PYWIN32_INSTALLER || exit 1
	start_installer `basename $PYWIN32_INSTALLER`
    fi
}

function configure_pyinstaller {
    cd pyinstaller-$PYINSTALLER_VERSION
    $PYTHON_EXE Configure.py
    cd -
}

function check_pyinstaller {
    if [ -d pyinstaller-$PYINSTALLER_VERSION ]; then
	echo "\"pyinstaller-$PYINSTALLER_VERSION\" exists"
    else
	download $PYINSTALLER_ZIP || exit 1
	unzip `basename $PYINSTALLER_ZIP`
	configure_pyinstaller
    fi

}

function check_wxpython {
    wxVERSION=`$PYTHON_EXE -c "import wx; print wx.VERSION_STRING" || echo "NONE"`
    if [ "$wxVERSION" == "NONE" ]; then
	check_python
	download $WXPYTHON_INSTALLER | exit 1
	start_installer `basename $WXPYTHON_INSTALLER`
	$PYTHON_EXE -c "import wx; print \"Succesfully installed wxPython\", wx.VERSION_STRING"
    else
	echo "Found wxPython $wxVERSION"
    fi
}

function make_iocbio_installer_spec
{
  $PYTHON_EXE pyinstaller-$PYINSTALLER_VERSION/Makespec.py --onefile ../iocbio_installer.py
}


function make_iocbio_installer
{
    export PYTHONPATH=pyinstaller-$PYINSTALLER_VERSION
    test -f iocbio_installer.spec || (make_iocbio_installer_spec || (configure_pyinstaller && (make_iocbio_installer_spec || exit 1)))
    $PYTHON_EXE pyinstaller-$PYINSTALLER_VERSION/Build.py iocbio_installer.spec
    mv -v dist/iocbio_installer.exe ../iocbio_installer_py$PYVR.exe
}

check_python
check_win32
check_pyinstaller
check_wxpython
make_iocbio_installer

test $? == 0 && echo "$0 OK" || echo "$0 FAILED"
