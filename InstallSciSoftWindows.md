# Introduction #

## MinGW and MSYS ##

Mingw - provides Windows header files and import libraries combined with GNU toolsets.

  * Select and run recent version of Automated MinGW Installer from http://sourceforge.net/projects/mingw/files/. When choosing components, do not select g++, g77 compilers as they will be available in MSYS.

MSYS - provides GNU utilites such as bash, etc.

  * Install http://downloads.sourceforge.net/mingw/MSYS-1.0.11.exe. Say YES to post installation procedure and follow its instructions.

To check MSYS installation, click MSYS icon on the Desktop and run
```
g++ --version
```
When installing third party software under MSYS, use `/mingw/local` as a prefix (avoid using `/usr/local`, for instance.)

Download and save http://users.ugent.be/~bpuype/cgi-bin/fetch.pl?dl=wget/wget.exe
to `C:\MSYS\1.0\bin`.

### SSH, gfortran ###

Download the following files from http://sourceforge.net/projects/mingw/files to `c:\msys\1.0`:

  * openssh-4.7p1-2-msys-1.0.11-bin.tar.lzma
  * libminires-1.02\_1-1-msys-1.0.11-dll.tar.lzma
  * libopenssl-0.9.8k-1-msys-1.0.11-dll-098.tar.lzma
  * zlib-1.2.3-1-msys-1.0.11-dll.tar.gz
  * gcc-full-4.4.0-mingw32-bin-2.tar.lzma
  * binutils-2.20.1-2-mingw32-bin.tar.gz

and then in MSYS prompt run the following commands for each file:
```
lzma -d file.tar.lzma # or gunzip file.tar.gz
tar xf file.tar
```

Finally, test `ssh` program from MSYS propmt.

Alternatively, you can install http://www.bitvise.com/tunnelier.

## Subversion ##

Install http://subversion.tigris.org/files/documents/15/43506/Setup-Subversion-1.5.2.en-us.msi. Now `svn.exe` will be available in MSYS propmt.

## Python ##

Install recent Python 2.6 version from http://python.org/download/.
Add `C:\Python26` to your system `PATH` environment variable (go to `Control Panel/System and Security/System/Advanced system settings/Advanced/Environment Variables/...`).

## wx Python ##

Install http://downloads.sourceforge.net/wxpython/wxPython2.8-win32-unicode-2.8.10.1-py26.exe.
To test wx installation, download also http://downloads.sourceforge.net/wxpython/wxPython2.8-win32-docs-demos-2.8.10.1.exe.

## BLAS, LAPACK, ATLAS ##

In MSYS prompt, run the following commands
```
# Get sources
mkdir -p ~/src/; cd ~/src
wget http://www.netlib.org/blas/blas.tgz
tar xzf blas.tgz
wget http://www.netlib.org/lapack/lapack.tgz
tar xzf lapack.tgz
# download ATLAS from http://sourceforge.net/projects/math-atlas/files/

# Build libfblas.a
cd ~/src/BLAS
gfortan -fno-second-underscore -O2 -c *.f
ar r libfblas.a *.o
ranlib libfblas.a
rm *.o

# Build libflapack.a
cd ~/src/lapack-3.2.1
wget http://iocbio.googlecode.com/files/make.inc.MINGW
mv make.inc.MINGW make.inc
make lapacklib
mv lapack_MINGW.a libflapack.a

# Build ATLAS
cd ~/src
bunzip2 atlas3.9.23.tar.bz2
tar xf atlas3.9.23.tar
cd ATLAS

```
## numpy ##

Get numpy from svn:
```
svn co http://svn.scipy.org/svn/numpy/trunk numpy
```

## Emacs ##

Download http://ftp.gnu.org/gnu/emacs/windows/emacs-23.1-bin-i386.zip and
unpack it to `C:`. Run `C:\emacs-23.1\bin\addpm.exe` that will add a shortcut to Start Menu. In addition, add emacs bin path to `PATH` environment variable.