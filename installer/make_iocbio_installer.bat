@rem Author: Pearu Peterson
@rem Created: Apr 2011
@ECHO OFF
echo %0

set MINGW_INSTALLER=mingw-get-inst-20110316.exe
set MINGW_ROOT=C:\MinGW
set GET_EXE=%MINGW_ROOT%\bin\mingw-get.exe
set BASH_EXE=%MINGW_ROOT%\msys\1.0\bin\bash.exe
set PATH=%MINGW_ROOT%\bin;%MINGW_ROOT%\msys\1.0\bin;%PATH%

if exist "%MINGW_INSTALLER%" GOTO MINGWINSTALL
@echo Download the following MinGW installer:
@echo  
@echo   http://sourceforge.net/projects/mingw/files/Automated%20MinGW%20Installer/mingw-get-inst/mingw-get-inst-20110316/mingw-get-inst-20110316.exe
@echo  
@echo and rerun this batch file "%0"
GOTO END
:MINGWINSTALL
@echo "%MINGW_INSTALLER%" exists

if exist "%MINGW_ROOT%" GOTO HAS_MINGW
mingw-get-inst-20110316.exe
:HAS_MINGW
@echo "%MINGW_ROOT%" exists

if exist "%BASH_EXE%" GOTO HAS_BASH
%GET_EXE% install msys-bash msys-core msys-coreutils msys-wget msys-unzip
GOTO SKIP
%GET_EXE% install msys-bash msys-core msys-coreutils msys-wget msys-unzip gfortran  g++ binutils msys-patch msys-tar msys-make
:SKIP
:HAS_BASH
@echo "%BASH_EXE%" exists
%BASH_EXE% make_iocbio_installer.sh
:END
@echo EOF %0