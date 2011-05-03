#!/bin/sh
export WINEPREFIX=$HOME/local/.wine
test -d $WINEPREFIX || mkdir -p $WINEPREFIX
export WINEDEBUG=fixme-all,warn-all
test -f mingw-get-inst-20110316.exe || wget http://sourceforge.net/projects/mingw/files/Automated%20MinGW%20Installer/mingw-get-inst/mingw-get-inst-20110316/mingw-get-inst-20110316.exe
wine cmd /c $1
