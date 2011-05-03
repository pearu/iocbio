#!/bin/sh
export WINEPREFIX=$HOME/local/.wine
export WINEDEBUG=fixme-all,warn-all
wine cmd /c $1
