#!/bin/sh
export WINEPREFIX=$HOME/local/.wine2
export WINEDEBUG=fixme-all,warn-all
wine cmd /c $1
