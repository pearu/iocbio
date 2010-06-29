#!/bin/sh

PYTHONPATH=`pwd`:$PYTHONPATH

cd doc && make clean && make html || exit 1
cd -
exit 0

