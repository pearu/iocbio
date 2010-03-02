#!/bin/sh

rm -rf doc/_build/ doc/source/generated /net/cens/home/www/sysbio/download/software/iocbio/*

PYTHONPATH=`pwd`:$PYTHONPATH
cd doc && make html || exit 1
cd -
exit 0

#rm -rf apidocs/.buildinfo doc/_build doc/source/generated # full rebuild

