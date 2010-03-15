#!/bin/sh

rm -rf doc/_build/ doc/source/generated /net/cens/home/www/sysbio/download/software/iocbio/*
echo `date -u` ": documentation is being updated... try again in few minutes." >  /net/cens/home/www/sysbio/download/software/iocbio/index.html
PYTHONPATH=`pwd`:$PYTHONPATH
cd doc && make html || exit 1
cd -
exit 0

#rm -rf apidocs/.buildinfo doc/_build doc/source/generated # full rebuild

