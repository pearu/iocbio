#!/bin/sh

SVNVERSION=`svnversion`


PYVER=26

PACKAGENAME=iocbio
SUMMARY="IOCBio software installer (revision $SVNVERSION)"
PROJECT=iocbio
LABELS="Type-Source,OpSys-Windows,Featured"

#cd installer
#./wineit.sh make_iocbio_installer.bat
#cd -

INSTALLER=installer/iocbio_installer_py$PYVER.exe
echo INSTALLER=$INSTALLER
test -f $INSTALLER || exit 1
python installer/googlecode_upload.py --summary="$SUMMARY" --project="$PROJECT" --labels="$LABELS" $INSTALLER
