#!/bin/sh

SVNVERSION=`svnversion`

PACKAGENAME=iocbio
SUMMARY="IOCBio software installer (revision $SVNVERSION)"
PROJECT=iocbio
LABELS="Type-Source,OpSys-Windows,Featured"

INSTALLER=installer/iocbio_installer_py26.exe
echo INSTALLER=$INSTALLER
test -f $INSTALLER || exit 1
python installer/googlecode_upload.py --summary="$SUMMARY" --project="$PROJECT" --labels="$LABELS" $INSTALLER
