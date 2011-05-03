#!/usr/bin/env python
#
# Copyright 2006, 2007 Google Inc. All Rights Reserved.
# Author: danderson@google.com (David Anderson)
#
# Script for uploading files to a Google Code project.
#
# This is intended to be both a useful script for people who want to
# streamline project uploads and a reference implementation for
# uploading files to Google Code projects.
#
# To upload a file to Google Code, you need to provide a path to the
# file on your local machine, a small summary of what the file is, a
# project name, and a valid account that is a member or owner of that
# project.  You can optionally provide a list of labels that apply to
# the file.  The file will be uploaded under the same name that it has
# in your local filesystem (that is, the "basename" or last path
# component).  Run the script with '--help' to get the exact syntax
# and available options.
#
# Note that the upload script requests that you enter your
# googlecode.com password.  This is NOT your Gmail account password!
# This is the password you use on googlecode.com for committing to
# Subversion and uploading files.  You can find your password by going
# to http://code.google.com/hosting/settings when logged in with your
# Gmail account. If you have already committed to your project's
# Subversion repository, the script will automatically retrieve your
# credentials from there (unless disabled, see the output of '--help'
# for details).
#
# If you are looking at this script as a reference for implementing
# your own Google Code file uploader, then you should take a look at
# the upload() function, which is the meat of the uploader.  You
# basically need to build a multipart/form-data POST request with the
# right fields and send it to https://PROJECT.googlecode.com/files .
# Authenticate the request using HTTP Basic authentication, as is
# shown below.
#
# Licensed under the terms of the Apache Software License 2.0:
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Questions, comments, feature requests and patches are most welcome.
# Please direct all of these to the Google Code users group:
#  http://groups.google.com/group/google-code-hosting

"""Google Code file uploader script.
"""

__author__ = 'danderson@google.com (David Anderson)'

import httplib
import os.path
import optparse
import getpass
import base64
import sys


def upload(file, project_name, user_name, password, summary, labels=None):
  """Upload a file to a Google Code project's file server.

  Args:
    file: The local path to the file.
    project_name: The name of your project on Google Code.
    user_name: Your Google account name.
    password: The googlecode.com password for your account.
              Note that this is NOT your global Google Account password!
    summary: A small description for the file.
    labels: an optional list of label strings with which to tag the file.

  Returns: a tuple:
    http_status: 201 if the upload succeeded, something else if an
                 error occured.
    http_reason: The human-readable string associated with http_status
    file_url: If the upload succeeded, the URL of the file on Google
              Code, None otherwise.
  """
  # The login is the user part of user@gmail.com. If the login provided
  # is in the full user@domain form, strip it down.
  if user_name.endswith('@gmail.com'):
    user_name = user_name[:user_name.index('@gmail.com')]

  form_fields = [('summary', summary)]
  if labels is not None:
    form_fields.extend([('label', l.strip()) for l in labels])

  content_type, body, content_length = encode_upload_request(form_fields, file)

  upload_host = '%s.googlecode.com' % project_name
  upload_uri = '/files'
  auth_token = base64.b64encode('%s:%s'% (user_name, password))
  headers = {
    'Authorization': 'Basic %s' % auth_token,
    'User-Agent': 'Googlecode.com uploader v0.9.4',
    'Content-Type': content_type,
    }

  # check existence first
  server = httplib.HTTPConnection(upload_host)
  path = os.path.join(upload_uri, os.path.basename(file))
  server.request('GET', path) # unfortunately HEAD does not work
  resp = server.getresponse()
  server.close ()
  target_location = None
  if resp.status==httplib.OK:
    c = int(resp.getheader('content-length', 0))
    target_location = 'http://%s%s'%(upload_host, path)
    print 'WARNING: Target %r exists.' % (target_location)
    if c!=content_length:
      print 'WARNING: Content length test failed. File size=%s, content size=%s.' % (content_length, c)
      target_location = None

  server = httplib.HTTPSConnection(upload_host)
  server.request('POST', upload_uri, body, headers)
  resp = server.getresponse()
  server.close()

  if resp.status == httplib.CREATED:
    location = resp.getheader('Location', None)
    print 'The file was uploaded successfully.'
  else:
    if target_location is not None:
      print 'Failed to upload file because it exists.'
    location = target_location
  return resp.status, resp.reason, location


def encode_upload_request(fields, file_path):
  """Encode the given fields and file into a multipart form body.

  fields is a sequence of (name, value) pairs. file is the path of
  the file to upload. The file will be uploaded to Google Code with
  the same file name.

  Returns: (content_type, body) ready for httplib.HTTP instance
  """
  BOUNDARY = '----------Googlecode_boundary_reindeer_flotilla'
  CRLF = '\r\n'

  body = []

  # Add the metadata about the upload first
  for key, value in fields:
    body.extend(
      ['--' + BOUNDARY,
       'Content-Disposition: form-data; name="%s"' % key,
       '',
       value,
       ])

  # Now add the file itself
  file_name = os.path.basename(file_path)
  f = open(file_path, 'rb')
  file_content = f.read()
  f.close()

  body.extend(
    ['--' + BOUNDARY,
     'Content-Disposition: form-data; name="filename"; filename="%s"'
     % file_name,
     # The upload server determines the mime-type, no need to set it.
     'Content-Type: application/octet-stream',
     #'Content-Length: %s' % (len(file_content)),
     '',
     file_content,
     ])

  # Finalize the form body
  body.extend(['--' + BOUNDARY + '--', ''])

  return 'multipart/form-data; boundary=%s' % BOUNDARY, CRLF.join(body), len (file_content)

def parse_auth_file(fn):
  f = open (fn, 'r')
  content = f.readlines()
  f.close()
  key = None
  data = {}
  while content:
    line = content.pop(0)
    if line.startswith('K '):
      c = int(line[2:].strip())
      key = content.pop(0).strip()
      assert len (key)==c,`len (key), c`
    elif line.startswith('V '):
      c = int(line[2:].strip())
      value = content.pop(0).strip()
      assert len (value)==c,`len (value), c`
      assert key is not None
      data[key] = value
      key = None
    elif line.startswith('END'):
      break
  return data

def resolve_user_name_and_password(project_name, user_name, password):
  """ Find credentials from .subversion/auth
  """
  import glob
  if user_name or password:
    return user_name, password
  files = glob.glob(os.path.abspath(os.path.join (os.environ['HOME'],'.subversion/auth/svn.simple/*')))
  for fn in files:
    data = parse_auth_file(fn)
    if '//%s.googlecode.com' % (project_name) in data.get('svn:realmstring'):
      return data.get('username'), data.get('password')
  return None, None

def upload_find_auth(file_path, project_name, summary, labels=None,
                     user_name=None, password=None, tries=3):
  """Find credentials and upload a file to a Google Code project's file server.

  file_path, project_name, summary, and labels are passed as-is to upload.

  Args:
    file_path: The local path to the file.
    project_name: The name of your project on Google Code.
    summary: A small description for the file.
    labels: an optional list of label strings with which to tag the file.
    config_dir: Path to Subversion configuration directory, 'none', or None.
    user_name: Your Google account name.
    tries: How many attempts to make.
  """
  user_name, password = resolve_user_name_and_password (project_name, user_name, password)

  while tries > 0:
    if user_name is None:
      # Read username if not specified or loaded from svn config, or on
      # subsequent tries.
      sys.stdout.write('Please enter your googlecode.com username: ')
      sys.stdout.flush()
      user_name = sys.stdin.readline().rstrip()
    if password is None:
      # Read password if not loaded from svn config, or on subsequent tries.
      print 'Please enter your googlecode.com password.'
      print '** Note that this is NOT your Gmail account password! **'
      print 'It is the password you use to access Subversion repositories,'
      print 'and can be found here: http://code.google.com/hosting/settings'
      password = getpass.getpass()

    status, reason, url = upload(file_path, project_name, user_name, password,
                                 summary, labels)
    # Returns 403 Forbidden instead of 401 Unauthorized for bad
    # credentials as of 2007-07-17.
    if status in [httplib.FORBIDDEN, httplib.UNAUTHORIZED]:
      if status==httplib.FORBIDDEN:
        if url is not None:
          break
      # Rest for another try.
      user_name = password = None
      tries = tries - 1
    else:
      # We're done.
      break

  return status, reason, url


def main():
  parser = optparse.OptionParser(usage='googlecode-upload.py -s SUMMARY '
                                 '-p PROJECT [options] FILE')
  parser.add_option('-s', '--summary', dest='summary',
                    help='Short description of the file')
  parser.add_option('-p', '--project', dest='project',
                    help='Google Code project name')
  parser.add_option('-u', '--user', dest='user',
                    help='Your Google Code username')
  parser.add_option('-w', '--password', dest='password',
                    help='Your Google Code password')
  parser.add_option('-l', '--labels', dest='labels',
                    help='An optional list of comma-separated labels to attach '
                    'to the file')

  options, args = parser.parse_args()

  if not options.summary:
    parser.error('File summary is missing.')
  elif not options.project:
    parser.error('Project name is missing.')
  elif len(args) < 1:
    parser.error('File to upload not provided.')
  elif len(args) > 1:
    parser.error('Only one file may be specified.')

  file_path = args[0]

  if options.labels:
    labels = options.labels.split(',')
  else:
    labels = None

  status, reason, url = upload_find_auth(file_path, options.project,
                                         options.summary, labels,
                                         options.user, options.password)
  if url:
    print 'URL: %s' % url
    return 0
  else:
    print 'An error occurred. Your file was not uploaded.'
    print 'Google Code upload server said: %s (%s)' % (reason, status)
    return 1


if __name__ == '__main__':
  sys.exit(main())
