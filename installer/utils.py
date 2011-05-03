# Author: Pearu Peterson
# Created: Apr 2011
import os
import sys
import subprocess
import tempfile
import urllib2
import shutil
import urlparse
import zipfile
import tarfile

def get_appdata_directory():
    try:
        from win32com.shell import shell, shellcon
        return shell.SHGetSpecialFolderPath(0,shellcon.CSIDL_APPDATA)
    except Exception, msg:
        print 'get_appdata_directory: %s' % (msg)
        return ''

def get_desktop_directory():
    from win32com.shell import shell, shellcon
    return shell.SHGetSpecialFolderPath(0,shellcon.CSIDL_DESKTOPDIRECTORY)

def get_program_files_directory ():
    try:
        from win32com.shell import shell, shellcon
        return shell.SHGetSpecialFolderPath(0,shellcon.CSIDL_PROGRAM_FILES)
    except Exception, msg:
        print 'get_program_files_directory: %s' % (msg)
        return r'C:\Program Files'

def get_windows_directory ():
    try:
        from win32com.shell import shell, shellcon
        return shell.SHGetSpecialFolderPath(0,shellcon.CSIDL_WINDOWS)
    except Exception, msg:
        print 'get_windows_directory: %s' % (msg)
        return r'C:\Windows'

def get_system_directory ():
    try:
        from win32com.shell import shell, shellcon
        return shell.SHGetSpecialFolderPath(0,shellcon.CSIDL_SYSTEM)
    except Exception, msg:
        print 'get_system_directory: %s' % (msg)
        return r'C:\Windows\System32'

def create_shortcut(shortcut_path, target_path):
    import pythoncom
    from win32com.shell import shell, shellcon
    working_path = shell.SHGetFolderPath (0, shellcon.CSIDL_PERSONAL, 0, 0)

    if os.path.exists(shortcut_path):
        shortcut = pythoncom.CoCreateInstance (
            shell.CLSID_ShellLink,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IShellLink
            )
        persist_file = shortcut.QueryInterface (pythoncom.IID_IPersistFile)
        persist_file.Load (shortcut_path)

        shortcut.SetPath (target_path)
        shortcut.SetDescription (target_path)
        shortcut.SetIconLocation (sys.executable, 0)
        shortcut.SetWorkingDirectory (working_path)

        persist_file.Save (shortcut_path, 0)
    else:
        shortcut = pythoncom.CoCreateInstance (
            shell.CLSID_ShellLink,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IShellLink
            )
        shortcut.SetPath (target_path)
        shortcut.SetDescription (target_path)
        shortcut.SetIconLocation (sys.executable, 0)
        shortcut.SetWorkingDirectory (working_path)

        persist_file = shortcut.QueryInterface (pythoncom.IID_IPersistFile)
        persist_file.Save (shortcut_path, 0)


def winreg_append_to_path(path):
    try:
        import _winreg
        verbose = True
        if verbose:
            print 'Adding "%s" to environment PATH' % (path)
        environ = _winreg.OpenKey(
            _winreg.HKEY_LOCAL_MACHINE,
            r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
            0,
            _winreg.KEY_ALL_ACCESS
            )
        current_path = _winreg.QueryValueEx (environ, 'PATH')[0]
        new_paths = current_path.split(os.pathsep)
        paths = path.split (os.pathsep)
        for path in path.split(os.pathsep):
            if path not in new_paths:
                new_paths.insert(0, path)
        new_path = os.pathsep.join(new_paths)

        if new_path != current_path:
            _winreg.SetValueEx(environ, "PATH", None, _winreg.REG_SZ, new_path)
            print 'PATH has been modified, system restart is required.'
        else:
            print 'PATH is already up-to-date.'    
        _winreg.CloseKey(environ)
    except Exception, msg:
        print msg
        return False
    return True

def run_command(cmd, verbose=False, env=None, cwd=None):
    if verbose:
        print 'In %r: %s' % (cwd or '.', cmd)
    if env is not None:
        new_env = {}
        for k,v in env.iteritems ():
            new_env[str(k)] = str (v)
        env = new_env
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env, cwd=cwd)
    stdout, stderr = p.communicate()
    stdout = stdout.replace ('\r\n', '\n')
    stderr = stderr.replace ('\r\n', '\n')
    status = p.returncode
    if verbose and status:
        print 'Command %r failed with returncode=%r' % (cmd, status)
        print '='*20
        print 'STDOUT: %s' % (stdout)
        print '-'*20
        print 'STDERR: %s' % (stderr)
        print '='*20
    elif verbose:
        print 'Command %r succesfully completed' % (cmd)
        print '='*20
        print 'STDOUT: %s' % (stdout)
        print '-'*20
        print 'STDERR: %s' % (stderr)
        print '='*20
    return status, stdout, stderr

def download(url, fileName=None):
    def getFileName(url,openUrl):
        info = openUrl.info()
        print info
        filename = None
        try:
            for s in info.get('Content-Disposition', '').split(';'):
                if s.startswith('filename='):
                    filename = s[9:].strip("\"'")
                    break
        except Exception, msg:
            print 'msg=%s' % (msg)
        if filename is None:
            filename = os.path.basename(urlparse.urlsplit(openUrl.url)[2])
        return filename
    try:
        r = urllib2.urlopen(urllib2.Request(url))
    except urllib2.HTTPError, msg:
        print 'Failed to open %r: %s' (url, msg)
        return
    result = None
    try:
        fileName = fileName or getFileName(url,r)
        f = open(fileName, 'wb')
        shutil.copyfileobj(r,f)
        f.close()
        result = os.path.abspath(fileName)
    finally:
        r.close()
    return result
        
def start_installer(installer):
    start_exe = r'C:\Windows\command\start.exe' # mingw
    if os.path.splitext (installer)[1]=='.exe':
        cmd = installer
    elif os.path.isfile (start_exe):
        cmd = '%s /W "%s"' % (start_exe, installer)
    else:
        comspec = os.environ.get('COMSPEC', 'cmd')
        cmd = '%s /c "%s"' % (comspec, installer)
    try:
        p = subprocess.Popen (cmd)
    except OSError, msg:
        print 'start_installer %r failed: %s' % (cmd, msg)
        return False
    p.communicate()
    status = p.returncode
    if status==3010:
        print 'cmd=%r' % (cmd)
        print '%s exited with returncode=%r which means "reboot required", IGNORING' % (installer, status)
        return True
    if status:
        print 'cmd=%r' % (cmd)
        print 'Starting %r failed with returncode=%r' % (installer, status)
        return False
    return True

def extract(filename, extract_to):
    print 'Extracting the content of %r to %r ...' % (filename, extract_to),
    if filename.endswith('.zip'):
        z = zipfile.ZipFile(filename)
        z.extractall(extract_to)
        r = [os.path.join(extract_to, info.filename) for info in z.infolist()]
    elif filename.endswith('.tar.gz'):
        z = tarfile.open(filename, 'r:gz')
        z.extractall(extract_to)
        r = [os.path.join(extract_to, info.name) for info in z.getmembers()]
    else:
        print 'extract(%r) not implemented' % (os.path.splitext (filename)[-1])
        return
    print 'DONE'
    return r

def run_python(python_exe, python_code, expected_output=None, suppress_errors=False, env=None):
    tmp = tempfile.mktemp('.py')
    f = open(tmp, 'w')
    f.write (python_code)
    f.close()
    r = run_command('%s %s' % (python_exe, tmp), env=env)
    if suppress_errors:
        pass
    elif r[0]:
        print '-'*20
        print 'python_exe: %s' % (python_exe)
        print 'Executing the following Python code failed with returncode=%r:' % (r[0])
        print python_code
        print 'STDOUT:'
        print r[1]
        print 'STDERR:'
        print r[2]
        print '-'*20
    elif expected_output is not None:
        output = r[1].rstrip().replace('\r\n', '\n').rstrip()
        expected_output = expected_output.rstrip().replace('\r\n', '\n').rstrip()
        if output != expected_output:
            print 'Executing the following Python code returned unexpected output:'
            print python_code
            print 'STDOUT:'
            print r[1]
            print 'EXPECTED STDOUT:'
            print expected_output
            print '-'*20
    os.remove(tmp)
    return r
