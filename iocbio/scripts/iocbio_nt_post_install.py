

import os
import sys
import glob
import _winreg as winreg

verbose = 0

def winreg_append_to_path(path):
    import _winreg
    if verbose:
        print 'Adding "%s" to environment PATH' % (path)
    environ = _winreg.OpenKey(
        _winreg.HKEY_LOCAL_MACHINE,
        r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
        0,
        _winreg.KEY_ALL_ACCESS
        )
    current_path = _winreg.QueryValueEx (environ, 'PATH')[0]
    if path not in current_path:
        new_path = ';'.join([current_path,path])
        _winreg.SetValueEx(environ, "PATH", None, _winreg.REG_SZ, new_path)
        print 'PATH has been modified, system restart is required.'
    elif verbose:
        print '"%s" already in PATH="%s". That is good.' % (path, current_path)
    
    _winreg.CloseKey(environ)

ver_string = "%d.%d" % (sys.version_info[0], sys.version_info[1])
root_key_name = "Software\\Python\\PythonCore\\" + ver_string
def get_root_hkey():
    try:
        winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                       root_key_name, winreg.KEY_CREATE_SUB_KEY)
        return winreg.HKEY_LOCAL_MACHINE
    except OSError, details:
        # Either not exist, or no permissions to create subkey means
        # must be HKCU
        return winreg.HKEY_CURRENT_USER

def get_shortcuts_folder():
    if get_root_hkey()==winreg.HKEY_LOCAL_MACHINE:
        try:
            fldr = get_special_folder_path("CSIDL_COMMON_PROGRAMS")
        except OSError:
            # No CSIDL_COMMON_PROGRAMS on this platform
            fldr = get_special_folder_path("CSIDL_PROGRAMS")
    else:
        # non-admin install - always goes in this user's start menu.
        fldr = get_special_folder_path("CSIDL_PROGRAMS")

    try:
        install_group = winreg.QueryValue(get_root_hkey(),
                                          root_key_name + "\\InstallPath\\InstallGroup")
    except OSError:
        vi = sys.version_info
        install_group = "Python %d.%d" % (vi[0], vi[1])
    return os.path.join(fldr, install_group)

def get_startmenu_folder (admin=True):
    if admin and get_root_hkey()==winreg.HKEY_LOCAL_MACHINE:
        try:
            fldr = get_special_folder_path("CSIDL_COMMON_STARTMENU")
        except OSError:
            fldr = get_special_folder_path("CSIDL_STARTMENU")
    else:
        fldr = get_special_folder_path("CSIDL_STARTMENU")
    return fldr

def test_libtiff():
    import ctypes
    import ctypes.util
    tiffdir = None
    lib = ctypes.util.find_library('libtiff3')
    if lib is None:
        tiffdir = r'C:\Program Files\GnuWin32\bin'
        if os.path.isdir(tiffdir):
            try:
                winreg_append_to_path(tiffdir)
            except Exception, msg:
                print "Calling winreg_append_to_path failed: %s" % (msg)
                print '"%s" must be added to PATH environment variable manually' % (tiffdir)
            lib = os.path.join(tiffdir, 'libtiff3.dll')
            if not os.path.isfile (lib):
                print 'Warning: File "%s" does not exists. Importing libtiff module may fail.' % (lib)
                lib = None
        else:
            tiffdir = None
    if lib is not None:
        print 'Found tiff library as "%s"' % (lib)
    else:
        if tiffdir is None:
            print 'Warning: Could not find tiff library'
        else:
            print 'Warning: Could not find tiff library in "%s"' % (tiffdir)
        print 'The tiff library installer can be downloaded from http://gnuwin32.sourceforge.net/packages/tiff.htm'
    return False

def test_numpy():
    try:
        import numpy
    except ImportError, msg:
        print 'Warning: failed to import numpy: %s' % (msg)
        return False
    print 'Found Numpy version %s' % (numpy.__version__)
    return True

def test_scipy():
    try:
        import scipy
    except ImportError, msg:
        print 'Warning: failed to import scipy: %s' % (msg)
        return False
    print 'Found Scipy version %s' % (scipy.__version__)
    return True

def test_wx():
    try:
        import wx
    except ImportError, msg:
        print 'Warning: failed to import wx: %s' % (msg)
        return False
    print 'Found WX version %s' % (wx.__version__)
    return True

def test_matplotlib():
    try:
        import matplotlib
    except ImportError, msg:
        print 'Warning: failed to import matplotlib: %s' % (msg)
        return False
    print 'Found matplotlib version %s' % (matplotlib.__version__)
    return True

def install():
    fldr = os.path.join(get_special_folder_path('CSIDL_DESKTOPDIRECTORY'), 'IOCBio Software')
    if not os.path.isdir(fldr):
        os.mkdir(fldr)
        directory_created(fldr)

    # create link to IOC Software folder:
    dst = os.path.join(get_startmenu_folder(), 'IOCBio Software.lnk')
    try:
        create_shortcut(fldr, 'IOCBio', dst)
    except OSError:
        dst = os.path.join(get_startmenu_folder(admin=False), 'IOCBio Software.lnk')
        create_shortcut(fldr, 'IOCBio', dst)
    file_created(dst)

    # create link to IOC Software scripts
    scripts_dir = os.path.join(sys.prefix, 'Scripts')
    ioc_scripts = glob.glob(os.path.join(scripts_dir, 'iocbio_*'))
    for fn in ioc_scripts:
        bn = os.path.basename(fn)
        b,e = os.path.splitext(bn)
        if b=='iocbio_nt_post_install':
            continue
        dst = os.path.join(fldr, "%s.lnk" % (b[7:].title().replace ('_',' ')))
        if verbose:
            print 'Creating shortcut', dst
        create_shortcut(fn, bn[7:], dst, "", fldr)
        file_created(dst)

    # test the availability of prerequisites
    test_libtiff()
    test_numpy()
    test_scipy()
    test_wx()
    test_matplotlib()

    print 
    print '''\
Please find an IOCBio Software folder in your desktop.
The folder contains links to IOC Software components
and will be used as a working directory.'''
    print

def remove():
    pass

if os.name != 'nt':
    pass
elif len(sys.argv)==2:
    task = sys.argv[1]
    if task=='-install':
        install()
    elif task=='-remove':
        remove()
    else:
        raise NotImplementedError (`task, sys.argv`)
