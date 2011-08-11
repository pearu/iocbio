# Author: Pearu Peterson
# Created: Apr 2011

import os
import glob
import shutil
from utils import run_python, run_command, get_program_files_directory, get_windows_directory, get_system_directory, winreg_append_to_path, unwin
from gui import ResourcePage


class PthreadsPage(ResourcePage):
    download_versions = ['2.8.0']
    download_path = {None: 'ftp://sourceware.org/pub/pthreads-win32/pthreads-w32-%(version-)s-release.exe'}


class Libfftw3Page(ResourcePage):

    depends = ['mingw']
    download_versions = ['3.3', '3.2.2']
    download_extensions = {None: '.exe'}
    download_path = {None: 'ftp://ftp.fftw.org/pub/fftw/fftw-%(version)s-dll.zip',
                     #'3.2.2': 'ftp://ftp.fftw.org/pub/fftw/fftw-%(version)s.pl1-dll32.zip',
                     '3.2.2':'http://www.fftw.org/fftw-%(version)s.tar.gz',
                     '3.3':'http://www.fftw.org/fftw-%(version)s.tar.gz',
                     }

    #prefix = os.path.join (get_program_files_directory (), 'fftw3')
    #prefix = os.path.join (r'c:\iocbio\fftw3')

    def __get_install_path(self, installer):
        if prefix is None:
            return
        path = os.path.join (prefix)
        v = os.path.basename(installer).split ('-')[1]
        if v.endswith('.pl1'):
            v = v[:-4]    
        return os.path.join(self.prefix, v.replace ('.','_'))

    def update_environ(self):
        if os.path.isfile (self.path):
            self.environ['FFTW_PATH'] = os.path.dirname(self.path)
        else:
            self.environ['FFTW_PATH'] = self.path
        self.environ['FFTW3'] = self.path

    def get_resource_options(self):
        labels = []
        tasks = {}
        prefix = self.get ('mingw prefix')
        dll = os.path.join(prefix,'bin','libfftw3-3.dll')
        if os.path.isfile (dll):
            label = 'Use %s ' % dll
            labels.append (label)
            tasks[label] = ('use', dll, None)
        '''
        for p in glob.glob(os.path.join(self.prefix, '?_?_?')):
            v = os.path.basename (p).replace ('_','.')
            dlls = map(lambda n: os.path.splitext(os.path.basename(n))[0], glob.glob (os.path.join (p, '*.dll')))
            if dlls:
                label = '%s/{%s}.dll %s' % (p,','.join (dlls), v)
                if label in labels:
                    continue
                labels.append (label)
                tasks[label] = ('use', p, v)
        '''
        return ResourcePage.get_resource_options (self, labels, tasks)

    def try_resource(self, version=None):
        prefix = self.get ('mingw prefix')
        if prefix is None:
            return
        dll = os.path.join(prefix,'bin','libfftw3-3.dll')
        if os.path.isfile (dll):
            return dll
        print 'dll=%s does not exist' % (dll)
        #d = os.path.join(self.prefix, version.replace ('.','_'))
        #if os.path.isdir (d):
        #    return d

    def apply_resource_selection(self):
        r = ResourcePage.apply_resource_selection(self)
        prefix = self.get ('mingw prefix')
        if 0 and r and not os.path.isfile(os.path.join(prefix, 'bin', 'libfftw3.lib')):
            print '%s.apply_resource_selection HACK: creating empty libfftw3.lib for numpy.distutils fftw detection' % (self.__class__.__name__)
            f = open(os.path.join (prefix, 'bin', 'libfftw3.lib'), 'w')
            f.close()
        return r

    def install_source (self, source_path):
        prefix = self.get ('mingw prefix')
        confflags="--prefix=%s --host=i586-mingw32msvc --with-gcc-arch=prescott --enable-portable-binary --with-our-malloc16 --with-windows-f77-mangling --enable-shared --disable-static --enable-threads --with-combined-threads" % (unwin(prefix))
        confflags="--prefix=%s --host=i586-mingw32msvc --with-gcc-arch=native --enable-portable-binary --with-our-malloc16 --with-windows-f77-mangling --enable-shared --disable-static" % (unwin(prefix))
        wd = os.path.join (source_path, 'double-mingw32')
        if 1:
            shutil.rmtree(wd, ignore_errors = True)
            if not os.path.isdir(wd):
                os.makedirs (wd)
        conf = unwin(os.path.join (source_path, 'configure'))
        bash = self.get('mingw bash')
        make = self.get('mingw make')
        if ' ' in conf: 
            raise RuntimeError("The path of fftw3 configure script cannot contain spaces: %r" % (conf))

        if 1:
            r = run_command('%s %s %s --enable-sse2' % (bash, conf, confflags), cwd=wd, env=self.environ,
                            verbose=True)
            if r[0]:
                return False
        r = run_command(make+' -j4', cwd=wd, env=self.environ, verbose=True)
        if r[0]:
            return False
        r = run_command(make+' install', cwd=wd, env=self.environ, verbose=True)
        return not r[0]

class LibtiffPage(ResourcePage):

    download_versions = ['3.8.2', '3.7.4', '3.6.1']
    download_extensions = {None: '.exe'}
    download_path = {None: 'http://sourceforge.net/projects/gnuwin32/files/tiff/%(version)s/tiff-%(version)s.exe',
                     '3.8.2': 'http://sourceforge.net/projects/gnuwin32/files/tiff/%(version)s-1/tiff-%(version)s-1.exe',
                     '3.6.1': 'http://sourceforge.net/projects/gnuwin32/files/tiff/%(version)s-2-win32/tiff-win32-%(version)s-2%(ext)s',
                     }

    def update_environ(self):
        self.update_environ_PATH(os.path.dirname(self.path))

    def get_tiff_dll(self, version):
        dllpath = os.path.join (get_program_files_directory (), r'GnuWin32\bin\libtiff%s.dll' % (version[0]))
        if os.path.isfile(dllpath):
            return dllpath
        return

    def get_tiff_version(self, dll):
        if dll is None:
            return

        r = run_python (self.get('python path'),\
'''
from ctypes import windll, c_char_p
lib = windll.libtiff3
lib.TIFFGetVersion.restype = c_char_p
v = windll.libtiff3.TIFFGetVersion()
print v
''',                            
                        env = dict (PATH=os.path.dirname(dll))
                        )
        
        if not r[0]:
            v = r[1].split('\n')[0]
            if v.startswith ('LIBTIFF, Version'):
                return v.split(' ')[-1]

    def get_resource_options(self):
        labels = []
        tasks = {}
        for v in ['3']:
            dll = self.get_tiff_dll(v)
            if dll is None:
                continue
            version = self.get_tiff_version(dll)
            if version is not None:
                label = '%s %s' % (dll, version)
                if label in labels:
                    continue
                labels.append(label)
                tasks[labels[-1]] = ('use', dll, version)
        return ResourcePage.get_resource_options (self, labels, tasks)

    def try_resource(self, version):
        dll =  self.get_tiff_dll(version)
        v = self.get_tiff_version(dll)
        if v is not None and version.startswith(v):
            return dll

class Vc6RedistPage(ResourcePage):
    download_versions = ['1']
    download_path = {'1':'http://sysbio/download/software/binaries/ms/vcredist_enu.exe',
                         #'http://download.microsoft.com/download/vc60pro/update/1/w9xnt4/en-us/vc6redistsetup_enu.exe',
                     '2':'http://download.microsoft.com/download/vc60pro/update/2/w9xnt4/en-us/vc6redistsetup_deu.exe'}

    def skip (self):
        return bool(self.get_msvcp60_dll())

    def get_msvcp60_dll(self):
        dll = os.path.join(get_system_directory (), r'msvcp60.dll')
        if os.path.isfile (dll):
            return dll

    def try_resource(self, version):
        return self.get_msvcp60_dll()

    def get_resource_options(self):
        labels, tasks = [], {}
        
        dll = self.get_msvcp60_dll ()
        if dll is not None:
            label = 'Use %s' % (dll,)
            if label not in labels:
                labels.append (label)
                tasks[label] = ('use', dll, None)

        return ResourcePage.get_resource_options (self, labels, tasks)

    def apply_resource_selection(self):
        r = ResourcePage.apply_resource_selection(self)
        if not r:
            dll = self.get_msvcp60_dll ()
            if dll is not None:
                print '%s.apply_resource_selection: ignoring failure, we only care about %r' % (self.__class__.__name__, dll)
                return True
        return r


class SubversionPage(ResourcePage):

    depends = ['vc6redist']

    download_versions = ['1.6.17', '1.6.15']
    download_path = {None:'http://sourceforge.net/projects/win32svn/files/%(version)s/Setup-Subversion-%(version)s.msi'}

    def get_svn_exe(self):
        exe = os.path.join (get_program_files_directory(), r'Subversion\bin\svn.exe')
        if os.path.isfile (exe):
            return exe

    def update_environ(self):
        self.update_environ_PATH(os.path.dirname (self.path))

    def get_svn_version (self, exe):
        if exe is None:
            return
        if ' ' in exe:
            r = run_command('"%s" --version' % (exe), verbose=True)
        else:
            r = run_command('%s --version' % (exe), verbose=True)
        if not r[0]:
            line = r[1].split('\n')[0]
            if line.startswith('svn, version'):
                return line.split(' ')[2]

    def try_resource(self, version):
        exe =  self.get_svn_exe()
        v = self.get_svn_version(exe)
        if v is not None and version.startswith(v):
            return exe

    def get_resource_options(self):
        labels, tasks = [], {}
        
        svn = self.get_svn_exe()
        if svn is not None:
            version = self.get_svn_version(svn)
            label = 'Use %s %s' % (svn, version)
            if label not in labels:
                labels.append(label)
                tasks[label] = ('use', svn, version)

        return ResourcePage.get_resource_options (self, labels, tasks)

class MingwPage (ResourcePage):
    
    download_versions = ['20110802-light', 
                         '20110316-light',
                         '20110802', '20110316', 
                         ]
    download_path = {
        '20110802':'http://sourceforge.net/projects/mingw/files/Automated%%20MinGW%%20Installer/mingw-get-inst/mingw-get-inst-20110802/mingw-get-inst-20110802.exe',
        '20110316':'http://sourceforge.net/projects/mingw/files/Automated%%20MinGW%%20Installer/mingw-get-inst/mingw-get-inst-20110316/mingw-get-inst-20110316.exe',
        '20110802-light':'http://sysbio.ioc.ee/download/software/binaries/latest/mingw-%(version)s.zip',
        '20110316-light':'http://sysbio.ioc.ee/download/software/binaries/latest/mingw-%(version)s.zip',
        }
    
    components = {r'bin\gfortran.exe': 'gfortran',
                  r'bin\g++.exe': 'g++', r'bin\ar.exe':'binutils',
                  r'msys\1.0\bin\bash.exe':'msys-bash msys-core msys-coreutils',
                  r'msys\1.0\bin\make.exe':'msys-make',
                  r'msys\1.0\bin\diff.exe':'msys-diffutils',
                  r'msys\1.0\bin\patch.exe': 'msys-patch',
                  r'bin\libpthread-2.dll':'pthreads-w32'}

    prefix = r'C:\iocbio\MinGW'

    def get_install_path (self, installer):
        if installer.endswith ('.zip'):
            return r'C:\\'
        return ResourcePage.get_install_path(self, installer)

    def get_get_exe(self):
        exe = os.path.join(self.prefix, r'bin\mingw-get.exe')
        if os.path.isfile (exe):
            return exe

    def get_get_version(self, exe):
        if exe is not None:
            r = run_command(exe + ' --version', env=self.environ)
            if not r[0]:
                line = r[1].splitlines()[0]
                if line.startswith ('mingw-get version'):
                    return line.split ()[2]

    def try_resource(self, version):
        return self.get_get_exe()

    def update_environ(self):
        self.update_environ_PATH([os.path.join(self.prefix,r'msys\1.0\bin'),
                                  os.path.join(self.prefix,'bin')])

    def get_resource_options(self):
        labels, tasks = [], {}
        
        exe = self.get_get_exe()
        if exe is not None:
            label = 'Use %s' % (exe,)
            if label not in labels:
                labels.append(label)
                tasks[label] = ('use', exe, self.get_get_version(exe))

        return ResourcePage.get_resource_options (self, labels, tasks)

    def apply_resource_selection(self):
        r = ResourcePage.apply_resource_selection(self)
        self.update_environ()
        winreg_append_to_path(self.environ['PATH'])
        if not r:
            return r
        for c, p in self.components.iteritems ():
            f = os.path.join (self.prefix, c)
            if not os.path.isfile (f):
                for p0 in p.split (' '):
                    r = run_command(self.path + ' install ' + p0, env=self.environ, verbose=True)
                    if r[0]:
                        return not r[0]
            if os.path.basename (f)=='bash.exe':
                self.bash = f
            if os.path.basename (f)=='make.exe':
                self.make = f
        r = run_command('%s -c "mkdir -p /tmp"' % (self.bash), env=self.environ)
        if r[0]:
            return False
        return True


