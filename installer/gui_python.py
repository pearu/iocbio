
# Author: Pearu Peterson
# Created: Apr 2011

import os
import sys
import glob
from gui import ResourcePage
from utils import run_python, run_command, extract, download, get_desktop_directory, create_shortcut

class PythonPackagePage(ResourcePage):

    depends = ['python']

    @property
    def packagename (self):
        return self.title.lower()

    def try_python_package(self, version=None):
        """ Return Python package path and version as 2-tuple.
        When package is not available, return (None, None).
        """
        r = self.run_python (\
'''
import os
import '''+self.packagename+''' as package
version = getattr(package, "__version__", "N/A")
if version=="N/A":
    try:
        from '''+self.packagename+'''.version import version
    except:
        pass
print("@%s@@@%s" % (os.path.dirname(package.__file__), version))
''')
        if not r[0]:
            i = r[1].find('@')+1
            PATH, VER = r[1][i:].split ('@@@')
            VER = VER.rstrip()
            if VER=='N/A' and version is not None: VER=version
            return PATH, VER
        return None, None

    def quick_test(self):
        if self.path is not None:
            r = run_python(self.get('python path'), '''
import %s as package
version = getattr(package, "__version__", "N/A")
if version=="N/A":
    try:
        from %s.version import version
    except:
        pass
print(version)
''' % (self.packagename, self.packagename), self.version)
            return not r[0]
        else:
            print '%s quick test skipped' % (self.title)

    def get_resource_options(self):
        labels = []
        tasks = {}
        #pyver = self.get('python version')[:3]

        path, version = self.try_python_package()
        if path is not None:
            labels.append('%s %s' % (path, version))
            tasks[labels[-1]] = ('use', path, version)

        for v, depvers in sorted(self.download_versions.items(), reverse=True):
            if v=='launchpad':
                label = 'Checkout %s from %s' % (self.title, v)
                if label in labels:
                    continue
                labels.append (label)
                tasks[label] = ('bzr', self.download_path.get(v, ''), v)
                continue
            if v=='svn':
                svn_repo = self.download_path.get(v, '')
                if not svn_repo:
                    print '%s.get_resource_options:Warning: no svn repository specified in download_path' % (self.__class__.__name__)
                    continue
                if ' ' in svn_repo:
                    sourcedir = svn_repo.rsplit(' ',1)[-1]
                else:
                    sourcedir = os.path.basename(svn_repo)
                if os.path.isdir (sourcedir):
                    label = 'Update and install %s from %s' % (self.title, v)
                else:
                    label = 'Checkout and install %s from %s' % (self.title, v)
                if label in labels:
                    continue
                labels.append (label)
                tasks[label] = ('svn', svn_repo, v)
                continue
            deps = dict (version=v)
            deps['version:3'] = v[:3]
            deps['version:3:2'] = v[:3:2]
            for dep, dvers in depvers.iteritems():
                dv = self.get('%s version' % (dep)).split()[0]
                if dv is None:
                    deps = None
                    break
                found = False
                for v1 in dvers:
                    if dv.startswith(v1):
                        found = True
                        break
                if not found:
                    deps = None
                    break
                deps['%s version' % dep] = dv
                deps['%s version:3' % dep] = dv[:3]
                deps['%s version:3:2' % dep] = dv[:3:2]
            if deps is None:
                continue
            download_path = self.download_path.get (v, self.download_path.get(None, ''))
            if download_path:
                installer = download_path % deps
                if installer.endswith('.tar.gz'):
                    if os.path.isfile(os.path.basename(installer)):
                        labels.append('Install %s %s from source' % (self.title, v))
                    else:
                        labels.append('Download and install %s %s from source' % (self.title, v))
                    tasks[labels[-1]]= ('get_source', installer, v)            
                else:
                    if os.path.isfile(os.path.basename(installer)):
                        labels.append('Install %s %s from binary' % (self.title, v))
                    else:
                        labels.append('Download and install %s %s from binary' % (self.title, v))
                    tasks[labels[-1]]= ('get', installer, v)            
        return labels, tasks

    def apply_resource_selection(self):
        selections = self.get_resource_selection ()
        if not selections:
            print 'Nothing selected for %s' % (self.title)
            return False
        selection = selections[0]
        task = self.selection_task_map[selection]

        if task[0]=='use':
            self.path = task[1]
            self.version = task[2]
            self.update_environ()
            return True

        if task[0]=='get':
            installer = task[1]
            version = task[2]
            if not self.get_and_run_installer (installer):
                return False
            path, version = self.try_python_package(version)
            if path is not None:
                self.path = path
                self.version = version
                self.update_environ()
                return True
            return False

        if task[0]=='bzr':
            r = run_command('bzr branch %s' % (task[1]), env=self.environ)
            raise NotImplementedError(`r`)
            return False

        if task[0]=='svn':
            if ' ' in task[1]:
                sourcedir = task[1].rsplit(' ',1)[-1]
            else:
                sourcedir = os.path.basename(task[1])
            svn = self.get('subversion path')
            if ' ' in svn:
                svn = '"%s"' % (svn)
            if os.path.isdir(sourcedir):
                r = run_command('%s update' % (svn), env=self.environ, cwd=sourcedir, verbose=True)
            else:
                r = run_command('%s checkout %s' % (svn, task[1]), env=self.environ, verbose=True)
            if r[0]:
                return False
            if not self.install_source(sourcedir):
                return False
            path, version = self.try_python_package()
            if path is not None:
                self.path = path
                self.version = version
                self.update_environ()        
                return True
            return False

        if task[0]=='get_source':
            installer = task[1]
            version = task[2]
            if not self.get_and_install_source(installer):
                return False
            path, version = self.try_python_package(version)
            if path is not None:
                self.path = path
                self.version = version
                self.update_environ()
                return True
            return False

        raise NotImplementedError (`task`)

    def run_python(self, *args, **kws):
        return run_python(self.get('python path'), *args, **kws)

    def install_source(self, sourcedir):
        r = run_command(self.get('python path')+' '+self.get_setup_install_command(),
                        env = self.environ, cwd = sourcedir,
                        verbose=True)
        return not r[0]

    def get_install_path (self, installer_file):
        base = os.path.basename (installer_file)
        if base.endswith('.tar.gz'):
            return base[:-7]
        return ResourcePage.get_install_path(self, installer_file)

    def get_and_install_source(self, installer):
        installer_file = os.path.abspath(os.path.basename (installer))
        if not os.path.isfile (installer_file):
            print 'Downloading', installer, '..',
            installer_file = download (installer)
            if installer_file is None:
                print 'Download FAILED'
                return False
            print 'DONE'
        install_path = self.get_install_path(installer_file)
        if install_path is not None:
            if not os.path.isdir (install_path):
                os.makedirs (install_path)
        else:
            install_path = '.'
        content = extract(installer_file, install_path)
        if not content:
            return False
        cwd = install_path
        for p in content:
            if os.path.isdir(p):
                cwd = p
                break
        return self.install_source(cwd)

    def install_source(self, source_path):
        print '%s.install_source (%r) not implemented' % (self.__class__.__name__, source_path)

    def get_setup_install_command (self):
        return 'setup.py install'

class PythonPage (ResourcePage):

    # specify only those versions that are provided with .msi file.
    download_versions = ['2.7.2', '2.6.7', '2.5.4', '2.4.4', 
                         '2.3.5', # Running Python fails with runtime error R6034
                         '3.2.1', '3.1.4', '3.0.1']
    download_extensions = {None: '.msi', # default
                           '2.3.5': '.exe'}
    download_path = {None: 'http://www.python.org/ftp/python/%(version)s/python-%(version)s%(ext)s',
                     '2.3.5': 'http://www.python.org/ftp/python/%(version)s/Python-%(version)s%(ext)s'}

    def try_python(self, python_exe):
        r = run_python(python_exe, 'import sys;print("%s@@@%s" % (sys.executable, sys.version))',
                       suppress_errors = True)
        if not r[0]:
            EXE, VER = r[1].split ('@@@')
            EXE = os.path.realpath(EXE)
            print '%r -> %r' % (python_exe, EXE)
            return EXE, VER.rstrip()
        return None, None

    def quick_test(self):
        if self.path is not None:
            r = run_python(self.path, 'print ("Hello!")', 'Hello!')
            return not r[0]
        else:
            print '%s quick test skipped' % (self.title)

    def get_python_exe(self, v):
        if not v:
            return 'python'
        if os.name=='nt':
            exe = 'C:\\Python%s\\python.exe' % (v.replace ('.','')[:2])
            if os.path.isfile (exe):
                return exe
        elif os.name=='posix':
            return 'python%s' % v[:3]
        else:
            raise NotImplementedError(`os.name`)
        return

    def get_resource_options(self):
        labels = []
        tasks = {}
        for v in ['', '3.2','3.1','3.0','2.9','2.8','2.7','2.6','2.5','2.4','2.3']:
            exe = self.get_python_exe (v)
            if exe is None:
                continue
            if not v and not sys.version.startswith(v):
                continue
            EXE, VER = self.try_python(exe)
            if EXE is not None:
                VER = VER.strip()
                label = '%s %s' % (EXE, VER)
                if label in labels:
                    continue
                if not sys.version.startswith (VER[:3]):
                    continue
                labels.append(label)
                tasks[labels[-1]] = ('use', EXE, VER)

        for v in self.download_versions:
            if not sys.version.startswith (v):
                continue
            download_path = self.download_path.get (v, self.download_path.get(None, ''))
            if download_path:
                ext = self.download_extensions.get(v, self.download_extensions.get(None,''))
                installer = download_path % dict(version=v, ext=ext)
                if os.path.isfile(os.path.basename(installer)):
                    labels.append('Install Python %s from binary' % (v))
                else:
                    labels.append('Download and install Python %s from binary' % (v))
                tasks[labels[-1]]= ('get', installer, v)
        return labels, tasks

    def update_environ(self):
        old_path = self.environ.get('PATH')
        rootpath = os.path.dirname (self.path)
        scriptpath = os.path.join (rootpath, 'Scripts')
        self.update_environ_PATH([rootpath, scriptpath])

    def apply_resource_selection(self):
        selections = self.get_resource_selection ()
        if not selections:
            print 'Nothing selected for %s' % (self.title)
            return False
        selection = selections[0]
        task = self.selection_task_map[selection]
        if task[0]=='use':
            self.path = task[1]
            self.version = task[2]
            self.update_environ()
            return True

        if task[0]=='get':
            installer = task[1]
            version = task[2]
            if not self.get_and_run_installer (installer):
                return False

            exe = self.get_python_exe(version)
            if exe:
                EXE, VER = self.try_python(exe)
                if EXE is not None:                
                    self.path = EXE
                    self.version = VER[:3]
                    self.update_environ()
                    return True
            print 'Installation failed:',`installer`
            return False

        raise NotImplementedError (`task`)


class NumpyPage(PythonPackagePage):

    download_versions = {
        '1.6.1':dict (python=['2.5','2.6','2.7','3.1','3.2']),
        '1.5.1':dict (python=['2.5', '2.6', '2.7', '3.1']),
        '1.4.1':dict (python=['2.5', '2.6']),
        '1.3.1':dict (python=['2.5', '2.6']),
        '1.2.1':dict (python=['2.4', '2.5']),
        '1.1.1':dict (python=['2.3', 
                              '2.4',
                              '2.5']),
                         }
    download_path = {None: 
                     'http://sourceforge.net/projects/numpy/files/NumPy/%(version)s/numpy-%(version)s-win32-superpack-python%(python version:3)s.exe'}

class ScipyPage (PythonPackagePage):

    depends = ['python', 'numpy']

    download_versions = {'0.9.0': dict(python=['2.5', '2.6', '2.7'], numpy=['1.5']),
                         '0.8.0': dict (python=['2.5', '2.6'], numpy=['1.4']),
                         '0.7.2': dict (python=['2.5', '2.6'], numpy=['1.4'])}
    download_path = {None:'http://sourceforge.net/projects/scipy/files/scipy/%(version)s/scipy-%(version)s-win32-superpack-python%(python version:3)s.exe'}


class MatplotlibPage (PythonPackagePage):

    depends = ['python', 'numpy']

    download_versions = {'1.0.1': dict(python=['2.4', 
                                               '2.5',
                                               '2.6', 
                                               '2.7',
                                               ],
                                       numpy=['1.5']),
                         '0.98.1': dict (python=['2.4', '2.5'], numpy=['1.2'])
                         }
    download_path = {None:'http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-%(version)s/matplotlib-%(version)s.win32-py%(python version:3)s.exe'}


class WxPage (PythonPackagePage):

    download_versions = {'2.9.1.1':dict (python=['2.5', '2.6','2.7']),
                         '2.9.2.1':dict (python=['2.6', '2.7'])}

    download_path = {None: 'http://sourceforge.net/projects/wxpython/files/wxPython/%(version)s/wxPython%(version:3)s-win32-%(version)s-py%(python version:3:2)s.exe'}

class Fftw3Page(PythonPackagePage):
    
    depends = ['libfftw3', 'numpy']

    download_versions = {#'launchpad': dict (python=['2.6', '2.7']),
        '0.2.1': dict(),
                         }
    download_path = {'launchpad': 'lp:pyfftw',
                     None: 'http://launchpad.net/pyfftw/trunk/%(version)s/+download/PyFFTW3-%(version)s.tar.gz',
                     '0.2.1': 'http://sysbio.ioc.ee/download/software/sources/PyFFTW3-0.2.1.tar.gz',
                     }

    def quick_test(self):
        if self.path is not None:
            r = run_python(self.get('python path'), 'import %s as package' % (self.packagename), '')
            return not r[0]
        else:
            print '%s quick test skipped' % (self.title)

class IocbioPage(PythonPackagePage):
    
    depends = ['numpy', 'scipy', 'subversion', 'fftw3', 'mingw', 'libtiff']

    download_versions = {'svn': dict(),
                         '1.2.0': dict()}
    download_path = {'svn': 'http://iocbio.googlecode.com/svn/trunk/ iocbio-read-only',
                     None: 'http://iocbio.googlecode.com/files/iocbio-%(version)s.tar.gz'}

    def get_setup_install_command (self):
        return 'setup.py config_fc --fcompiler=gnu95 build --compiler=mingw32 install'

    def quick_test(self):
        if self.path is not None:
            r = run_python(self.get('python path'), 'import %s.version' % (self.packagename), self.version)
            return not r[0]
        else:
            print '%s quick test skipped' % (self.title)

    def update_environ(self):
        fldr = os.path.join(get_desktop_directory(), 'IOCBio Software')
        if not os.path.isdir(fldr):
            os.mkdir(fldr)
        
        scripts_dir = os.path.join(os.path.dirname(self.get('python path')), 'Scripts')
        ioc_scripts = glob.glob(os.path.join(scripts_dir, 'iocbio_*'))
        for fn in ioc_scripts:
            bn = os.path.basename(fn)
            b,e = os.path.splitext(bn)
            if b=='iocbio_nt_post_install':
                continue
            dst = os.path.join(fldr, "%s.lnk" % (b[7:].title().replace ('_',' ')))
            create_shortcut(dst, fn)

class BzrlibPage(PythonPackagePage):

    download_versions = {'2.3.1': dict (python=['2.4', '2.5', '2.6', '2.7']),
                         '2.2.3': dict (python=['2.4', '2.5', '2.6'])}
    download_path = {None:'http://launchpad.net/bzr/%(version:3)s/%(version)s/+download/bzr-%(version)s.win32-py%(python version:3)s.exe'}

    def quick_test(self):
        if self.path is not None:
            r = run_python(self.get('python path'), 'import %s as package; print(package.__version__)' % (self.packagename), self.version)
            if r[0]:
                return False
            r = run_command('bzr --version', env=self.environ)
            if not r[0]:
                v = r[1].split ('\n')[0]
                if v.startswith ('Bazaar (bzr)'):
                    v = v.rsplit (' ', 1)[-1]
                    return self.version==v
        else:
            print '%s quick test skipped' % (self.title)
