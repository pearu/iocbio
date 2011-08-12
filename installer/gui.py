
# Author: Pearu Peterson
# Created: Apr 2011

import os
import time
import sys
import wx
import wx.wizard as wiz
import subprocess
import urllib2
import urlparse
import shutil
import tempfile
import platform

from utils import run_command, download, start_installer, extract, winreg_append_to_path, isadmin

class Model(wx.Frame):

    page_classes = {}
    
    def __init__ (self, 
                  logfile=None,
                  working_dir = None):
        if working_dir is not None:
            if not os.path.isdir (working_dir):
                os.makedirs(working_dir)
            print 'chdir',working_dir
            os.chdir(working_dir)
        self.working_dir = working_dir or '.'
        if logfile is None:
            self.app = wx.App(redirect=False)
        else:
            logfile = os.path.abspath (logfile)
            if os.path.isfile (logfile):
                os.remove(logfile)
            print 'All output will be redirected to %r' % (logfile)
            print 'When finished, press ENTER to close this program...'
            self.app = wx.App(redirect=True, filename=logfile)
        self.logfile = logfile
        print 'time.ctime()->%r' % (time.ctime())
        print 'sys.executable=%r' % (sys.executable)
        print 'sys.path=%r' % (sys.path)
        print 'sys.platform=%r' % (sys.platform)
        print 'os.name=%r' % (os.name)
        print 'platform.uname()->%r' % (platform.uname(),)
        print 'os.environ["PATH"]=%r' % (os.environ["PATH"])
        print 'os.getcwd()->%r' % (os.getcwd())
        print 'isadmin()->%r' % (isadmin())

        wx.Frame.__init__(self, None, -1)
        self.Bind(wiz.EVT_WIZARD_PAGE_CHANGED, self.OnWizPageChanged)
        self.Bind(wiz.EVT_WIZARD_PAGE_CHANGING, self.OnWizPageChanging)

        #nb = wx.Notebook(self, -1)
        wizard = wiz.Wizard(self, -1, __file__)
        wizard.SetPageSize ((600, 500))
        wizard.model = self
        self.wizard = wizard
        #nb.AddPage(wizard, 'Installer wizard')
        #nb.AddPage(wizard, 'Installer log')
        #self.environ = os.environ.copy()

    @classmethod
    def _get_dependencies(cls, r):
        deps = []
        page_cls = eval(r.title()+'Page', cls.page_classes)
        for rs in page_cls.depends:
            for d in cls._get_dependencies(rs):
                if d not in deps:
                    deps.append(d)
            if rs not in deps:
                deps.append (rs)
        return deps

    def run(self, resources):
        new_resources = []
        for r in resources:
            for d in self._get_dependencies(r):
                if d not in new_resources:
                    new_resources.append(d)
            if r not in new_resources:
                new_resources.append(r)
        resources = new_resources
        pages = []
        software_list = []
        for r in resources:
            page_cls = eval(r.title()+'Page', self.page_classes)
            page = page_cls(self.wizard)
            if page.skip ():
                print '%s.run: Skipping %s' % (self.__class__.__name__, page_cls.__name__)
                continue
            if pages:
                pages[-1].SetNext(page)
                page.SetPrev(pages[-1])
            pages.append(page)
            software_list.append (page.title)

        page = TitlePage (self.wizard, software_list)

        if pages:
            pages[0].SetPrev(page)
            page.SetNext(pages[0])
        pages.insert(0, page)

        page = FinalPage(self.wizard)
        if pages:
            pages[-1].SetNext(page)
            page.SetPrev(pages[-1])
        pages.append(page)

        self.pages = pages
        self.resources = resources

        self.wizard.RunWizard(pages[0])


    def OnWizPageChanged(self, evt):
        if evt.GetDirection():
            dir = "forward"
        else:
            dir = "backward"

        page = evt.GetPage()
        #print "OnWizPageChanged: %s, %s\n" % (dir, page.__class__)

        if dir=='backward':
            page.reset_resource()
        else:
            prev = page.GetPrev()
            if prev is not None:
                page.environ.update(prev.environ)

        page.set_resource_options()

    def OnWizPageChanging(self, evt):
        if evt.GetDirection():
            dir = "forward"
        else:
            dir = "backward"

        page = evt.GetPage()
        #print "OnWizPageChanging: %s, %s\n" % (dir, page.__class__)

        if dir=='forward':
            infosource = self.logfile or 'terminal'

            page.apply_resource_message = ''
            page.start_apply_selection()
            try:
                r = page.apply_resource_selection()
            except:
                message = page.apply_resource_message
                if message:
                    print message
                wx.MessageBox("Failed to apply resource selection:\n\t%s\nSee %s for information." % (message, infosource), "Cancelling Next")
                evt.Veto()
                page.set_resource_options()
                page.stop_apply_selection()
                raise
            if not r:
                message = page.apply_resource_message
                if message:
                    print message
                wx.MessageBox("Failed to apply resource selection:\n\t%s\nSee %s for information." % (message, infosource), "Cancelling Next")
                evt.Veto()
                page.set_resource_options()
            else:
                if not page.quick_test():
                    page.quick_test_message = 'TEST FAILED'
                    wx.MessageBox("%s quick test failed, see %s for information." % (page.title, infosource), "Warning")
                else:
                    page.quick_test_message = 'TEST OK'
            page.stop_apply_selection()

class PopupMessage (wx.PopupTransientWindow):

    def __init__(self, parent, message):
        wx.PopupTransientWindow.__init__(self, parent, wx.SIMPLE_BORDER)
        self.SetBackgroundColour("#FFB6C1")
        st = wx.StaticText(self, -1, message, pos=(10,10))
        sz = st.GetBestSize()
        self.SetSize( (sz.width+20, sz.height+20) )

class PageMetaclass (type):
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        Model.page_classes[name] = cls
        return cls

class WizardPage(wiz.WizardPageSimple):

    __metaclass__ = PageMetaclass

    def __init__(self, parent):
        self.model = parent.model
        wiz.WizardPageSimple.__init__(self, parent)    
        self.environ = os.environ.copy()
        try: del self.environ['PYTHONPATH']
        except KeyError: pass
        self.reset_resource()

    def skip (self):
        return False

    def update_environ_PATH(self, paths):
        if isinstance(paths, (str, unicode)):
            paths = paths.split(os.pathsep)
        current_paths = self.environ.get('PATH', [])
        if isinstance(current_paths, (unicode, str)):
            current_paths = current_paths.split(os.pathsep)
        for p in paths:
            if p not in current_paths:
                current_paths.append(p)
        self.environ['PATH'] = os.pathsep.join(current_paths)

    def get_environ_string (self):
        l = []
        for k in sorted (self.environ):
            l.append ('%s = %s' % (k, self.environ[k]))
        return '\n\t'.join(l)

    @property
    def title(self):
        return self.__class__.__name__[:-4]    

    def set_resource_options(self):
        pass

    def apply_resource_selection (self): return True
    def get_resource_info (self): return ''
    def reset_resource (self):
        self.path = None
        self.version = None
        self.quick_test_message = ''
        self.apply_resource_message = ''
    def start_apply_selection (self): pass
    def stop_apply_selection (self): pass
    def quick_test(self): return True

class TitlePage(WizardPage):
    
    def __init__(self, parent, software_list=[]):
        WizardPage.__init__(self, parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        title = wx.StaticText(self, -1, 'Welcome to IOCBio installer')
        title.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        sizer.Add(title, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND|wx.ALL, 5)

        for name in ['PATH', 'FFTW_PATH', 'COMSPEC']:
            value = os.environ.get (name)
            if value is not None:
                self.environ[name] = value

        message = '''
The IOCBio project provides open-source software that is developed in Laboratory of Systems Biology at Institute of Cybernetics, see
http://sysbio.ioc.ee

This installer wizard will help you to install IOCBio software and all of its prerequisites. The installer will step through a list of software (shown below) and ask you to make a selection of desired software versions. When needed, the software components will be installed by their native installers. So, be ready to step through of various installer wisards.

List of software:
\t%s

For IOCBio software sources, documentation and support, see 
  http://code.google.com/p/iocbio/

Log messages are saved to:
\t%s

Current environment:
\t%s
''' % ('\n\t'.join (software_list), 
       self.model.logfile,
       self.get_environ_string())

        if self.model.logfile:
            print message

        st = wx.TextCtrl(self, -1, message, style=wx.TE_MULTILINE|wx.TE_READONLY)
        sizer.Add(st, 1, wx.EXPAND, 5)

    def apply_resource_selection (self):
        if isadmin():
            print 'You are Administrator'
            return True
        self.apply_resource_message = 'This program must be run as Administrator'

class FinalPage(WizardPage):

    def __init__(self, parent):
        WizardPage.__init__(self, parent)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        title = wx.StaticText(self, -1, 'Congratulations!')
        title.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        sizer.Add(title, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND|wx.ALL, 5)

        self.text_ctrl = wx.TextCtrl(self, -1, '', style=wx.TE_MULTILINE|wx.TE_READONLY)
        sizer.Add(self.text_ctrl, 1, wx.EXPAND, 5)

    def set_resource_options(self):
        message = '''
You have succesfully installed the IOCBio software with the following software selection:

%s

For IOCBio software sources, documentation and support, see
  http://code.google.com/p/iocbio/

See log messages in:
\t%s

Current environment:
\t%s

Clicking Finish will update systems PATH environment variable.

*** Important note for Windows users ***
\tComputer restart is required for PATH changes to become effective.
''' % (self.GetPrev().get_current_state_message (prev=self), 
       self.model.logfile,
       self.get_environ_string ())
        self.text_ctrl.SetValue(message)
        if self.model.logfile:
            print message
        return True

    def apply_resource_selection(self):
        if winreg_append_to_path(self.environ['PATH']):
            return True
        self.apply_resource_message = 'Failed to update winreg PATH: make sure you run as Administrator'

    def stop_apply_selection(self):
        self.model.app.RestoreStdio()

class ResourcePage (WizardPage):

    depends = []
    download_versions = []
    download_extensions = {None:''}
    download_path = {}
    export_attrs = []

    def get_install_path(self, installer):
        if installer.endswith ('.tar.gz'):
            return os.path.basename (installer)[:-7]
        print 'Warning: %s.get_install_path (%r) not implemented' % (self.__class__.__name__, installer)
        return

    def __init__(self, parent):
        WizardPage.__init__(self, parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        title = wx.StaticText(self, -1, '%s configuration' % (self.title))
        title.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        sizer.Add(title, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND|wx.ALL, 5)

        prev_selection_button = wx.Button(self, -1, 'Show previous selections')
        sizer.Add(prev_selection_button, 0, wx.EXPAND, 5)
        self.Bind(wx.EVT_BUTTON, self.OnShowSelections, prev_selection_button)

        select_title = wx.StaticText(self, -1, 'Select %s resource option' % (self.title))
        sizer.Add(select_title, 0, wx.ALIGN_LEFT|wx.ALL, 5)
        self.select_list = wx.ListBox (self, -1, style=wx.LB_SINGLE, name='Select %s' % (self.title))
        sizer.Add(self.select_list, 1, wx.EXPAND, 5)

        self.status = wx.StatusBar(self, -1)

        sizer.Add(self.status, 0, wx.EXPAND, 5)

        self.sizes = sizer
        self.stop_apply_selection()


    def start_apply_selection(self):        
        self.status.SetStatusText('Processing selection may take awhile ... Please wait! Wait even when window is reported as "Not responding"')
    def stop_apply_selection (self):
        self.status.SetStatusText('Please select and press Next.')

    @property
    def title(self):
        return self.__class__.__name__[:-4]

    def get_current_state_message(self, prev=None):
        messages = []
        for k in sorted(self.environ):
            v = self.environ[k]
            if k=='PATH':
                # in windows PATH may be very long..
                l = ''
                line = ''
                for p in v.split (os.pathsep):
                    if len (line)<100:
                        line = os.pathsep.join([line, p])
                    else:
                        if l:
                            l += ';\n' + line
                        else:
                            l = line
                        line = ''
                if line:
                    if l:
                        l += ';\n' + line
                    else:
                        l = line
                v = l
            messages.append ('  %s = %s' % (k,v))
        messages.append('Current environment state:')

        if prev is None:
            prev = self.GetPrev()            
        while prev is not None:
            messages.append (prev.get_resource_info())
            prev = prev.GetPrev()
        messages.append('Current software state:')

        messages = '\n'.join (reversed(messages))
        return messages.strip()

    def OnShowSelections(self, evt):
        win = PopupMessage(self, self.get_current_state_message())
        btn = evt.GetEventObject()
        pos = btn.ClientToScreen( (0,0) )
        sz =  btn.GetSize()
        win.Position(pos, (0, sz[1]))
        win.Popup()

    def get(self, name):
        if ' ' in name:
            title, name = name.split(' ',1)
        else:
            title = ''
        title = title.lower().strip()
        name = name.strip()
        attr = None
        if self.title.lower()==title or not title:
            attr = getattr (self, name, None)
        if attr is None:
            prev = self.GetPrev()            
            while prev is not None:
                if prev.title.lower()==title or not title:
                    attr = getattr (prev, name, None)
                    if attr is not None:
                        break
                prev = prev.GetPrev()
        if attr is None:
            raise AttributeError('no attribute or resource found: %s %s' % (title, name))
        return attr

    def set_resource_options(self):
        self.selections, self.selection_task_map = self.get_resource_options()
        self.select_list.Set(self.selections)
        if self.selections:
            self.select_list.SetSelection(0)
        else:
            self.status.SetStatusText('Please go Back and try other version selections')

    def get_resource_options(self, labels=None, tasks=None):
        if labels is None:
            labels = []
        if tasks is None:
            tasks = {}

        for v in self.download_versions:
            download_path = self.download_path.get (v, self.download_path.get(None, ''))
            if download_path:
                ext = self.download_extensions.get(v, self.download_extensions.get(None,''))
                deps = dict(version=v, ext=ext)
                deps['version-'] = v.replace ('.','-')
                installer = download_path % deps
                if installer.endswith ('.tar.gz'):
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

    def get_resource_selection (self):
        l = []
        for i in self.select_list.GetSelections():
            l.append (self.selections[i])
        return l

    def try_resource(self, version=None):
        print '%s.try_resource not implemented' % (self.__class__.__name__)
        return None

    def update_environ(self):
        pass

    def apply_resource_selection(self):
        selections = self.get_resource_selection ()
        if not selections:
            self.apply_resource_message = 'Nothing selected for %s' % (self.title)
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
            if not self.get_and_run_installer(installer):
                self.apply_resource_message = 'Failed to download and run installer: %r' % (installer)
                return False

            path = self.try_resource(version)
            if path is not None:
                self.path = path
                self.version = version
                self.update_environ()
                return True
            self.apply_resource_message = 'Installation from binary failed: %s.try_resource(%r) returned None' % (self.__class__.__name__, version)
            return False

        if task[0]=='get_source':
            source = task[1]
            version = task[2]
            if not self.get_and_install_source(source):
                self.apply_resource_message = 'Failed to get and install source: %r' % (source)
                return False
            path = self.try_resource(version)
            if path is not None:
                self.path = path
                self.version = version
                self.update_environ()
                return True
            self.apply_resource_message = 'Installation from source failed: %s.try_resource(%r) returned None' % (self.__class__.__name__, version)
            return False
        raise NotImplementedError (`task`)

    def get_resource_info (self):
        if self.path is not None:
            return '*** %s %s (%s) %s' % (self.title, self.version, self.path, self.quick_test_message)
        return '*** %s (unknown)' % (self.title)

    def get_and_run_installer (self, installer):
        installer_exe = os.path.abspath(os.path.basename (installer))
        if not os.path.isfile (installer_exe):
            print 'Downloading', installer, '..',
            installer_exe = download (installer)
            if installer_exe is None:
                print 'Download FAILED'
                return False
            print 'DONE'
        if os.path.splitext(installer_exe)[-1] in ['.zip']:
            install_path = self.get_install_path(installer_exe)
            if install_path is not None:
                if not os.path.isdir (install_path):
                    os.makedirs (install_path)
                return bool(extract(installer_exe, install_path))
        elif not start_installer(installer_exe):
            print 'Failed to start', installer_exe
            return False
        return True

    def get_source_path (self, source_file):
        if source_file.endswith ('.tar.gz'):
            return os.path.basename(source_file)[:-7].split('-')[0]

    def get_and_install_source(self, source):
        source_file = os.path.abspath(os.path.basename (source))
        if not os.path.isfile (source_file):
            print 'Downloading', source, '..',
            source_file = download (source)
            if source_file is None:
                print 'Download FAILED'
                return False
            print 'DONE'
        source_path = self.get_source_path(source_file)
        if source_path is not None:
            if not os.path.isdir (source_path):
                os.makedirs (source_path)
        else:
            source_path = '.'
        content = extract(source_file, source_path)
        if not content:
            return False
        cwd = source_path
        for p in content:
            if os.path.isdir(p):
                cwd = p
                break
        return self.install_source(os.path.abspath(cwd))
