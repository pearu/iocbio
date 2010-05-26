#!/usr/bin/env python
""" Customized GUI to StrathKelvin program.

Requires StrathKelvin 929 Oxygen System software version 4.4.0.2 or higher.
"""
# Author: Pearu Peterson <pearu.peterson@gmail.com>
# Created: March 2010
from __future__ import division

import os
import time
import subprocess

mailslotname = r'\\.\mailslot\Strathkelvin_Output'
if os.name=='nt':
    from mailslot import MailSlot
else:
    from fakemailslot import MailSlot

from ..version import version as VERSION
from .model import Model

import matplotlib
matplotlib.interactive( True )
matplotlib.use( 'WXAgg' )

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from .menu_tools_wx import create_menus

import wx
import wx.gizmos as gizmos
import wx.lib.dialogs

class GlobalAttr:

    def _setattr_from_parents(self, name):
        if hasattr (self, name):
            return
        setattr (self, name, None)
        p = getattr (self, 'parent', None)        
        while p is not None:
            if hasattr (p, name):
                setattr (self, name, getattr(p, name))
                break
            p = getattr (p, 'parent', None)

        #if getattr (self, name) is None:
        #    print 'Failed to set %r for %s instance' % (name, self.__class__.__name__)

    def __init__(self, parent = None):
        self.parent = parent
        self._setattr_from_parents ('mailslot')
        self._setattr_from_parents ('model')
        self._setattr_from_parents ('statusbar')

    def NotifySizeChange(self):
        self.InvalidateBestSize()
        self.SendSizeEvent()

    def Entering (self):
        pass

    def Leaving (self):
        pass

    def info(self, message):
        self.statusbar.SetStatusText('%s' % (message))

    def warning(self, message):
        self.statusbar.SetStatusText('WARNING: %s' % (message))

    @staticmethod
    def force_stop(enable='get', _cache=[None]):
        if enable != 'get':
            _cache[0] = enable
        return _cache[0]

class Pages(wx.Notebook, GlobalAttr):

    def __init__(self, parent, *PageClasses):
        wx.Notebook.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)

        for Cls in PageClasses:
            page = Cls(self)
            self.AddPage(page, getattr (Cls, 'title', Cls.__name__))            

        self._old = 0
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanging)


    def OnPageChanging(self, event):
        new = event.GetSelection()
        old = self._old
        page = self.GetPage(new)
        prev_page = self.GetPage(old)
        prev_page.Leaving ()
        page.Entering ()
        if new > old:
            page.Populate()
        self._old = new
        event.Skip()

class MainFrame(wx.Frame, GlobalAttr):
    
    menu_defs = [
        dict(label = '&File',
             content = [
                None,
                dict(label = '&Quit\tCtrl-Q', help='Quit the program'),
                ]),
        dict(label = 'StrathKelvin',
             content = [
                dict(label = 'Start', help='Start StrathKelvin application.',
                     action = 'OnStrathKelvinStart'),
                dict(label = 'Interrupt', help='Interrupt receiving data from (possibly crashed) StrathKelvin application.',
                     action = 'OnStrathKelvinStop'),
                dict (label='Workflow help...',
                      help = 'Displays typical workflow of using this program.',
                      action = 'OnHelpStrathKelvin',
                      ),
                ]
             ),
        dict (label = 'Help',
              content = [
                dict(label = 'About...', 
                     help='About this software and configuring StrathKelvin software.',
                     action = 'OnStrathKelvinInfo'),
                dict (label='Protocols...',
                      help = 'Displays Protocols help.',
                      action = 'OnHelpProtocols',
                      ),
                dict (label='Configuration...',
                      help = 'Displays Configuration help.',
                      action = 'OnHelpConfiguration',
                      ),
                dict (label='Chambers...',
                      help = 'Displays Chambers help.',
                      action = 'OnHelpChambers',
                      ),
                dict (label='Measurements...',
                      help = 'Displays Measurements help.',
                      action = 'OnHelpMeasurements',
                      ),
                dict (label='Workflow...',
                      help = 'Displays typical workflow of using this program.',
                      action = 'OnHelpStrathKelvin',
                      ),
                ],
              )
        ]
    def __init__(self, parent, mytitle, mysize):

        wx.Frame.__init__(self, None, wx.ID_ANY, mytitle,
            size=mysize)
        GlobalAttr.__init__(self, parent)

        self.SetMenuBar(create_menus(self, self.menu_defs))

        self.CreateStatusBar()
        self.statusbar = self

        Pages(self, ProtocolsPage, ConfigurationPage, ChambersPage, MeasurementsPage)

    def OnQuit (self, event):
        #print 'OnQuit'
        self.Close()

    def OnStrathKelvinStart(self, event):
        if self.force_stop():
            self.force_stop(False)
            self.info('Resuming data retrival from StartKelvin application.')
            return
        strathkelvin_program = None
        for param in self.model.get_configuration():
            if param.name.lower()=='path_to_strathkelvin_program':
                strathkelvin_program = param.get_value()

        if strathkelvin_program is None or os.name!='nt':
            import thread
            thread.start_new_thread(start_fake_strathkelvin, ())
        else:
            # Fire up strathkelvin program
            subprocess.Popen(strathkelvin_program)
            self.info('Executed "%s"' % (strathkelvin_program))

    def OnStrathKelvinStop(self, event):
        self.force_stop(True)

    def OnStrathKelvinInfo (self, event):
        from wx.lib.wordwrap import wordwrap
        info = wx.AboutDialogInfo()
        info.Name = 'IOCBio.StrathKelvin'
        info.Description = wordwrap('''
This program, "IOCBio.StrathKelvin", is a wrapper of StrathKelvin 929 \
Oxygen System software.  The program can process and display \
experiment data during measurements by the StrathKelvin software.

Please report any bugs or feature requests to http://code.google.com/p/iocbio/issues/
''', 350, wx.ClientDC (self))
        info.Version = VERSION
        info.Copyright = "(C) 2010 Pearu Peterson"
        info.WebSite = ("http://sysbio.ioc.ee", "Laboratory of Systems Biology")
        info.Developers = ["Pearu Peterson <pearu.peterson@gmail.com>"]

        license_txt = '''
BSD

For full license text, see
  http://iocbio.googlecode.com/svn/trunk/LICENSE.txt
'''
        info.License = wordwrap (license_txt, 600, wx.ClientDC(self))
        wx.AboutBox(info)

    def OnHelpProtocols (self, event):
        help_text = getattr(ProtocolsPage, "help", "PFI")
        dlg = wx.lib.dialogs.ScrolledMessageDialog (self, help_text, "Protocols page help.", size=(500,600))
        dlg.ShowModal()

    def OnHelpConfiguration (self, event):
        help_text = getattr(ConfigurationPage, "help", "PFI")
        dlg = wx.lib.dialogs.ScrolledMessageDialog (self, help_text, "Configuration page help.", size=(500,600))
        dlg.ShowModal()

    def OnHelpChambers(self, event):
        help_text = getattr(ChambersPage, "help", "PFI")
        dlg = wx.lib.dialogs.ScrolledMessageDialog (self, help_text, "Chambers page help.", size=(500,600))
        dlg.ShowModal()

    def OnHelpMeasurements(self, event):
        help_text = getattr(MeasurementsPage, "help", "PFI")
        dlg = wx.lib.dialogs.ScrolledMessageDialog (self, help_text, "Measurements page help.", size=(500,600))
        dlg.ShowModal()

    def OnHelpStrathKelvin(self, event):
        help_text = """
Typical workflow
================

1.1 Start IOCBio.StrathKelvin program.

1.2 Select StrathKelvin/Start menu that will start StrathKelvin 929
    Oxygen System software. Note that the order of starting
    IOCBio.StrathKelvin and StrathKelvin 929 Oxygen System programs is
    important.

1.3 Make sure that all StrathKelvin 929 Oxygen System devices are
    turned on. Press Experiment button in the StrathKelvin 929 Oxygen
    System software.

1.4 Configure StrathKelvin 929 Oxygen System according to experiments
    setup while taking into account configuration notes below.

1.5 Configure IOCBio.StrathKelvin program: define protocols (see
    Help/Protocols menu), set configuration parameters (see
    Help/Configuration menu), save configuration, attach protocols to
    chambers (see Help/Chambers menu) and configure chambers
    parameters.

2.1 To start the experiments, press Start button in the StrathKelvin
    929 Oxygen System software. Switch to IOCBio.StrathKelvin program
    window and go to Measurments page where you will see graphs of
    recieved oxygen measurments (blue lines) and computed respiration
    rates (read lines). Markers can be added either by pressing
    chamber number key (1 to 6) or right clicking to the plot of a
    particular chamber (see Help/Measurements menu).

2.2 To stop the experiment, press Stop button in the StrathKelvin 929
    Oxygen System software. Then switch back to IOCBio.StrathKelvin
    program Measurements page and press 'Save results to ...'
    button. This will save oxygen measurments and computed respiration
    rates to the file indicated in the 'Save results to ...' button.

    Note that additional markers can be added also after stopping the
    experiment.

2.3 Repeat 2.1 and 2.2 for additional experiments.

3.1 To quit the programs, just exit both IOCBio.StrathKelvin and the
    StrathKelvin 929 Oxygen System programs.

Interrupting data acquisition
-----------------------------

When StrathKelvin 929 Oxygen System program has become unresponsive
then you can stop the experiment by selecting StrathKelvin/Interrupt
menu. This will able you to save the experiment data that has been
recieved so far.

Changing protocols during an experiment
---------------------------------------

It is allowed to change protocols during the experiment. For that go
to Chambers page and select a new protocol for a given chamber.  As a
result, the markers of the new protocol will be available in the
markers dialog. Note that all added markers (of the previous protocol)
will be still visible and saved to the results file.

It is also allowed to modify protocols during the experiment (to add
new tasks, for example).  For that go to Protocols page and modify
protocols. In order to apply the modifications to the running
experiment, you must reselect the modified protocols in the Chambers
page. Note that reselecting resets also the values of specified
parameters, so you must reenter the parameter values again.

Configuring StrathKelvin System software
========================================

To use IOCBio.StrathKelvin program, you must enable "Output data to
external program" (see under "Arrangement/Basic setup..." menu in
StrathKelvin 929 Oxygen System software available starting from
version 4.4.0.2.).

Note that the IOCBio.StrathKelvin program computes the respiration
rate itself. So, make sure that the StrathKelvin Oxygen System is NOT
emitting respiration rates as output data.

Make sure that the IOCBio.StrathKelvin program is using the same
oxygen units as the StrathKelvin Oxygen System software.  You can
specify the units changing ``oxygen_units`` parameter in Configuration
page.
"""
        dlg = wx.lib.dialogs.ScrolledMessageDialog (self, help_text, "StrathKelvin System configuration help.", size=(500,600))
        dlg.ShowModal()

class TasksCtrl (wx.Panel, GlobalAttr):

    def __init__(self, parent, ID, tasks):
        wx.Panel.__init__ (self, parent, ID)
        GlobalAttr.__init__(self, parent)

        self.elb = elb = gizmos.EditableListBox (self, wx.ID_ANY, "Tasks/parameters - select and press operation button")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add (elb, 1, wx.EXPAND|wx.ALL)
        self.SetSizer(sizer)

        self.lst_ctrl = lst_ctrl = elb.GetListCtrl()
        lst_ctrl.Bind (wx.EVT_LIST_END_LABEL_EDIT, self.OnEndLabelEdit)
        lst_ctrl.Bind (wx.EVT_LIST_DELETE_ITEM, self.OnItemDelete)

        elb.GetDownButton().Bind(wx.EVT_BUTTON, self.OnDown)
        elb.GetUpButton().Bind(wx.EVT_BUTTON, self.OnUp)

        self.tasks = []
        self.set_tasks (tasks)

    def set_tasks(self, tasks = None):
        if tasks is not None:
            self.tasks = tasks
        self.elb.SetStrings(self.tasks)

    def OnDown(self, event):
        event.Skip ()
        index = self.lst_ctrl.GetFirstSelected()
        if index<0 or index+1==len (self.tasks):
            return
        task = self.tasks[index]
        del self.tasks[index]
        self.tasks.insert(index+1, task)

    def OnUp(self, event):
        event.Skip ()
        index = self.lst_ctrl.GetFirstSelected()
        if index<=0:
            return
        task = self.tasks[index]
        del self.tasks[index]
        self.tasks.insert(index-1, task)

    def OnEndLabelEdit (self, event):
        index = event.GetIndex()
        text = event.GetText()
        if index==len(self.tasks):
            # new task
            self.tasks.append(text)
        else:
            self.tasks[index] = text
        event.Skip ()

    def OnItemDelete (self, event):
        index = event.GetIndex()
        del self.tasks[index]
        event.Skip ()

class ProtocolsPage(wx.Treebook, GlobalAttr):

    title = 'Protocols'

    help = '''
Protocols
=========

In the Protocols page one can create and modify protocols.

A protocol has a name, list of tasks (events), and user-defined
parameters.  Protocols can be attached to channels (chambers) to
define channels tasks and channel parameters. See Chambers page
for more information.

There exists special protocol, "_Configuration", that
is used to define general configuration parameters as well
as for creating new protocols.

Creating, copying, renaming, deleting protocols
-----------------------------------------------

To create a new protocol, right click to "_Configuration" label and
from a popup menu choose "New". A dialog window is opened for entering
the name of the new protocol.  The new protocol will appear in the
protocols column of the Protocols page that can be selected for
entering tasks and defining parameters.

To copy a protocol, right click to protocols name and select "Copy"
from the popup menu. This will create a new protocol that has copies
of all the tasks and parameters of the existing protocol.

To rename a protocol, right click to protocols name and select
"Rename".  This will open a dialog where you can enter the new name
for the protocol.

To delete a protocol, right click to protocols name and select
"Delete".

Adding, editing, ordering, and deleting protocols tasks
-------------------------------------------------------

To add a new task to a protocol, click on the "New Item" button and
enter task description to the task row. The task description can be
later edited by selecting the task row and clicking the "Edit Item"
button.  

To change the order of tasks, select a task row and click on the "Move
Down" or "Move Up" buttons.

To delete a task, select the corresponding task row and click
"Delete Item" button.

Adding protocols parameters
---------------------------

Adding parameters is similar to adding tasks. The difference is
that the definition of a parameter must start with `param:`.

The following syntax is used for defining parameters::

  <optional-type> <optional-choice-list> <parameter-name> = <optional-default-value>

Here `<optional-type>` can be `string`, `text`, `int`, `float`,
`file`, or `directory`. By default, the type of a parameter is
`string`. The `string` type is interpreted as one-line `text` type.

The `<optional-choice-list>` contains a comma separated list of
choices starting and ending with opening and closeing square brackets,
respectively.

The `<parameter-name>` specifies the name of the parameter.

The `<optional-default-value>` specifies the default value of the
parameter.

See "Examples Protocol" for more examples.

Notes
-----

All protocols will define `float volume_ml = 1` parameter.
'''


    def __init__ (self, parent):
        wx.Treebook.__init__(self, parent, wx.ID_ANY,
                             style = wx.BK_DEFAULT)
        GlobalAttr.__init__(self, parent)

        self.Populate()

        self.Bind(wx.EVT_TREE_ITEM_MENU, self.OnItemRightClick)

        self.popup = dict(rename=wx.NewId(), copy=wx.NewId(), delete=wx.NewId(),
                          new=wx.NewId(), refresh=wx.NewId())
        self.inv_popup = {}
        for op, Id in self.popup.items():
            self.inv_popup[Id] = op
            self.Bind(wx.EVT_MENU, self.OnItemRightClickPopup, id = Id)

    def OnItemRightClick (self, event):
        selection = self.GetSelection()
        protocol = self.GetPageText(selection)
        menu = wx.Menu()
        for op in sorted(self.popup):
            if protocol.startswith('_'):
                if op in ['delete', 'rename']:
                    continue
            else:
                if op=='new':
                    continue
            item = wx.MenuItem(menu, self.popup[op], op.title())
            menu.AppendItem(item)
        self.PopupMenu(menu)
        menu.Destroy()
        self.current_page = None

    def OnItemRightClickPopup(self, event):
        op = self.inv_popup[event.GetId()]
        selection = self.GetSelection()
        protocol = self.GetPageText(selection)
        if op=='copy':
            tasks = self.model.protocols[protocol]
            i = 0
            while 1:
                i += 1
                new_protocol = 'Copy %s of %s' % (i, protocol)
                if new_protocol not in self.model.protocols:
                    break
            new_tasks = tasks[:]
            self.model.protocols[new_protocol] = new_tasks
            self.Populate()
        elif op=='refresh':
            self.model.refresh()
            self.Populate()
        elif op=='delete':
            if protocol.startswith('_'):
                self.warning('Special protocol "%s" cannot be deleted' % (protocol))
            else:
                del self.model.protocols[protocol]
                self.Populate()
        elif op=='rename':
            dlg = wx.TextEntryDialog(self, 'Enter new protocol name:','Renaming protocol', protocol)
            if dlg.ShowModal() == wx.ID_OK:
                new_protocol = dlg.GetValue().strip()
                if not new_protocol:
                    self.warning('Not renaming the protocol "%s" to "%s" because name cannot have zero length' % (protocol, new_protocol))
                elif new_protocol not in self.model.protocols:
                    tasks = self.model.protocols[protocol]
                    del self.model.protocols[protocol]
                    self.model.protocols[new_protocol] = tasks
                    self.Populate()
                elif new_protocol==protocol:
                    pass
                else:
                    self.warning('Not renaming the protocol "%s" to "%s" because other protocol has this name' % (protocol, new_protocol))
            dlg.Destroy()
        elif op=='new':
            dlg = wx.TextEntryDialog(self, 'Enter new protocol name:','Creating new protocol', '<enter name for new protocol>')
            if dlg.ShowModal() == wx.ID_OK:
                new_protocol = dlg.GetValue().strip()
                if not new_protocol:
                    self.warning('Not creating a new protocol with name "%s" because name cannot have zero length' % (new_protocol))
                elif new_protocol not in self.model.protocols:
                    self.model.protocols[new_protocol] = []
                    self.Populate()
                else:
                    self.warning('Not creating a new protocol with name "%s" because other protocol has this name' % (new_protocol))
            dlg.Destroy()
        else:
            raise NotImplementedError(`op`)

    def Populate(self):
        self.DeleteAllPages()
        for protocol in sorted(self.model.protocols):
            win = TasksCtrl(self, wx.ID_ANY, self.model.protocols[protocol])
            self.AddPage(win, protocol)
        # resize:
        self.GetTreeCtrl().InvalidateBestSize()
        self.SendSizeEvent()

class Parameters(wx.Panel, GlobalAttr):

    def __init__(self, parent, protocol_or_channel):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)
        self.obj = protocol_or_channel

        self.container = wx.BoxSizer(wx.VERTICAL)

        title = self.obj if isinstance (self.obj, str) else 'Parameters'
        self.box = box = wx.StaticBox(self, wx.ID_ANY, title)
        bsizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        bsizer.Add(self.container, 0, wx.EXPAND|wx.ALL)
        self.SetSizer(bsizer)
        self.ids = {}
        self.Populate()

    def Populate(self):
        params = self.model.get_parameters(self.obj)
        self.container.DeleteWindows()
        self.ids.clear()
        if params is not None:
            sizer = wx.FlexGridSizer(cols=2, hgap=0, vgap=5)
            sizer.AddGrowableCol(1)
            for param in params:
                label = wx.StaticText(self, wx.ID_ANY, "%s:" % (param.name))
                value = param.get_value()
                Id = wx.NewId()
                self.ids[Id] = param
                widest_value = None
                if param.choices is not None:
                    widest_value = max([(len(p),p) for p in param.choices])[1]+'_'*5
                    ctrl = wx.Choice(self, Id, choices = param.choices)
                    if value is not None:
                        try:
                            i = param.choices.index(value)
                        except ValueError, msg:
                            #print 'value=%r, choices=%r, msg: %s' % (value,param.choices, msg)
                            i = 0
                    else:
                        i = 0
                        param.set_value(param.choices[i])
                    ctrl.SetSelection(i)
                    self.Bind(wx.EVT_CHOICE, self.OnChoiceChange, ctrl, id=Id)

                    ctrl.SetMaxSize((200,-1))
                elif param.type in ['file', 'directory']:
                    if value is not None:
                        ctrl = wx.Button(self, Id, value)
                    else:
                        ctrl = wx.Button(self, Id, 'Browse %s ...' % (param.type))
                    self.Bind(wx.EVT_BUTTON, self.OnBrowse, id=Id)
                else:
                    if param.type=='text':
                        ctrl = wx.TextCtrl(self, Id, '', style=wx.TE_MULTILINE,
                                           )
                    else:
                        ctrl = wx.TextCtrl(self, Id, '')
                        if param.type in ['int', 'float']:
                            ctrl.SetMaxSize((100,-1))
                        else:
                            ctrl.SetMinSize((200,-1))
                    self.Bind(wx.EVT_TEXT, self.OnTextChange, ctrl, id=Id)
                    if value is not None:
                        ctrl.ChangeValue(value)
                sizer.Add (label, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                sizer.Add (ctrl, 1, wx.EXPAND|wx.ALL)
            self.container.Add(sizer, 0, wx.EXPAND|wx.ALL)

        self.box.InvalidateBestSize()
        self.SendSizeEvent()

    def OnChoiceChange (self, event):
        Id = event.GetId()
        self.ids[Id].set_value(event.GetString().strip())

    def OnTextChange (self, event):
        Id = event.GetId()
        self.ids[Id].set_value(event.GetString().strip())


    def OnBrowse(self, event):
        Id = event.GetId()
        parameter = self.ids[Id]
        obj  = event.GetEventObject()
        if parameter.type=='file':
            dlg = wx.FileDialog (self, 'Choose a file for "%s"' % (parameter.name),
                                 defaultFile = parameter.get_value() or '',
                                 style = wx.FD_OPEN)

        elif parameter.type=='directory':
            dlg = wx.DirDialog (self, 'Choose a directory for "%s"' % (parameter.name),
                                 defaultPath = parameter.get_value() or '',
                                style = wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
        else:
            raise NotImplementedError (`parameter.type`)

        if dlg.ShowModal ()==wx.ID_OK:
            value = dlg.GetPath()
            parameter.set_value(value)
            obj.SetLabel(value)

        dlg.Destroy()

class ConfigurationPage (wx.Panel, GlobalAttr):

    title = 'Configuration'

    help = '''
Configuration
=============

In the "Configuration" page one can modify the values of configuration
parameters as defined in "_Configuration" protocol.  

See Protocols help page for more information.
'''

    def __init__ (self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)

        
        self.save_button_id = wx.NewId()

        self.Bind (wx.EVT_BUTTON, self.OnSaveConfiguration, id = self.save_button_id)

        self.container = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.container)
        self.Populate ()

    def Populate(self):
        self.container.DeleteWindows()
        for protocol in self.model.protocols:
            if protocol.startswith('_'):
                p = Parameters(self, protocol)
                self.container.Add(p, 1, wx.EXPAND|wx.ALL)

        save_button = wx.Button (self, self.save_button_id,
                                 "Save configuration and protocols")
        self.container.Add(save_button, 0, wx.EXPAND)

        self.NotifySizeChange()

    def OnSaveConfiguration (self, event):
        filename = self.model.save_protocols()
        self.info('Configuration and protocols saved to "%s"' % (filename))

class ChambersPage (wx.Panel, GlobalAttr):

    title = 'Chambers'

    help = '''
Chambers
========

In the "Chambers" page one can attach protocols to six chambers.
After attaching a protocol to a chamber, one can modify chambers
parameters as defined by the protocol.
'''

    def __init__ (self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)

        self.container = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.container)
        self.Populate ()


    def Populate(self):
        self.container.DeleteWindows()

        chambers = PanelSplitter(self, self.title, ChannelPanel, self.model.channels)
        self.container.Add(chambers, 1, flag=wx.EXPAND|wx.ALL)
        self.NotifySizeChange()
        
class PanelSplitter(wx.Panel, GlobalAttr):

    rows_cols_map = {
        0: (1,1),
        1: (1,1),
        2: (1,2),
        3: (1,3),
        4: (2,2),
        5: (2,3),
        6: (2,3),
        }

    def __init__ (self, parent, title, PageCls, page_args):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)

        box = wx.StaticBox(self, wx.ID_ANY, title)
        bsizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        rows, cols = self.rows_cols_map.get (len (page_args))
        self.panels = []
        sizer = wx.GridSizer (rows=rows, cols=cols, hgap=5, vgap=5)
        for index, page_arg in enumerate(page_args):
            page = PageCls(self, page_arg)
            sizer.Add(page, 1, flag=wx.EXPAND|wx.ALL)
            self.panels.append(page)

        bsizer.Add(sizer, 1, wx.EXPAND|wx.ALL)

        self.SetSizer (bsizer)


class MeasurementsPage(wx.Panel, GlobalAttr):
    
    title = 'Measurements'

    help = '''
Measurments
===========

In the Measurments page one can view the measurment results
from the StrathKelvin Oxygen System software. The results
include oxygen concentrations and oxygen respiration rates.

Note that pxygen respiration rates are computed by the
IOCBio.StrathKelvin program using linear regression algorithm on the
user-specified number of recent sample points. This number can be
specified by modifying ``rate_regression_points`` parameter in
Configuration page.

In addition, the Measurments page allows one to add markers (events)
to results. There are two ways to open markers dialog: either press
keys 1, 2, 3, 4, 5, or 6 in the keyboard that correspond to chamber
numbers, or right click to the corresponding axis of the chambers.

In the markers dialog one can specify the time stamp (seconds from the
start of the experiment), choose the pre-defined event or task, and
add a comment.


'''

    check_data_ms = 1000 # ms
    min_draw_ms = 1000 #ms

    def __init__ (self, parent):
        wx.Panel.__init__ (self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)
        self.data = []

        self.has_new_data = False

        self.timer_get_data = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnGetData, self.timer_get_data)
        
        self.timer_draw_data = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnDataChanged, self.timer_draw_data)

        self.timer_get_data.Start(self.check_data_ms)
        # timer_draw_data is started and stopped in OnDataChanged

        self.save_button = wx.Button(self, wx.ID_ANY, "Save measurement results")
        self.save_button.Enable(False)

        dpi = 80
        self.figure = Figure( dpi=dpi )

        self.canvas = FigureCanvas( self, -1, self.figure )


        self.Bind(wx.EVT_BUTTON, self.OnSave, self.save_button)

        self._resizeflag = True

        self.canvas.mpl_connect('button_press_event', self.onpress)
        self.canvas.mpl_connect('key_press_event', self.onkeypress)

        self.disable_draw = False
        self.experiment_title = None
        self.experiment_started = False
        self.have_axes = False

        self.ids = {}

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.ALL|wx.EXPAND)
        sizer.Add(self.save_button, 0, wx.EXPAND)
        self.SetSizer(sizer)

        self.axis_units = None

    def OnSave (self, event):
        self.model.save()
        self.save_button.Enable(False)

    def OnGetData(self, event):
        if self.force_stop() and self.experiment_started:
            self.info('Data retrieval stopped by user. Choose StarthKelvin/Start menu to resume.')
            self.save_button.Enable(True)
            return
        elif self.force_stop()==False:
            self.save_button.Enable(False)
            self.force_stop(None)

        messages = self.mailslot.read(timeout=1)
        while messages:
            message = messages.pop(0)
            if message.startswith('*** End'):
                self.experiment_started = False
                self.timer_get_data.Start(self.check_data_ms)
                self.timer_draw_data.Stop()
                self.OnDataChanged(None) # draw last results
                self.info('Experiment stopped: total number of samples=%s' % (self.data_index))
                self.save_button.Enable(True)
            elif not self.experiment_started:
                if messages:
                    dt = messages.pop(0)
                else:
                    dt = self.mailslot.read(1, timeout=1)[0]
                try:
                    dt = float(dt)
                except ValueError, msg:
                    self.error('Failed to convert dt=%r to float. Try stopping and starting the experiment.' % (dt))
                    raise
                self.experiment_title = message
                self.model.set_title(message)
                self.model.start()
                self.dt = dt
                self.model.init_slope(dt, s=10)
                update = int(dt*1e3/2.1)
                self.timer_get_data.Start(update)
                self.timer_draw_data.Start(max(self.min_draw_ms, update))
                self.experiment_started = True
                self.data_index = 0
                self.marks = {}
                self.create_axes()

                self.save_button.SetLabel('Save results to %s' % (self.model.channel_data_template.replace (r'%d','#')))
                self.save_button.Enable(False)

                self.info('Experiment started: title="%s", sample interval=%ss'\
                              % (message, dt))
            elif message:
                row = map(float, message.split())
                cols = len(row)
                t = self.dt * self.data_index
                for i,channel in enumerate(self.model.channels):
                    if i<cols:
                        channel.add_data(t, row[i])
                self.has_new_data = True
                self.data_index += 1

    def OnDataChanged(self, event, init_time = time.time()):
        if self.has_new_data:
            self.has_new_data = False
            self.draw()

    def onpress(self, event):
        if event.inaxes is None:
            #print 'onpress: click not in any axes'
            return
        chamber_index = self.axes2_lst.index(event.inaxes) + 1
        x,y = event.xdata, event.ydata
        if x is not None:
            channel = self.model.channels[chamber_index-1]
            t = channel.convert_time(x, inverse=True)
        #print 'onpress: clicked in axes %d at (x,y)=%s, %s' % (chamber_index, x, y)
        if event.button==3:
            if os.name=='nt':
                self.select_task_dialog(chamber_index, t)
            else:
                # on linux select_task_dialog will hang, could be matplotlib/wx bug.
                pass
        elif event.button==1:
            self.disable_draw = True
            if self.marks and t is not None:
                items = [(abs(t-xx),(ci,xx)) for ci,xx in self.marks if ci==chamber_index]
                if items:
                    item = min(items)[1]
                    self.info("Chamber %s event@%.3fsecs: %s" % (item[0], item[1], self.marks[item]))
            self.disable_draw = False
        return

    def select_task_dialog(self, channel_index, t):
        self.disable_draw = True
        dlg = SelectTaskDialog(self, channel_index, t)
        dlg.CenterOnScreen()
        val = dlg.ShowModal()
        if val==wx.ID_OK:
            #print 'ok'
            task = dlg.get_task()
            t = dlg.get_time()
            comment = dlg.get_comment()
            if comment:
                task = '%s [%s]' % (task, comment)
            self.model.add_channel_task(channel_index, t, task)
            if self.have_axes:
                self.marks[channel_index, t] = task
                self.draw_mark(channel_index, t, task)
                self.update_axes()
                self.canvas.draw()
                if not self.experiment_started:
                    self.save_button.Enable(True)
        else:
            #print 'cancel'            
            pass
        dlg.Destroy()        
        self.disable_draw = False

    def draw_mark(self, channel_index, t, task):
        if self.have_axes:
            channel = self.model.channels[channel_index-1]
            axes1 = self.axes1_lst[channel_index-1]
            t = channel.convert_time(t)
            line = axes1.axvline(t, color='g')
            
            ylim = axes1.get_ylim ()
            axes1.annotate(task, (t, 0.5*(ylim[0]+ylim[1])))

    def onkeypress(self, event):
        t = self.model.get_current_time()
        if t is None:
            return
        if event.key in '123456':
            self.select_task_dialog(int(event.key), t)

    def need_axis_update(self):
        if not self.axes1_lst:
            return True
        units = [self.model.get_axis_unit(axis=i) for i in range (3)]
        if units != self.axis_units:
            return True
        return False

    def update_axes(self):
        axis1range = self.model.get_axis_range(axis=1)
        axis2range = self.model.get_axis_range(axis=2)
        
        for axes1, axes2 in zip (self.axes1_lst, self.axes2_lst):
            axes1.set_ylim(axis1range)
            axes2.set_ylim(axis2range)

    def create_axes(self):
        self.axes1_lst = []
        self.axes2_lst = []
        self.line1_index_lst = []
        self.line2_index_lst = []
        #self.marks = {}

        self.figure.clear()
        title = str(self.experiment_title)
        self.figure.suptitle(title, fontsize=12)
        xlabel = self.model.get_axis_label(0)
        ylabel = self.model.get_axis_label(1)
        dylabel = self.model.get_axis_label(2).replace('_', ' ')
        for i,sp in enumerate([231,232,233,234,235,236]):
            axes1 = self.figure.add_subplot( sp )
            if i in [3,4,5]:
                axes1.set_xlabel(xlabel)
            if i in [0,3]:
                axes1.set_ylabel(ylabel, color='blue')
            axes1.set_title('Chamber %d' % (i+1))
            axes2 = axes1.twinx()
            if i in [2,5]:
                axes2.set_ylabel(dylabel, color='red')
            self.axes1_lst.append(axes1)
            self.axes2_lst.append(axes2)
            self.line1_index_lst.append(None)
            self.line2_index_lst.append(None)

            #axes2.invert_yaxis()
        self.figure.subplots_adjust(left=0.125-0.08, right=0.9+0.05,
                                    wspace = 0.15, hspace=0.15,
                                    top = 0.9, bottom=0.1-0.05)

        self.have_axes = True
        self.axis_units = [self.model.get_axis_unit(axis=i) for i in range (3)]

        for (channel_index, t), task in self.marks.items ():
            self.draw_mark(channel_index, t, task)

    def draw(self):
        from numpy.testing.utils import memusage
        print memusage ()

        if not self.have_axes:
            return
        if self.disable_draw:
            return

        if self.need_axis_update():
            self.create_axes()

        slope_n = self.model.get_slope_n()

        for index, channel in enumerate(self.model.channels):
            time_lst = channel.get_time()
            data_lst = channel.get_data()
            slope_lst = channel.get_data_slope(slope_n)
            if not time_lst:
                continue
            axes1 = self.axes1_lst[index]
            axes2 = self.axes2_lst[index]
            axes1.clear ()
            axes2.clear ()
            line1, = axes1.plot(time_lst, data_lst, 'b')
            line2, = axes2.plot(time_lst, slope_lst, 'r')

        self.update_axes()
        try:
            self.canvas.draw()
        except RuntimeError, msg:
            print '%s.draw: ignoring RuntimeError(%s)' % (self.__class__.__name__, msg)
        return

    def Populate(self):
        pass

    def Entering (self):
        self.canvas.SetFocus ()
        self.disable_draw = False
        self.draw()

    def Leaving (self):
        self.disable_draw = True
    

class SelectTaskDialog(wx.Dialog, GlobalAttr):

    def boxit (self, ctrl, title,expand=0):
        box = wx.StaticBox(self, wx.ID_ANY, title)
        bsizer = wx.StaticBoxSizer (box, wx.VERTICAL)
        if expand:
            bsizer.Add(ctrl,expand,wx.EXPAND|wx.ALL|wx.GROW)
        else:
            bsizer.Add(ctrl,0,wx.EXPAND)
        return bsizer

    def __init__(self, parent, channel_index, t):
        title = 'Select event for Chamber %s' % (channel_index)
        wx.Dialog.__init__(self, parent, wx.ID_ANY, title)
        GlobalAttr.__init__(self, parent)
        self.channel_index = channel_index
        self.t = t

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.time_ctrl = time_ctrl = wx.TextCtrl(self, wx.ID_ANY, '')
        self.time_ctrl.SetValue('%s' % (t))

        sizer.Add(self.boxit(time_ctrl, 'Specify time for event'),
                  0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)

        tasks, task = self.model.get_channel_tasks(channel_index)
        if tasks:
            self.tasks_ctrl = tasks_ctrl = wx.ListBox(self, wx.ID_ANY, choices=tasks,
                                                      style=wx.LB_SINGLE,
                                                      size=(200,300))
            if task is not None:
                tasks_ctrl.SetStringSelection(task)
            sizer.Add(self.boxit(tasks_ctrl,'Specify event',0),
                0, wx.GROW|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
            self.tasks_ctrl.SetFocus()
        else:
            self.tasks_ctrl = None
            t = wx.StaticText(self, wx.ID_ANY, "Chamber protocol does not define tasks.")
            sizer.Add(t)

        self.comment_ctrl = comment_ctrl = wx.TextCtrl(self, wx.ID_ANY)
        sizer.Add(self.boxit(comment_ctrl, 'Add a comment to event'),
                  0, wx.GROW, 5)

        line = wx.StaticLine(self, wx.ID_ANY, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)

        btnsizer = wx.StdDialogButtonSizer()
        self.ok_ctrl = btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()
        sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def get_task(self):
        if self.tasks_ctrl is None:
            return ''
        s = self.tasks_ctrl.GetSelection()
        if s==wx.NOT_FOUND:
            return ''
        return self.tasks_ctrl.GetString(s)

    def get_time(self):
        t = self.time_ctrl.GetValue().strip()
        if not t:
            return 0
        try:
            t = float(t)
        except Exception, msg:
            self.warning('Failed to evaluate time string %r, using %s' % (t, self.t))
            t = self.t
        return t

    def get_comment (self):
        return self.comment_ctrl.GetValue ()

class ChannelPanel(wx.Panel, GlobalAttr):
    def __init__(self, parent, channel):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        GlobalAttr.__init__(self, parent)

        self.channel = channel

        self.container = wx.BoxSizer(wx.VERTICAL)

        self.box = box = wx.StaticBox(self, wx.ID_ANY, 'Chamber %s' % (channel.index))
        bsizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        bsizer.Add(self.container, 0, wx.EXPAND|wx.ALL)
        self.SetSizer (bsizer)

        self.Bind (wx.EVT_COMBOBOX, self.OnSelect)

        protocols = [p for p in self.model.protocols if not p.startswith('_')]

        protocol_ctrl = wx.ComboBox(self, wx.ID_ANY, choices=sorted(protocols))
        if channel.protocol in protocols:
            protocol_ctrl.SetValue(channel.protocol)
        else:
            protocol_ctrl.SetValue('<select protocol>')

        self.protocol_ctrl = protocol_ctrl
        self.parameters_ctrl = Parameters(self, self.channel)

        self.container.Add(protocol_ctrl, 0, wx.EXPAND|wx.ALL)
        self.container.Add(self.parameters_ctrl, 0, wx.EXPAND|wx.ALL)
        self.Populate()

    def Populate(self):
        self.parameters_ctrl.Populate()
        self.NotifySizeChange()

    def OnSelect(self, event):
        protocol = self.protocol_ctrl.GetValue()
        #print 'OnSelect', protocol
        
        self.channel.set_protocol(protocol)

        self.Populate()
        self.parent.parent.NotifySizeChange()

def start_fake_strathkelvin():
    print 'Starting fake strathkelvin application.'
    # fake application:
    import random
    import time
    import math
    #time.sleep(4)
    sleep_time = 0.05
    mailslot = MailSlot(mailslotname, mode='w')
    mailslot.write('This is test experiment.')
    mailslot.write('%s' % (sleep_time))
    start_time = time.time()
    index = 0
    while time.time() < start_time + 60:
        index += 1
        time.sleep(sleep_time)
        row = [time.time()-start_time, index, random.random (),
               math.sin((time.time()-start_time) * 2*math.pi/10),
               math.sin(index*sleep_time * 2*math.pi/10),
               math.sin(index*sleep_time * 2*math.pi/10) + 0.1*random.random()
               ]
        mailslot.write (' '.join(map (str, row)))
    mailslot.write('*** End')
    
class App(wx.App):
    
    name = 'IOCBio-StrathKelvin'

    def __init__ (self, model, mailslot):
        self.model = model
        self.mailslot = mailslot
        wx.App.__init__(self, redirect=not True, filename='IOCBio-StrathKelvin.log')

    def RedirectStdio(self, filename=None):
        if filename:
            if not os.path.isabs (filename):
                filename = os.path.join(wx.StandardPaths.Get().GetUserDataDir(), filename)
            d = os.path.dirname (filename)
            if not os.path.isdir (d):
                os.makedirs(d)
            print 'Application std streams will be directed to ',filename
        return wx.App.RedirectStdio(self, filename)

    def OnInit(self):
        self.SetAppName(self.name)
        main_dir = wx.StandardPaths.Get().GetUserDataDir()
        self.model.init(main_dir)

        frame = MainFrame(self, self.name, (800, 600))
        self.SetTopWindow(frame)

        frame.Show()

        return True

def main():
    app = App(Model(), MailSlot(mailslotname, mode='r'))
    app.MainLoop()
    app.model.save_protocols()

if __name__ == '__main__':
    main()
