"""Provides create_menus function.
"""
# Author: Pearu Peterson
# Created: April, 2010

__all__ = ['create_menus']

import wx

def create_menus(parent, menu_defs):
    """ Create menu bar to parent using menu definition list.

    Parameters
    ----------
    parent : wx.Frame
      Specify frame where menu bar will be added.
    menu_defs : list
      Specify a list of dictionary objects that define menu bar structure.

    Returns
    -------
    menubar : wx.MenuBar

    Notes
    -----
    Menu bar structure is represented by a dictionary::

      dict (label = <menu label in menu bar>,
            content = <list of menu structure dicts or None>)

    where menu structure is a dictionary::

      dict (label = <menu label>,
            help = <menu help>,
            [kind = None|'check'|'radio']
            [content = None]
            )

    and ``None`` in content list represent separator line.

    Each menu structure will be bind to parent method with
    name ``On<Action>`` where ``<Action>`` substring is
    composed from label as follows::
    
      Action = label.replace ('&','').label.replace(' ','').split('\t')[0]

    That is, label may contain mnemonic shortcuts as well as
    accelerator shortcuts. In general, label can be in the following
    form::

      label = '&Action\t<Accel>-<Key>'

    where ``<Accel>`` can be one of the following strings::

      'Alt', 'Ctrl', 'Shift', 'Normal'
    
    or any combination of these joined with ``-``.  The ``<Key>`` can
    be any number or letter, or one of the following special words::

      back execute f1 numpad_space windows_left
      tab snapshot f2 numpad_tab windows_right
      return insert f3 numpad_enter windows_menu
      escape help f4 numpad_f1 special1
      space numpad0 f5 numpad_f2 special2 
      delete numpad1 f6 numpad_f3 special3
      lbutton numpad2 f7 numpad_f4 special4 
      rbutton numpad3 f8 numpad_home special5 
      cancel numpad4 f9 numpad_left special6
      mbutton numpad5 f10 numpad_up special7
      clear numpad6 f11 numpad_right special8
      shift numpad7 f12 numpad_down special9
      alt numpad8 f13 numpad_prior special10
      control numpad9 f14 numpad_pageup special11
      menu multiply f15 numpad_next special12
      pause add f16 numpad_pagedown special13
      capital separator f17 numpad_end special14
      prior subtract f18 numpad_begin special15
      next decimal f19 numpad_insert special16
      end divide f20 numpad_delete special17
      home numlock f21 numpad_equal special18
      left scroll f22 numpad_multiply special19
      up pageup f23 numpad_add special20
      right pagedown f24 numpad_separator
      down numpad_subtract
      select numpad_decimal
      print numpad_divide

    The kind key can be used to create check or radio menus.
    The kind key is automatically applied to menu structures
    in content list when submenus do not define the kind key.

    Examples
    --------

    ::

      create_menus(<wx.Frame instance>, [
        dict(label='File',
             content=[
               dict(label='Open', help='Open File'),
               dict(label='Close', help='Close File'),
               dict(label='Quit', help='Quit program'),
             ])
        dict(label='Help',
             content=[dict(label='About')]) 
            ])

    """
    menubar = wx.MenuBar()

    accel_list = []

    for menu_def in menu_defs:
        menu = wx.Menu()
        
        for submenu_def in menu_def.get('content',[]):
            if submenu_def is None:
                menu.AppendSeparator()
                continue
            menu_append(parent, menu, submenu_def, accel_list)

        menubar.Append(menu, menu_def['label'])

    if accel_list:
        parent.SetAcceleratorTable (wx.AcceleratorTable(accel_list))

    return menubar

def menu_append(parent, menu, menu_def, accel_list):
    # TODO: when label ends with `...` then interpret it as dialog box
    if menu_def is None:
        menu.AppendSeparator()
        return
    label = menu_def['label']
    help = menu_def.get('help','')
    content = menu_def.get('content')
    kind = menu_def.get('kind')
    action_method = menu_def.get('action')

    if content is not None:
        submenu = wx.Menu()
        for submenu_def in content:
            if 'kind' not in submenu_def:
                submenu_def['kind'] = kind
            menu_append (parent, submenu, submenu_def, accel_list)
        menu.AppendMenu(wx.ID_ANY, label, submenu)            
        return

    if kind=='check':
        menu.AppendCheckItem(wx.ID_ANY, menu_def['label'])
        return
    if kind=='radio':
        menu.AppendRadioItem(wx.ID_ANY, menu_def['label'])
        return

    item = menu.Append (wx.ID_ANY, label, help)
    action = label.replace ('&', '')
    accel = None
    if '\t' in label:
        action, accel = action.split('\t')
        accel = accel.strip().replace('+','-').upper()

    if action_method is None:
        #TODO: Add parent information to action_method
        action_method = 'On%s' % (action.strip().title().replace(' ',''))

    if hasattr(parent, action_method):
        parent.Bind(wx.EVT_MENU, getattr (parent, action_method), item)
    else:
        menu.SetHelpString(item.GetId(), '%s --- NOT IMPLEMENTED: %s.%s method' % (help, parent.__class__.__name__, action_methoc))
        print 'Warning: unimplemented method %s.%s' % (parent.__class__.__name__, action_method)

    if accel is not None:
        ctrl, key = accel.rsplit('-',1)
        flags = 0
        for flag in ctrl.split('-'):
            flags |= getattr(wx, 'ACCEL_'+flag)
        if len (key)==1:
            keycode = ord (key)
        else:
            keycode = getattr(wx, 'WXK_%s' % (key.upper()))
        accel_list.append((flags, keycode, item.GetId()))
