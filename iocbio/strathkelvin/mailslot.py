"""Provides MailSlot class.

Example
-------

  >>> from mailslot import MailSlot
  >>> reader = Mailslot(r'\\.\mailslot\example', mode='r')
  >>> writer = Mailslot(r'\\.\mailslot\example', mode='w')
  >>> writer.write('Hello')
  >>> print reader.read()
  ['Hello']

"""
# Author: Pearu Peterson
# Created: April, 2010

__all__ = ['MailSlot']
import sys
import time
import ctypes
try:
    from win32file import *
except ImportError:
    # wine support:
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    NULL = 0x0
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = -1
    MAILSLOT_WAIT_FOREVER = -1
    FORMAT_MESSAGE_FROM_STRING = 0x00000400
    MAILSLOT_NO_MESSAGE = -1

CreateFile = ctypes.windll.kernel32.CreateFileA 
ReadFile = ctypes.windll.kernel32.ReadFile
WriteFile = ctypes.windll.kernel32.WriteFile
CloseHandle = ctypes.windll.kernel32.CloseHandle
CreateMailslot = ctypes.windll.kernel32.CreateMailslotA 
GetLastError = ctypes.windll.kernel32.GetLastError
GetMailslotInfo = ctypes.windll.kernel32.GetMailslotInfo
c_int = ctypes.c_int
c_char_p = ctypes.c_char_p
pointer = ctypes.pointer

class MailSlot:
    """MailSlot reader and writer.

    Notes
    -----
    Failure to read or write will trigger IOError containing error code.
    Error code values are explained in:

      http://msdn.microsoft.com/en-us/library/ms681381%%28VS.85%%29.aspx

    See also
    --------
    __init__
    """

    def __init__(self, slotname, mode='r'):
        """ Create MailSlot instance and the mailslot.

        Parameters
        ----------
        slotname : str
          Specify the name of mailslot.
        mode : {'r', 'w'}
          Specify the reading or writing mode of the mailslot, respectively.

        See also
        --------
        create
        """
        self.slotname = slotname
        self.mode = mode
        self.read_slot = None
        self.write_slot = None
        
        slot = self.create(slotname, mode)

        if mode == 'r':
            self.read_slot = slot
        elif mode == 'w':
            self.write_slot = slot

    def create(self, slotname, mode):
        """Create a mailslot for reading or writing.

        Parameters
        ----------
        slotname : str
        mode : {'r', 'w'}

        Returns
        -------
        slot : int
          Mailslot descriptor.

        Notes
        -----
        The mailslot for reading must be created before the mailslot
        for writing.

        See also
        --------
        __init__, read, write
        """

        if mode=='r':
            slot = CreateMailslot(slotname, 
                                  0,
                                  MAILSLOT_WAIT_FOREVER,
                                  NULL)

            if slot==INVALID_HANDLE_VALUE:
                error_id = GetLastError()
                error_msg = FormatError(error_id)
                #print self.get_error_message(error_id)
                raise IOError('Creating mailslot %r failed with status: %s (%s)'\
                              % (slotname, error_id, error_msg))
        elif mode=='w':
            slot = CreateFile(slotname, 
                              GENERIC_WRITE, 
                              FILE_SHARE_READ,
                              NULL, 
                              OPEN_EXISTING, 
                              FILE_ATTRIBUTE_NORMAL, 
                              NULL)
            if slot==INVALID_HANDLE_VALUE:
                error_id = GetLastError()
                error_msg = FormatError(error_id)
                raise IOError('Creating file %r failed with status: %s (%s)'\
                              % (slotname, error_id, error_msg))
        else:
            raise NotImplementedError(`mode`)
        return slot

    def read(self, maxcount=None, timeout=None):
        """Read messages currently available in the mailslot.

        Parameters
        ----------
        maxcount : {int, None}
          Specify the maximal number of messages to read. If maxcount
          is None then read until no messages will be in mailslot.
          Note that if the rate of creating new messages is higher
          than the rate of reading messages then this method may
          never complete.
        timeout : {int, None}
          Specify timeout in seconds for reading one message.
          When specified together with maxcount, the method tries
          to read exactly maxcount messages.

        Returns
        -------
        messages : list
          A list of message strings.

        See also
        --------
        read, create
        """
        if self.read_slot is None:
            raise TypeError('MailSlot instance not open for reading')
        messages = []
        if maxcount==0:
            return messages
        next_size = c_int(0)
        message_count = c_int(0)
        bytes_read = c_int(0)
        exact = maxcount is not None and timeout is not None
        index = -1
        while 1:
            index += 1
            success = GetMailslotInfo(self.read_slot, NULL,
                                      pointer(next_size), pointer(message_count), NULL)
            if not success:
                error_id = GetLastError()
                error_msg = FormatError(error_id)
                raise IOError('getting info from %r failed with status: %s (%s)'\
                              % (self.slotname, error_id, error_msg))

            if next_size.value!=MAILSLOT_NO_MESSAGE:
                buf = c_char_p(' '*next_size.value)
                success = ReadFile(self.read_slot, buf, next_size, pointer(bytes_read), NULL)
                if not success:
                    error_id = GetLastError()
                    error_msg = FormatError(error_id)
                    raise IOError('reading message from %r failed with status: %s (%s)'\
                                  % (self.slotname, error_id, error_msg))                
                assert bytes_read.value==next_size.value,`bytes_read, next_size, status`
                messages.append(buf.value)
                success = True
            else:
                success = False
            if len(messages)==maxcount:
                break
            if exact:
                if success or index==0:
                    time.sleep(timeout)
                    continue
                else:
                    break
            if success:
                continue
            break
        return messages

    def write(self, message):
        """ Write message to mail slot.

        Parameters
        ----------
        message : str
          Specify message to be written to mailslot.

        See also
        --------
        read, create
        """
        if self.write_slot is None:
            raise TypeError('MailSlot instance not open for writing')
        bytes_written = c_int(0)
        length = len(message)
        success = WriteFile(self.write_slot, 
                           message,
                           length,
                           pointer(bytes_written),
                           NULL)
        if not success:
            error_id = GetLastError()
            error_msg = FormatError(error_id)
            raise IOError('writing message to %r failed with status: %s (%s)'\
                          % (self.slotname, error_id, error_msg))
        assert bytes_written.value==length,`bytes_written, length, status`

    def __del__(self, CloseHandle=CloseHandle):
        #if self.read_slot is not None:
        #    CloseHandle(self.read_slot)
        if self.write_slot is not None:
            CloseHandle(self.write_slot)
        

def _test():
    reader = MailSlot(r'\\.\mailslot\test', mode='r')
    writer = MailSlot(r'\\.\mailslot\test', mode='w')

    messages = reader.read()
    assert messages==[],`messages`
    
    writer.write('hey')
    messages = reader.read()
    assert messages==['hey'],`messages`

    writer.write('hey')
    writer.write('hoo')
    messages = reader.read()
    assert messages==['hey', 'hoo'],`messages`

    writer.write('tere')
    writer.write('tsau')
    messages = reader.read(1)
    assert messages==['tere'],`messages`
    messages = reader.read(1)
    assert messages==['tsau'],`messages`

    print 'trying to read one message during 1 second..'
    messages = reader.read(1, timeout=1)
    print 'done', messages

    print 'test ok'
    
if __name__=='__main__':
    _test()
