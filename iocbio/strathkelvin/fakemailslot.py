"""Provides MailSlot class.

This module implements fake interface to mailslots and can
be used for testing mailslot applications under Linux.

Example
-------

  >>> from fakemailslot import MailSlot
  >>> reader = Mailslot(r'\\.\mailslot\example', mode='r')
  >>> writer = Mailslot(r'\\.\mailslot\example', mode='w')
  >>> writer.write('Hello')
  >>> print reader.read()
  ['Hello']

"""
# Author: Pearu Peterson
# Created: April, 2010

__all__ = ['MailSlot']
import os
import sys
import time

mailslot_test_dir = '/tmp'

class MailSlot:
    """MailSlot reader and writer.

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
        slot = os.path.join(mailslot_test_dir, slotname)
        if mode=='r':
            if os.path.exists(slot):
                for fn in os.listdir(slot):
                    os.remove(os.path.join(slot, fn))
            else:
                os.makedirs(slot)
        elif mode=='w':
            pass
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
          Specify timeout in seconds for reading a message.
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
        exact = maxcount is not None and timeout is not None
        index = -1
        while 1:
            index += 1
            times = [fn for fn in os.listdir(self.read_slot)]
            if times:
                times.sort()                
                fn = os.path.join(self.read_slot, times[0])
                f = open(fn, 'r')
                messages.append(f.read())
                f.close()
                os.remove(fn)
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
        fn = os.path.join(self.write_slot, '%020.6f' % (time.time()))
        f = open(fn, 'w')
        f.write(message)
        f.close()

    def __del__(self, os=os):
        if self.read_slot is not None:
            for fn in os.listdir(self.read_slot):
                os.remove(os.path.join(self.read_slot, fn))
            os.rmdir(self.read_slot)

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
