
import os
import cPickle as pickle

class Cache:

    def __init__ (self, filename):
        self.filename = filename
        self.data = {}

    def load(self):
        if os.path.isfile (self.filename):

            try:
                f = open (self.filename, 'rb')
                self.data = pickle.load(f)
                f.close()
            except EOFError, msg:
                print 'Loading %r failed with EOFError: %s' % (self.filename, msg)
                self.data = {}

        return self

    def dump (self):
        f = open (self.filename, 'wb')
        pickle.dump (self.data, f)
        f.close()
