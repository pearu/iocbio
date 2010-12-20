import time
from iocbio.utils import time2str, bytes2str
from numpy.testing import memusage

class TimeIt(object):

    def __init__ (self, viewer, title):
        self.title = title
        self.viewer =  viewer
        viewer.status = '%s..' % (title)
        self.start_memusage = memusage()
        self.start_time = time.time()

    def stop(self, message = None):
        if self.viewer is not None:
            elapsed = time.time() - self.start_time
            memdiff = memusage() - self.start_memusage
            if message:
                status = '%s %s' % (self.title, message)
            else:
                status = '%s took %s and %s' % (self.title, time2str(elapsed), bytes2str(memdiff))
            self.viewer.status = status
            self.viewer = None

    __del__ = stop

    def update (self, message):
        if self.viewer is not None:
            self.viewer.status = '%s..%s' % (self.title, message)
