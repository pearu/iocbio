import os
import cPickle as pickle

class Cacher:
    """

    Attributes
    ----------
    cache : list
      A list of cache indices and parameters.

    """

    def __init__(self, cachefile):
        for ext in ['', '.pkl', '.pickle']:
            if os.path.isfile (cachefile + ext):
                cachefile = cachefile + ext
        cachedir = os.path.splitext (cachefile)[0]+'_cache'
        if not os.path.exists(cachedir):
            os.makedirs (cachedir)
        self.cachefile = cachefile
        self.cachedir = cachedir
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.isfile (self.cachefile):
            f = open(self.cachefile, 'rb')
            cache = pickle.load (f)
            f.close()        
        else:
            cache = []
        return cache
            
    def save_cache (self, cache):
        f = open(self.cachefile, 'wb')
        pickle.dump(cache, f)
        f.close()

    def get(self, **params):
        index = None
        for cacheindex, cacheparams in self.cache:
            if cacheparams==params:
                index = cacheindex
        if index is None:
            result = self.compute(**params)
            if self.cache:
                index = self.cache[-1][0]+1
            else:
                index = 0
            self.cache.append((index, params))
            filename = os.path.join(self.cachedir, str (index))
            print 'Saving result to', filename
            f = open (filename, 'wb')
            pickle.dump(result, f)
            f.close()
            self.save_cache(self.cache)
        else:
            filename = os.path.join(self.cachedir, str (index))
            print 'Loading result from', filename
            f = open (filename, 'rb')
            result = pickle.load(f)
            f.close ()
        return result

    def compute(self, **params):
        """ Compute model with parameters and return result to be cached.
        """
        return params

    def show(self):
        print 'Results available:'
        for index, params in self.cache:
            print params

