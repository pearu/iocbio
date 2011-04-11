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

    def compare(self, params1, params2, skip_list):
        keys1 = params1.keys ()
        keys2 = params2.keys ()
        for s in skip_list:
            if s in keys1: keys1.remove (s)
            if s in keys2: keys2.remove (s)
        if keys1==keys2:
            for k in keys1:
                if params1[k] != params2[k]:
                    return False
            return True
        else:
            return False

    def get(self, **params):
        index = None
        skip_list =  params.get('skip',[])
        for ind, (cacheindex, cacheparams) in enumerate (self.cache):
            if self.compare(cacheparams, params, skip_list):
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
            if os.path.isfile (filename):
                print 'Loading result from', filename
                f = open (filename, 'rb')
                result = pickle.load(f)
                f.close ()
                if skip_list:
                    params['old_result'] = result
                    result = self.compute(**params)
                    self.cache[ind] = (index, params)
                    print 'Saving result to', filename
                    f = open (filename, 'wb')
                    pickle.dump(result, f)
                    f.close()
                    self.save_cache(self.cache)
            else:
                result = self.compute(**params)
                self.cache[ind] = (index, params)
                print 'Saving result to', filename
                f = open (filename, 'wb')
                pickle.dump(result, f)
                f.close()
                self.save_cache(self.cache)
        return result

    def compute(self, **params):
        """ Compute model with parameters and return result to be cached.
        """
        return params

    def show(self):
        print 'Results available:'
        for index, params in self.cache:
            print params

