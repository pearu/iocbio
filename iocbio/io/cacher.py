""" Provides Cacher that implements computational model with caching results.
"""
# Author: Pearu Peterson
# Created: April 2011

import os
import cPickle as pickle

class Cacher:
    """ Provides a computational model with caching results.

    Example usage
    -------------
    >>> class System(Cacher):
            def compute (self, a=1, t=0):
                if self.previous_results is not None:
                    last_t = self.previous_parameters['t']
                    last_value = self.previous_results
                else:
                    last_t = 0
                    last_value = 1
                return last_value + a*(t-last_t)

    >>> system = System('cacher_test', dynamic_parameters=['t'])
    >>> print system.get(a=2,t=3)
    7
    >>> system.show_cache()
    """

    def __init__(self, cachedir, dynamic_parameters = [], verbose=False):
        """
        Parameters
        ----------
        cachedir : str
          Specify the model cache name. It is also the directory
          name where the results will be cached.
        dynamic_parameters : list
          Specify a list of parameters that are allowed to change
          within the same model computations. For example, the
          integration time is a dynamic parameter. The users
          `compute` method can use `previous_results` attribute
          to setup continuiong integration.
        verbose : bool
          When True then report cache operations.
        """
        self.__verbose = verbose
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)        
        cachefile = os.path.join(cachedir, 'index.pkl')
        self.__cachefile = cachefile
        self.__cachedir = cachedir
        self.__cache = {}
        self.__load_cache()
        self.dynamic_parameters = dynamic_parameters

    def __load_cache(self):
        if os.path.isfile (self.__cachefile):
            f = open(self.__cachefile, 'rb')
            self.__cache = pickle.load(f)
            f.close()        
            
    def __save_cache (self):
        f = open(self.__cachefile, 'wb')
        pickle.dump(self.__cache, f)
        f.close()

    def __compare_parameters(self, params1, params2, discard_dynamic=True):
        keys1 = sorted(params1.keys ())
        keys2 = sorted(params2.keys ())
        if discard_dynamic:
            for s in self.dynamic_parameters:
                if s in keys1: keys1.remove (s)
                if s in keys2: keys2.remove (s)
        if keys1==keys2:
            for k in keys1:
                if params1[k] != params2[k]:
                    return False
            return True
        else:
            return False

    def __save_results(self, index, parameters, results):
        if index is None:
            index = max (self.__cache.keys ()+[0])+1
        filename = os.path.join(self.__cachedir, '%s.pkl' % (index))
        if self.__verbose:
            print 'Saving result to', filename
        f = open (filename, 'wb')
        pickle.dump(results, f)
        f.close()
        self.__cache[index] = parameters
        self.__save_cache()

    def __load_results (self, index, parameters):
        assert index is not None,`index`
        filename = os.path.join(self.__cachedir, '%s.pkl' % (index))

        need_compute = True
        if os.path.isfile(filename):
            if self.__verbose:
                print 'Loading results from', filename
            f = open (filename, 'rb')
            results = pickle.load(f)
            f.close ()
            need_compute = not self.__compare_parameters(self.__cache[index], parameters, discard_dynamic=False)
        else:
            if self.__verbose:
                print 'The results file',filename,'has disappeared'
            results = None
            del self.__cache[index]
        return results, need_compute

    def __find_results_index (self, parameters):
        for index, cache_parameters in self.__cache.iteritems ():
            if self.__compare_parameters (cache_parameters, parameters):
                return index

    def get(self, **parameters):
        """ Return results with parameters.

        When results do not exist, then user defined `compute` method is
        called that returned values will be cached and returned.

        Parameters
        ----------
        parameters : dict
          Specify model parameters.

        Returns
        -------
        """
        index = self.__find_results_index (parameters)
        self.previous_results = None
        self.previous_parameters = None
        if index is None:
            if self.__verbose:
                print 'Computing results'
            results = self.compute(**parameters)
            self.__save_results(index, parameters, results)
        else:
            results, need_compute = self.__load_results(index, parameters)
            if need_compute:
                self.previous_parameters = self.__cache.get(index)
                self.previous_results = results
                print 'Re-computing results'
                results = self.compute(**parameters)
                self.__save_results(index, parameters, results)
        return results

    def compute(self, **parameters):
        """ Compute model with parameters and return the results to be cached.

        This method should be redefined by the user.

        Cached computations can be continued using
        `previous_parameters` and `previous_results` attributes. Their
        values are None by default.

        Parameters
        ----------
        parameters : dict
          Specify model parameters.
        """
        raise NotImplementedError ('compute (%s)' % (parameters))

    def show_cache(self):
        """ Print the content of cache to stdout.
        """
        print('Available results in %r:' % (self.__cachedir))
        for index, parameters in self.__cache.iteritems():
            filename = os.path.join(self.__cachedir, '%s.pkl' % (index))
            if os.path.isfile(filename):
                params = ['%s=%r' % (k,v) for k,v in parameters.iteritems()]
                print('%s: %s' % (index, ', '.join(sorted(params))))

if __name__=='__main__':

    class System(Cacher):
        def compute (self, a=1, t=0):
            if self.previous_results is not None:
                last_t = self.previous_parameters['t']
                last_value = self.previous_results
            else:
                last_t = 0
                last_value = 1
            return last_value + a*(t-last_t)

    system = System('cacher_test', dynamic_parameters=['t'], verbose=True)
    print system.get(a=2,t=1)
    system.show_cache()
