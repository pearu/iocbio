"""Provides find_clusters function.

The used 3D algorithm is a generalization of the 2D algorithm:
  http://stackoverflow.com/questions/411837/finding-clusters-of-mass-in-a-matrix-bitmap/413828

Module content
--------------
"""
# Author: Pearu Peterson
# Created: May 2009

__all__ = ['find_clusters']

import numpy
import scipy.stats

from collections import defaultdict

from ..utils import ProgressBar

class DisjointSets(object):

    def __init__(self):
        self.item_map = defaultdict(DisjointNode)

    def add(self,item):
        """Add item to the forest."""
        # It's gets initialized to a new node when
        # trying to access a non-existant item.
        return self.item_map[item]

    def __contains__(self,item):
        return (item in self.item_map)

    def __getitem__(self,item):
        if item not in self:
            raise KeyError
        return self.item_map[item]

    def __delitem__(self,item):
        del self.item_map[item]

    def __iter__(self):
        # sort all items into real sets
        all_sets = defaultdict(set)
        for item,node in self.item_map.iteritems():
            all_sets[node.find_set()].add(item)
        return all_sets.itervalues()

    def __len__(self):
        return len(self.item_map)

class DisjointNode(object):

    def __init__(self,parent=None,rank=0):
        if parent is None:
            self.parent = self
        else:
            self.parent = parent
        self.rank = rank

    def union(self, other):
        """Join two sets."""
        node1 = self.find_set()
        node2 = other.find_set()
        # union by rank
        if node1.rank > node2.rank:
            node2.parent = node1
        else:
            node1.parent = node2
            if node1.rank == node2.rank:
                node2.rank += 1
        return node1

    def find_set(self):
        """Finds the root node of this set."""
        node = self
        while node is not node.parent:
            node = node.parent
        # path compression
        root, node = node, self
        while node is not node.parent:
            node, node.parent = node.parent, root
        return root

def compute_incr_list(r2 = 5):
    incr_list = []
    for di in [-2,-1,0,1,2]:
        for dj in [-2,-1,0,1,2]:
            for dk in [-2,-1,0,1,2]:
                dijk = (di,dj,dk)
                s = di*di + dj*dj + dk*dk
                if s and s < r2:
                    incr_list.append(dijk)
    return incr_list

def find_clusters(data, background_level=None, voxel_sizes = None):
    """Find clusters in data.

    Parameters
    ----------
    data : :numpy:`ndarray`

    background_level : {None, float}
    
      Specifies maximum background level for finding clusters. By
      default, the level will be estimated using the first and the
      last image in the stack.

    voxel_sizes : {None, tuple}

    Returns
    -------
    filtered_data : list
      List items are (coordinates, values).
    """
    print '  Finding clusters from image stack (shape=%s)..' % (data.shape,)
    mn, mx = data.min(), data.max()
    print '    Image stack minimum/maximum: %s/%s' % (mn, mx)

    N = len (data.shape)
    if voxel_sizes is None:
        voxel_sizes = (1,)*N

    background_mean = 0
    if background_level is not None:
        pass
    else:
        background_data = data[((0,-1),)] # bottom and top
        background_mean = background_data.mean()
        background_std = background_data.std()
        background_var = background_data.var()
        background_min = background_data.min()
        background_max = background_data.max()
        background_offset = background_mean-background_var
        n = 10
        bins = [int(background_min + float(i * (background_max-background_min))/(n-1)) for i in range(n)]
        bins = sorted(set(bins))
        print '    Background mean:', background_mean
        print '    Background std:', background_std
        print '    Background variance:', background_var
        print '    Background median:', numpy.median (background_data)
        print '    Background minimum/maximum: %s/%s' % (background_min, background_max)
        print '    Estimated offset: %s' % (background_offset)

        hist, bins_edges = numpy.histogram(background_data, bins = bins)
        image_distribution = hist/float(hist.sum())

        if background_mean==0:
            background_level = 0
        else:
            p = 1-background_var/background_mean
            if p <= 1e-3: # try Poisson noise
                noise_distribution = scipy.stats.poisson(background_mean).pmf(bins[:-1])        
                relative_diff = image_distribution/noise_distribution
                indices = numpy.where(abs(numpy.log10(relative_diff)) > 1)[0]
                if indices.any():
                    background_level = bins_edges[1+indices[0]]
                else:
                    background_level = bins_edges[-1]
                if background_level > background_mean:
                    print '    Background noise distribution: Poisson(%s)' % (background_mean)
                else:
                    print '    Background noise distribution: indetermined (use --cluster-background-level=..)'
                    background_level = background_max*0.98
            else:
                # Binomial distribution
                n = max(1,background_mean/p)
                print '    Background noise distribution: Binomial(%s, %s)' % (n, p)
                noise_distribution = scipy.stats.binom(n, p).pmf(bins[:-1])
                relative_diff = image_distribution/noise_distribution
                indices = numpy.where(abs(numpy.log10(relative_diff)) > 1)[0]
                if indices.any():
                    background_level = bins_edges[1+indices[0]]
                else:
                    background_level = bins_edges[-1]
    print '    Background field <=', background_level
    signal_indices = numpy.where(data>background_level)
    sbr = 100*signal_indices[0].size/float(data.size)
    print '    Signal/background ratio: %s/%s (=%.5f%%)' % (signal_indices[0].size, data.size, sbr)
    if sbr > 20:
        raise 'Too large ratio, use --cluster-background-level=..'
    
    incr_list = compute_incr_list ()
                    
    n = len(signal_indices[0])
    print '    Finding clusters from %s points:' % (n)

    bar = ProgressBar(0,n-1, prefix='    ')
    disj = DisjointSets()
    for m, ijk in enumerate(zip(*signal_indices)):
        bar.updateComment (' %s' % m)
        bar(m)
        node = disj.add(ijk)
        for dijk in incr_list:
            ijk1 = tuple([ijk[i]+dijk[i] for i in range(3)])
            if ijk1 in disj:
                node.union(disj[ijk1])
    bar(m)
    print

    sorted_data = sorted([(len(s), s) for s in disj], reverse=True)
    
    print '    Found %s cluster candidates,' % (len(sorted_data)),

    filtered_data = []
    param = []
    param_ab = []
    for i, (sz, s) in enumerate(sorted_data):
        if background_level>0:
            mean_sz = (mean_sz * (i-1) + sz)/float(i) if i else sz
            if mean_sz / sz > 100 or sz <= 7:
                break
        s = list(s)
        values = data[tuple(zip(*s))] - background_mean
        if abs((values.min() - values.max())/values.mean()) > 0.1:
            coordinates = numpy.array(s)
            filtered_data.append((coordinates, values))

    print 'out of which %s are large and hump-like enough.' % (len (filtered_data))
    if 0:
        print '  Eccentricity of clusters:', map(lambda p:'%.1f/%.1f'%p ,param_ab)
        l = find_outliers(param_ab, zoffset=3, data_contains_seq=True)
        if l:
            print '  Indices of outliers:',l
            for i in sorted(l, reverse=True):
                del filtered_data[i]
            print '  Final cluster count after removing outliers: %s' % (len (filtered_data))
        else:
            print '  No outliers detected'
    if len(filtered_data)==1:
        print '    Warning: only one cluster found. Use --cluster-background-level=... with larger value than %s' % (background_level)

    return filtered_data

def find_outliers(data, zoffset=3, data_contains_seq = False):
    """
    Return indices of the most significant outliers in data.
    """
    if data_contains_seq:
        s = set (range (len (data[0])))
        for d in zip(*data):
            s = s.intersection(set(find_outliers(d, zoffset=zoffset)))
        return sorted(s)

    data = list(data)
    if len (data)<=2:
        return []
    zmax = 0
    rmax = 0
    index = None
    lst = []
    for i in range(len(data)):
        l = data[:i] + data[i+1:]
        m = numpy.mean(l)
        s = numpy.std(l)
        z = abs(data[i] - m)/s
        if z > zoffset and z > zmax:
            zmax = z
            index = i
    if index is not None:
        outliers = [index]
        for i in find_outliers(data[:index] + data[index+1:], zoffset=zoffset):
            outliers.append(i if i<index else i+1)
        return outliers
    return []
