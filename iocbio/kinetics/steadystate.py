""" Provides methods to import chemical kinetic models.

This module provides tools for reading chemical kinetic models.
It currently supports a string based format in addition to SBML.

obj2num currently depends on the routine from_float inside mpmath.libmp.

Example
-------

"""
from __future__ import division

__all__ = ['load_stoic_from_sbml',
           'load_stoic_from_text',
           'obj2num',
           'SteadyFluxAnalyzer',
           ]

import re
import os
import time
import random
import cPickle as pickle

from lxml import etree
from collections import defaultdict

from sympycore.matrices import Matrix

class SteadyFluxAnalyzer(object):
    """ Base class for analyzing the steady state of a metabolic network.


    Attributes
    ----------
    species : list
      A list of species names or row names.
    reaction : list
      A list of reaction names or column names.
    stoichiometry : Matrix
      A stoichiometric matrix.

    """

    def __init__(self, source,
                 discard_boundary_species=False,
                 add_boundary_fluxes=False,
                 split_bidirectional_fluxes = False,
                 growth_factor = 1,
                 growth_shuffle = True,
                 ):
        """
        Parameters
        ----------
        source : str
          A file name to sbml file or a string with reaction definitions.
          For internal processing source can also be tuple.

        discard_boundary_species : bool

          When True then discard species that are reactants or
          products of the full network. The corresponding
          stoichiometry system will be open. For example, in a
          reaction ``A -> B -> C`` the species A and C are reactant
          and product of the system and after discarding A and C, the
          system will be open: ``-> B ->`` .  In the case of a
          reaction `` -> A -> B + C`` the system will be made open by
          adding new reactions `` B-> `` and `` C -> ``.

        add_boundary_fluxes : bool

          When True then add boundary fluxes to boundary
          species.  The corresponding stoichiometry system will be
          open.  For example, in a reaction ``A -> B -> C`` the
          species A and C are reactant and product of the system and
          after introducing boundary fluxes, the system will be open:
          ``-> A -> B -> C ->``.  New flux names start with prefix
          ``BR_``.

        split_bidirectional_fluxes : bool
          When True the bidirectional fluxes are split into two unidirectional fluxes.
          For example, the system ``A<=>B`` is treated as ``A=>B and B=>A``.

        growth_factor : int
          Increase the size of network by growth_factor.
        growth_shiffle : bool
          When True then shuffle the rows and columns of increased network.

        See also
        --------
        load_stoic_from_sbml, load_stoic_from_text, discard_boundary_species, add_boundary_fluxes
        """
        source_type = 'tuple'
        if isinstance (source, str):
            if os.path.isfile (source) or (source.count ('\n')==0 and '=' not in source):
                stoic_dict, species, reactions, species_info, reactions_info = \
                    load_stoic_from_sbml(source, split_bidirectional_fluxes=split_bidirectional_fluxes)
            else:
                source_type = 'string'
                stoic_dict, species, reactions, species_info, reactions_info = \
                    load_stoic_from_text(source, split_bidirectional_fluxes=split_bidirectional_fluxes)
        elif isinstance (source, tuple):
            stoic_dict, species, reactions, species_info, reactions_info = source
        else:
            raise TypeError ('expected source to be str or tuple but got %r' % (type(source)))

        discarded_species = []
        if discard_boundary_species:
            stoic_dict, species, discarded_species, reactions, species_info, reactions_info = \
                self.discard_boundary_species(stoic_dict, species, reactions, species_info, reactions_info)
        if add_boundary_fluxes:
            stoic_dict, species, reactions, species_info, reactions_info = \
                self.add_boundary_fluxes(stoic_dict, species, reactions, species_info, reactions_info)
        if growth_factor>1:
            stoic_dict, species, reactions, species_info, reactions_info = \
                self.apply_growth(stoic_dict, species, reactions, species_info, \
                                  reactions_info, growth_factor, growth_shuffle)

        self.discarded_species = discarded_species
        self.source_type = source_type
        self.source = source
        self.options = (discard_boundary_species, add_boundary_fluxes, \
                        split_bidirectional_fluxes, growth_factor, growth_shuffle)
        self.source_data = stoic_dict, species, reactions, dict(species_info), dict(reactions_info)

        self._stoichiometry = None
        self.compute_kernel_GJE_data = None
        self.compute_kernel_SVD_data = None

    @property
    def system_string(self):
        if self.source_type == 'string':
            return self.source
        elif self.source_type == 'tuple':
            out = str()
            for r in self.reactions:
                out += '\n{0}'.format(r)
            return out
    @property
    def species(self): return self.source_data[1]
    @property
    def reactions(self): return self.source_data[2]
    @property
    def species_info(self): return self.source_data[3]
    @property
    def reactions_info(self): return self.source_data[4]
    @property
    def stoichiometry(self):
        if self._stoichiometry is None:
            self._stoichiometry = Matrix(len (self.species), len (self.reactions), self.source_data[0])
        return self._stoichiometry

    def _get_pickle_file_name(self, file_name):
        for ext in ['', '.pkl', '.pickle']:
            if os.path.isfile(file_name+ext):
                file_name = file_name + ext
                break
        if os.path.splitext(file_name)[1]=='':
            file_name = file_name + '.pkl'
        return file_name

    def save(self, file_name):
        """ Save instance data to file_name.
        """
        file_name = self._get_pickle_file_name(file_name)
        dirname = os.path.dirname (file_name)
        if os.path.isfile(file_name):

            f = open(file_name, 'rb')
            try:
                data = pickle.load(f)
            except Exception, msg:
                print 'Failed to load %r: %s' % (file_name, msg)
                data = []
            f.close()
        else:
            data = []
        if not isinstance(data, list):
            data = []

        key = (self.source, self.options)
        value = {}
        for a in dir(self):
            if a.endswith('_data') or a.endswith('_elapsed'):
                value[a] = getattr(self, a)
        
        data_file_name = None
        for key0, data_file_name0 in data:
            if key0==key:
                data_file_name = os.path.join(dirname, os.path.basename(data_file_name0))
                break
        if data_file_name is None:
            base, ext = os.path.splitext(file_name)
            data_file_name = '%s_%s%s' % (os.path.basename(base), len(data), ext)
            data.append((key, data_file_name))

            f = open(file_name, 'wb')
            pickle.dump(data, f)
            f.close()

        f = open(data_file_name, 'wb')
        pickle.dump(value, f)
        f.close()
        print 'Succesfully wrote data to', data_file_name
        
    def load(self, file_name):
        """ Load instance data from file_name.
        """
        file_name = self._get_pickle_file_name(file_name)
        dirname = os.path.dirname (file_name)
        if os.path.isfile(file_name):
            f = open(file_name, 'rb')
            try:
                data = pickle.load(f)
            except Exception, msg:
                print 'Failed to load %r: %s' % (file_name, msg)
                data = []
            f.close()
        else:
            return
        data_file_name = None
        key = (self.source, self.options)
        for key0, data_file_name0 in data:
            if len(key)==len(key0)+1:
                # for backward compatibility
                key0 = key0[0], key0[1][:2] + (False,) + key0[1][2:]
                
            if key0==key:
                data_file_name = os.path.join(dirname, os.path.basename(data_file_name0))
                break
        if data_file_name is None:
            return
        f = open(data_file_name, 'rb')
        value = pickle.load(f)
        f.close()
        print 'Succesfully loaded data from', data_file_name
        self._stoichiometry = None
        for a,v in value.iteritems():
            setattr(self, a, v)
        return True

    def __repr__(self):
        return '%s((%r, %r, %r, %r, %r))' % (self.__class__.__name__,
                                             self.stoichiometry, self.species, self.reactions,
                                             self.species_info, self.reactions_info)

    def __str__(self, max_nrows=20, max_ncols=20):
        mat = self.label_matrix(self.stoichiometry, self.species, self.reactions)
        return mat.__str__(max_nrows=max_nrows, max_ncols=max_ncols)

    def label_matrix(self, matrix, row_labels, col_labels):
        """ Return a matrix with labeled rows and columns.
        """
        m, n = matrix.shape
        new_matrix = Matrix(m+1, n+1)
        new_matrix[1:, 1:] =  matrix
        for i, r in enumerate (col_labels):
            new_matrix[0,i+1] = r
        for i, r in enumerate (row_labels):
            new_matrix[i+1,0] = r
        new_matrix[0,0] = ' '
        return new_matrix

    @property
    def shape(self):
        return self.stoichiometry.shape

    @property
    def sparsity(self):
        m,n = self.stoichiometry.shape
        return 1-len (self.stoichiometry.data)/(m*n)

    @property
    def rank (self):
        self.compute_kernel_GJE(use_cache=True)        
        return len(self.dependent_variables)

    def get_splitted_reactions (self):
        """ Split reactions to a list of dependent and independent
        reactions.

        Returns
        -------
        leading_cols_canditates, trailing_cols_canditates : list
          Lists of integers.
        """
        reactions = self.reactions
        independent_cols = []
        for reaction_id in reactions:
            if reaction_id.startswith('BR_'):
                independent_cols.append(reactions.index(reaction_id))
        leading_cols_canditates = sorted(set(range(len(reactions))).difference(independent_cols))
        trailing_cols_canditates = independent_cols
        return leading_cols_canditates, trailing_cols_canditates

    @classmethod
    def apply_growth(cls, stoic_dict, species, reactions, species_info, \
                     reactions_info, growth_factor=1, growth_shuffle=True):
        """ Increase the size of network by a growth_factor.

        Useful for testing.
        """
        if growth_factor==1:
            return stoic_dict, species, reactions, species_info, reactions_info
        m0, n0 = len(species), len(reactions)
        m1, n1 = m0*growth_factor, n0*growth_factor
        species1 = []
        reactions1 = []
        for i in range (growth_factor):
            for s in species:
                species1.append('%s_%s' % (s, i))
            for s in reactions:
                reactions1.append('%s_%s' % (s, i))
        rindices = range(m1)
        indices = range(n1)
        if growth_shuffle:
            random.shuffle (rindices)
            random.shuffle (indices)
            species1 = [species1[i] for i in rindices]
            reactions1 = [reactions1[i] for i in indices]

        stoic_dict1 = {}
        for k in range (growth_factor):
            for (i,j), v in stoic_dict.iteritems():
                stoic_dict1[rindices[i+ k*m0], indices[j+ k*n0]] = v 
        return stoic_dict1, species1, reactions1, {}, {}

    def compute_kernel_GJE(self,
                           leading_row_selection='sparsest first',
                           leading_column_selection='sparsest first',
                           dependent_candidates = None,
                           force = False,
                           use_cache = None
                           ):
        """ Compute the kernel of stoichiometric matrix via GJE routine.

        Parameters
        ----------
        leading_row_selection, leading_column_selection : str
          Specify parameters to get_gauss_jordan_elimination_operations method.

        dependent_candidates : list

          Specify the list of dependent fluxes. Note that this is
          considered only as a suggestion. The actual list of
          dependent fluxes may be different.

        See also
        --------
        MatrixBase.get_gauss_jordan_elimination_operations
        """
        if use_cache and self.compute_kernel_GJE_data is not None:
            options = self.compute_kernel_GJE_parameters_data
        else:
            options = leading_row_selection, leading_column_selection, dependent_candidates
        if self.compute_kernel_GJE_data is not None and self.compute_kernel_GJE_parameters_data != options:
            force = True
        if self.compute_kernel_GJE_data is None or force:
            if dependent_candidates is None:
                lcc, tcc = self.get_splitted_reactions()
            else:
                lcc = [self.reactions.index(r) for r in dependent_candidates]
                tcc = [self.reactions.index(r) for r in self.reactions if r not in dependent_candidates]
            start = time.time ()
            
            gje_fn = self.stoichiometry.get_gauss_jordan_elimination_operations
            result = gje_fn(leading_cols=lcc,
                            trailing_cols = tcc,
                            leading_row_selection=leading_row_selection,
                            leading_column_selection=leading_column_selection)
            
            end = time.time()
            self.compute_kernel_GJE_data = result
            self.compute_kernel_GJE_elapsed = end - start

            independent_variables = []
            dependent_variables = []
            for k, variable in enumerate(self.reactions):
                if k in result[3]:
                    dependent_variables.append(variable)
                else:
                    independent_variables.append(variable)
            self.variables_data = dependent_variables, independent_variables
            self.compute_kernel_GJE_parameters_data = options

            
    @property
    def dependent_variables(self):
        return self.variables_data[0]

    @property
    def independent_variables(self):
        return self.variables_data[1]


    def get_sorted_reactions(self):
        from sympycore.matrices.linalg import  get_rc_map
        self.compute_kernel_GJE(use_cache=True)
        start = time.time ()
        gj, row_operations, leading_rows, leading_cols, zero_rows = self.compute_kernel_GJE_data
        l = []
        rows = get_rc_map(gj.data)
        for i0,j0 in zip (leading_rows, leading_cols):
            row = rows[i0]
            row.remove (j0)
            if row:
                l.append ((len(row),-min(row), self.reactions[j0]))
            else:
                l.append ((len(row),0, self.reactions[j0]))
        reactions = [s for i,i1,s in sorted(l, reverse=True)]
        reactions += [s for s in self.reactions if s not in reactions]
        end = time.time ()
        self.get_sorted_reactions_elapsed = end-start
        return reactions

    def get_kernel_GJE(self, reactions=None):
        """ Return the kernel K from GJE routine.
        
        Notes
        -----
        The steady state solution is given by ``reactions = K * indep_variables``
        where ``indep_variables = reactions[-K.shape[1]:]``

        Parameters
        ----------
        reactions : list
          When specified then the reaction list defines the order of flux rows.
          Otherwise the rows are order by the sparsity.

        Returns
        -------
        fluxes, indep_fluxes, kernel : list, list, Matrix
        """
        from sympycore.matrices.linalg import  get_rc_map
        self.compute_kernel_GJE(use_cache=True)
        gj, row_operations, leading_rows, leading_cols, zero_rows = self.compute_kernel_GJE_data
        if reactions is None:
            reactions = self.get_sorted_reactions()
        start = time.time ()
        kernel = Matrix(len(reactions), len(reactions)-len(leading_cols))
        indep_vars = [r for r in reactions if r in self.independent_variables]
        n = len (indep_vars)
        rows = get_rc_map(gj.data)
        
        def indep_index (r):
            try: return indep_vars.index (r)
            except ValueError: pass
        indep_indices = []
        indices = []
        for r in self.reactions:
            indices.append(reactions.index(r))
            indep_indices.append(indep_index (r))

        for i0,j0 in zip(leading_rows, leading_cols):
            i = indices[j0]
            for j1 in rows[i0]:
                if j1!=j0:
                    j = indep_indices[j1]
                    kernel[i,j] = -gj.data[(i0,j1)]

        for j0 in range (len (self.reactions)):
            if j0 not in leading_cols:
                i = indices[j0]
                j = indep_indices[j0]
                kernel[i,j] = 1
        end = time.time()
        self.get_kernel_GJE_elapsed = end-start
        return reactions, indep_vars, kernel

    def get_relation_GJE(self, reactions=None):
        """ Return relation matrix R from GJE routine.

        Notes
        -----
        The steady state solution is given by ``dep_fluxes = R * indep_fluxes``

        Parameters
        ----------
        reactions : list
          When specified then the reaction list defines the order of rows.
          Otherwise the rows are order by the sparsity.

        Returns
        -------
        dep_fluxes, indep_fluxes, rmatrix : list, list, Matrix

        See also
        --------
        get_relation_SVD
        """
        reactions, indep_reactions, kernel = self.get_kernel_GJE(reactions=reactions)
        start = time.time()
        rank = kernel.shape[0] - kernel.shape[1]
        dep_reactions = [r for r in reactions if r not in indep_reactions]
        r = kernel[:rank]
        end = time.time()
        self.get_relation_GJE_elapsed = end-start
        return dep_reactions, indep_reactions, r

    def get_dense_stoichiometry(self, species, reactions):
        import numpy
        r = numpy.zeros (self.stoichiometry.shape, dtype=float)
        data = self.stoichiometry[:].data # todo: optimize when not transposed
        for (i,j), v in data.iteritems ():
            i1 = species.index (self.species[i])
            j1 = reactions.index (self.reactions[j])
            r[i1,j1] = v
        return r

    def compute_kernel_SVD(self):
        """Compute the kernel of stoichiometric matrix via SVD routine.
        """
        import numpy
        if self.compute_kernel_SVD_data is None:
            array = self.get_dense_stoichiometry (self.species, self.reactions)
            start = time.time ()
            U, s, V = numpy.linalg.svd(array)
            end = time.time()
            self.compute_kernel_SVD_data = U, s, V
            self.compute_kernel_SVD_elapsed = end - start

    def get_kernel_SVD(self, reactions=None):
        """Return the kernel K from SVD routine.

        Notes
        -----
        The steady state solution is given by ``reactions = K * parameters``.
        
        Parameters
        ----------
        reactions : list
          When specified then the reaction list defines the order of rows.
          Otherwise the rows are order by the sparsity.

        Returns
        -------
        reactions, kernel : list, Matrix
        """
        import numpy
        self.compute_kernel_SVD()
        if reactions is None:
            reactions = self.get_sorted_reactions()
        start = time.time ()
        U, s, V = self.compute_kernel_SVD_data
        Vker = V[self.rank:].T
        Vker1 = numpy.zeros (Vker.shape, dtype=float)
        for i, r in enumerate(reactions):
            Vker1[i] = Vker[self.reactions.index(r)]
        end = time.time()
        self.get_kernel_SVD_elapsed = end-start
        return reactions, Vker1

    def get_relation_SVD(self, reactions=None):
        """ Return relation matrix R from SVD routine.

        Notes
        -----
        The steady state solution is given by ``dep_fluxes = R * indep_fluxes``

        Parameters
        ----------
        reactions : list
          When specified then the reaction list defines the order of rows.
          Otherwise the rows are order by the sparsity.

        Returns
        -------
        dep_fluxes, indep_fluxes, rmatrix : list, list, numpy.ndarray
        
        See also
        --------
        get_relation_GJE
        """
        import numpy
        reactions, kernel = self.get_kernel_SVD(reactions=reactions)
        start = time.time ()
        rank = kernel.shape[0]-kernel.shape[1]
        dep_kernel = kernel[:rank]
        indep_kernel = kernel[rank:]
        dep_reactions = reactions[:rank]
        indep_reactions = reactions[rank:]
        r = numpy.dot(dep_kernel, numpy.linalg.inv(indep_kernel))
        end = time.time ()
        self.get_relation_SVD_elapsed = end-start
        return dep_reactions, indep_reactions, r

    def get_relation_SVD_error(self, reactions=None):
        """ Compute maximal relative flux error of the SVD relation routine.

        Notes
        -----
        See the paper for definition.

        Returns
        -------
        rerr : float
        """
        import numpy
        dep_reactions_GJE, indep_reactions_GJE, relation_GJE = self.get_relation_GJE (reactions=reactions)
        dep_reactions_SVD, indep_reactions_SVD, relation_SVD = self.get_relation_SVD (reactions=reactions)
        assert dep_reactions_GJE==dep_reactions_GJE
        assert indep_reactions_GJE==indep_reactions_GJE
        dense_relation_GJE = numpy.zeros(relation_GJE.shape, dtype=float)
        for (i,j),v in relation_GJE.data.iteritems ():
            dense_relation_GJE[i,j]=v
        abserrdata = abs(relation_SVD - dense_relation_GJE).sum (axis=1)
        absdata = numpy.maximum(numpy.maximum(abs(relation_SVD),
                                              abs(dense_relation_GJE)),
                                numpy.ones(relation_SVD.shape)).sum (axis=1)
    
        return max(abserrdata/absdata)

    @classmethod
    def add_boundary_fluxes(cls, matrix, species, reactions, species_info, reactions_info):
        """ Make network open by adding new columns that correspond to
        the boundary fluxes.
        
        Boundary fluxes are defines as fluxes from boundary species
        that enter or leave the network. The names of boundary fluxes
        start with prefix ``BR_``.
        """
        for specie_index in range(len(species)):
            reaction_indices = [j for i,j in matrix if i==specie_index]
            if len (reaction_indices)>1 and\
                not (len(reaction_indices)==2 and len(set(reaction_indices))==1): # to catch polymerization reactions
                continue
            reaction_index = reaction_indices[0]
            stoichiometry = matrix[specie_index, reaction_index]
            specie_id = species[specie_index]
            new_reaction_id = 'BR_%s' % (specie_id)
            new_reaction_index = len(reactions)
            reactions.append(new_reaction_id)
            new_stoichiometry = -1 if stoichiometry>0 else 1
            matrix[specie_index, new_reaction_index] = new_stoichiometry
            if new_stoichiometry>0:
                reactions_info[new_reaction_id]['products'].append(specie_id)
            else:
                reactions_info[new_reaction_id]['reactants'].append(specie_id)
        return matrix, species, reactions, species_info, reactions_info

    @classmethod
    def discard_boundary_species(cls, matrix, species, reactions, species_info, reactions_info):
        """ Make the network open by discarding rows that correspond
        to boundary species.

        Boundary species are defined as the products or reactants of the network.
        """
        boundary_species_indicies = []
        extra_reactions = []
        for specie_index in range(len(species)):
            reaction_indices = [j for i,j in matrix if i==specie_index]
            if len(reaction_indices)>1:
                continue
            reaction_index = reaction_indices[0]
            reaction_id = reactions[reaction_index]
            stoichiometry = matrix[specie_index, reaction_index]
            boundary_species_indicies.append(specie_index)

        i = 0
        new_matrix = {}
        new_species = []
        boundary_species = []
        for specie_index in range(len(species)):
            if specie_index in boundary_species_indicies:
                boundary_species.append(species[specie_index])
                continue
            new_species.append(species[specie_index])
            for i0, reaction_index in matrix:
                if i0 != specie_index:
                    continue
                new_matrix[i, reaction_index] = matrix[specie_index, reaction_index]
            i += 1
        species = new_species
        matrix = new_matrix

        return matrix, species, boundary_species, reactions, species_info, reactions_info

    @property
    def sparsity_kernel_GJE(self):
        reactions, indep_reactions, kernel = self.get_kernel_GJE()
        return 1-len (kernel.data)/(kernel.shape[0]*kernel.shape[1])

    @property
    def sparsity_kernel_SVD(self):
        reactions, kernel = self.get_kernel_SVD()
        return 1-(abs(kernel)>1e-3).sum()/kernel.size

    @property
    def condition_number (self):
        self.compute_kernel_SVD()
        U, s, V = self.compute_kernel_SVD_data        
        return s[0]/s[self.rank-1]

    def matrix_plot(self, matrix, figure_name='matrix_plot.pdf'):
        import numpy
        from matplotlib import pylab
        def _blob(x,y,area,colour):
            hs = numpy.sqrt(area) / 2
            xcorners = numpy.array([x - hs, x + hs, x + hs, x - hs])
            ycorners = numpy.array([y - hs, y - hs, y + hs, y + hs])
            pylab.fill(xcorners, ycorners, colour, edgecolor=colour)
        reenable = False
        if pylab.isinteractive():
            pylab.ioff()
        pylab.clf()
        
        maxWeight = 2**numpy.ceil(numpy.log(numpy.max(numpy.abs(matrix)))/numpy.log(2))
        height, width = matrix.shape
        pylab.fill(numpy.array([0,width,width,0]),numpy.array([0,0,height,height]),'white')
        pylab.axis('off')
        pylab.axis('equal')
        for x in xrange(width):
            for y in xrange(height):
                _x = x+1
                _y = y+1
                w = matrix[y,x]
                if w > 0:
                    _blob(_x - 0.5, height - _y + 0.5, 0.2,'#0099CC')
                elif w < 0:
                    _blob(_x - 0.5, height - _y + 0.5, 0.2,'#660000')

        if reenable:
            pylab.ion()
        pylab.savefig(figure_name) 

def obj2num(s, precision=53):
    """ Converts an object into a number while avoiding numerical noise.

    If the object is a string, this string is evaluated.
    If the original object or result of this evaluation is an integer,
    this integer is returned. Otherwise, an mpf object is returned from
    the from_float routine within mpmath.libmp.  

    Parameters
    ----------
    s : object
      Usually a string that evaluates to a numerical object.
      
    precision : int
      Parameter passed to from_float to specify the precision required
      in the conversion.  Defaults to 53 as recommended by from_float.
    """
    from mpmath.libmp import from_float
    
    if isinstance(s, str):
        f = eval(s)
    else:
        f = s
    #################################################################    
    #return float(f) # will induce numerical errors in some routines.
    #################################################################
    i = int(f)
    if i==f:
        return i
    return from_float(f, prec=prec)


def load_stoic_from_sbml(file_name, split_bidirectional_fluxes=False):
    """ Return stoichiometry information of a network described in a SBML file.

    Parameters
    ----------
    file_name : str
      Path to SMBL file.

    split_bidirectional_fluxes : bool
      When True the bidirectional fluxes are split into two unidirectional fluxes.
      For example, the system ``A<=>B`` is treated as ``A=>B and B=>A``.

    Returns
    -------
    matrix : dict
      A stoichiometry matrix defined as mapping {(species, reaction): stoichiometry}.
    species : list
      A list of species names.
    reactions : list
      A list of reaction names.
    species_info : dict
    reactions_info : dict
    """
    from lxml import etree
    for ext in ['', '.xml', '.xml.gz', '.gz']:
        if os.path.isfile(file_name+ext):
            file_name = file_name+ext
            break
    tree = etree.parse(file_name)
    root = tree.getroot()
    assert root.tag.endswith ('sbml'), `root.tag`
    version = int(root.attrib['version'])
    level = int(root.attrib['level'])
    if level in [2,3]:
        default_stoichiometry = '1'
    else:
        default_stoichiometry = None
    compartments = {}
    species = []
    modifiers = []
    species_all = []
    reactions = []
    species_reactions = defaultdict (lambda:[])
    reactions_species = defaultdict (lambda:[])
    reactions_info = defaultdict(lambda:dict(modifiers=[],reactants=[],products=[],
                                            boundary_specie_stoichiometry={},annotation=[],
                                            compartments = set()))
    species_info = defaultdict(lambda:dict())
    matrix = {}
    for model in root:
        for item in model:
            if item.tag.endswith('listOfCompartments'):
                for compartment in item:
                    compartments[compartment.attrib['id']] = compartment.attrib
            elif item.tag.endswith('listOfSpecies'):
                for specie in item:
                    species_all.append(specie.attrib['id'])
                    species_info[specie.attrib['id']]['compartment'] = specie.attrib['compartment']
                    species_info[specie.attrib['id']]['name'] = specie.attrib.get('name', specie.attrib['id'])
            elif item.tag.endswith('listOfReactions'):
                for reaction in item:

                    reversible =eval(reaction.attrib.get('reversible', 'False').title())
                    reaction_id = reaction.attrib['id']
                    name = reaction.attrib.get('name', reaction_id)
                    assert reaction_id not in reactions,`reaction_id`
                    reactions.append(reaction_id)
                    reaction_index = len(reactions)-1
                    reactions_info[reaction_id]['name'] = name

                    if split_bidirectional_fluxes and reversible:
                        reaction_id2 = '%s_r' % (reaction_id)
                        assert reaction_id2 not in reactions,`reaction_id2`
                        reactions.append(reaction_id2)
                        reaction_index2 = len(reactions)-1
                        reactions_info[reaction_id2]['name'] = name+'_r'
                        reactions_info[reaction_id]['reversible'] = False
                        reactions_info[reaction_id2]['reversible'] = False
                    else:
                        reaction_id2 = reaction_index2 = None
                        reactions_info[reaction_id]['reversible'] = reversible

                    for part in reaction:
                        if part.tag.endswith ('listOfReactants'):
                            for reactant in part:
                                assert reactant.tag.endswith('speciesReference'), `reactant.tag`
                                specie_id = reactant.attrib['species']
                                stoichiometry = -obj2num(reactant.attrib.get('stoichiometry', default_stoichiometry))
                                reactions_info[reaction_id]['reactants'].append(specie_id)
                                try:
                                    specie_index = species.index(specie_id)
                                except ValueError:
                                    species.append(specie_id)
                                    specie_index = len(species)-1
                                assert stoichiometry,`stoichiometry`
                                matrix[specie_index, reaction_index] = stoichiometry
                                species_reactions[specie_index].append(reaction_index)
                                reactions_species[reaction_index].append(specie_index)                                
                                reactions_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                                if reaction_index2 is not None:
                                    reactions_info[reaction_id2]['reactants'].append(specie_id)
                                    matrix[specie_index, reaction_index2] = -stoichiometry
                                    species_reactions[specie_index].append(reaction_index2)
                                    reactions_species[reaction_index2].append(specie_index)                                
                                    reactions_info[reaction_id2]['compartments'].add(species_info[specie_id]['compartment'])
                        elif part.tag.endswith ('listOfProducts'):
                            for product in part:
                                assert product.tag.endswith('speciesReference'), `product.tag`
                                specie_id = product.attrib['species']
                                stoichiometry = obj2num(product.attrib.get('stoichiometry', default_stoichiometry))
                                reactions_info[reaction_id]['products'].append(specie_id)
                                try:
                                    specie_index = species.index(specie_id)
                                except ValueError:
                                    species.append(specie_id)
                                    specie_index = len(species)-1
                                assert stoichiometry,`stoichiometry`
                                matrix[specie_index, reaction_index] = stoichiometry
                                species_reactions[specie_index].append(reaction_index)
                                reactions_species[reaction_index].append(specie_index)
                                reactions_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                                if reaction_index2 is not None:
                                    reactions_info[reaction_id2]['products'].append(specie_id)
                                    matrix[specie_index, reaction_index2] = -stoichiometry
                                    species_reactions[specie_index].append(reaction_index2)
                                    reactions_species[reaction_index2].append(specie_index)
                                    reactions_info[reaction_id2]['compartments'].add(species_info[specie_id]['compartment'])
                        elif part.tag.endswith ('listOfModifiers'):
                            for modifier in part:
                                assert modifier.tag.endswith('modifierSpeciesReference'), `modifier.tag`
                                specie_id = product.attrib['species']
                                reactions_info[reaction_id]['modifiers'].append(specie_id)
                                reactions_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                            continue
                        elif part.tag.endswith ('annotation'):
                            reactions_info[reaction_id]['annotation'].append(part.text)
                            continue
                        elif re.match(r'.*(kineticLaw|notes)\Z', part.tag):                            
                            continue
                        else:
                            print 'get_stoichiometry:warning:unprocessed reaction element: %r' % (part.tag)
                            continue


            elif re.match (r'.*(annotation|notes|listOfSpeciesTypes|listOfUnitDefinitions)\Z', item.tag):
                pass
            else:
                print 'get_stoichiometry:warning:unprocessed model element: %r' % (item.tag)

    return matrix, species, reactions, species_info, reactions_info
    

def load_stoic_from_text(text, split_bidirectional_fluxes=False):
    """ Parse stoichiometry matrix from a string.

    Parameters
    ----------
    text : str
      A multiline string where each line contains a chemical reaction
      description. The description must be given in the following
      form: ``<sum of reactants> (=> | <=) <sum of producats>``. For example,
      ``A + 2*B => C``. Lines starting with ``#`` are ignored.

    split_bidirectional_fluxes : bool
      When True the bidirectional fluxes are split into two unidirectional fluxes.
      For example, the system ``A<=>B`` is treated as ``A=>B and B=>A``.

    Returns
    -------
    matrix_data : dict
      A dictionary representing a stoichiometry matrix.

    species : list
      A list of row names.

    reactions : list
      A list of column names.

    species_info : dict
    reactions_info : dict
    """
    #TODO: fill up species_info and reactions_info dictionaries

    def _split_sum (line):
        for part in line.split('+'):
            part = part.strip()
            coeff = ''
            while part and part[0].isdigit():
                coeff += part[0]
                part = part[1:].lstrip()
            if not coeff:
                coeff = '1'
            if not part:
                continue
            c = eval(coeff)
            assert type(c) == type(int())
            yield part, c

    matrix = {}
    reactions = []
    species = []
    reactions_info = defaultdict(lambda:dict(modifiers=[],reactants=[],products=[],
                                            boundary_specie_stoichiometry={},annotation=[],
                                            compartments = set()))
    species_info = defaultdict(lambda:list())
    info = defaultdict(lambda:list())
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith ('#'): continue

        if '|' in line:
            pair_name, mapping = line.split('|',1)
            pair_name = pair_name.strip()
            mapping = mapping.strip()
            assert '|' not in mapping, `mapping`
            if mapping == str(): continue
            info['rxn_pairs'].append((pair_name, mapping))
            continue
        
        if ':' in line:
            reaction_name, line = line.split (':',1)
            reaction_name = reaction_name.strip()
            line = line.strip()
            assert ':' not in line, `line`
            assert '=' in line, `line`
            if line == str(): continue
        else:
            reaction_name = None

        reaction_string = line
        info['rxns'].append((reaction_name, reaction_string))

    for pair_name, str_mapping in info['rxn_pairs']:
        mapping = eval(str_mapping)
        assert type(mapping) == type(dict()), `mapping`
        mets = pair_name.split('+')
        metA, metB = mets
        metA = metA.strip()
        metB = metB.strip()
        reverse_mapping = dict()
        for k, v in mapping.items():
            reverse_mapping[v] = k
        species_info[metA].append({metB:mapping})
        species_info[metB].append({metA:reverse_mapping})

    species_info = dict(species_info)
    length_dic = dict()
    for met, mappings in species_info.items():
        largest = 0
        for mapping in mappings:
            for atom_dic in mapping.values():
                for atom in atom_dic.keys():
                    if atom > largest:
                        largest = atom
        length_dic[met] = largest
    species_info['metabolite_lengths'] = length_dic
    #print species_info
    
    for reaction_name, reaction_string in info['rxns']:
        reversible = False
        left, right = reaction_string.split ('=')
        direction = '='
        if right.startswith('>'):
            right = right[1:].strip()
            direction = '>'
            if left.endswith ('<'):
                left = left[:-1].strip()
                reversible = True
        elif left.endswith ('>'):
            left = left[:-1].strip()
            direction = '>'
        elif left.endswith ('<'):
            left = left[:-1].strip()
            direction = '<'
        elif right.startswith ('<'):
            right = right[1:].strip()
            direction = '<'

        left_specie_coeff = list(_split_sum(left))
        right_specie_coeff = list(_split_sum(right))
        left_specie_names = [n for n,c in left_specie_coeff if n]
        right_specie_names = [n for n,c in right_specie_coeff if n]

        fname = ['R']
        rname = ['R']
        name0 = ''.join(left_specie_names)
        name1 = ''.join(right_specie_names)
        if name0:
            rname.append (name0)
        if name1:
            fname.append (name1)
            rname.append (name1)
        if name0:
            fname.append (name0)

        if direction=='<':
            if not reaction_name:
                reaction_name = '_'.join(fname)
                reaction_name2 = '_'.join(rname)
            else:
                reaction_name2 = 'r'+reaction_name
                if split_bidirectional_fluxes:
                    reaction_name = 'f'+reaction_name
        else:
            if not reaction_name:
                reaction_name2 = '_'.join(fname)
                reaction_name = '_'.join(rname)
            else:
                reaction_name2 = 'r'+reaction_name
                if split_bidirectional_fluxes:
                    reaction_name = 'f'+reaction_name

        reactions.append (reaction_name)
        reaction_index = reactions.index (reaction_name)
        if split_bidirectional_fluxes and reversible:
            reactions.append (reaction_name2)
            reaction_index2 = reactions.index (reaction_name2)
        else:
            reaction_index2 = None

        def matrix_add (i,j,c):
            v = matrix.get ((i,j))
            if v is None:
                matrix[i, j] = c
            else:
                matrix[i, j] = v + c

        for specie, coeff in left_specie_coeff:
            if specie not in species:
                species.append (specie)
            specie_index = species.index (specie)
            if direction=='<':
                if reaction_index2 is not None:
                    matrix_add(specie_index, reaction_index2, -coeff)
                matrix_add(specie_index, reaction_index, coeff)
            else:
                if reaction_index2 is not None:
                    matrix_add(specie_index, reaction_index2, coeff)
                matrix_add(specie_index, reaction_index, -coeff)
            
        for specie, coeff in right_specie_coeff:
            if specie not in species:
                species.append (specie)
            specie_index = species.index (specie)
            if direction=='<':
                if reaction_index2 is not None:
                    matrix_add(specie_index, reaction_index2, coeff)
                matrix_add(specie_index, reaction_index, -coeff)
            else:
                if reaction_index2 is not None:
                    matrix_add(specie_index, reaction_index2, -coeff)
                matrix_add(specie_index, reaction_index, coeff)

        if split_bidirectional_fluxes:
            reactions_info[reaction_name]['reversible'] = False
            reactions_info[reaction_name]['reactants'] = left_specie_names
            reactions_info[reaction_name]['products'] = right_specie_names
            reactions_info[reaction_name]['forward'] = reaction_name
            reactions_info[reaction_name]['reverse'] = None

            if reversible:
                reactions_info[reaction_name2]['reversible'] = False
                reactions_info[reaction_name2]['reactants'] = right_specie_names
                reactions_info[reaction_name2]['products'] = left_specie_names
                reactions_info[reaction_name2]['forward'] = reaction_name2
                reactions_info[reaction_name2]['reverse'] = None
            # TODO: set  reactions_info[reaction_name]['name']
        else:
            reactions_info[reaction_name]['reversible'] = reversible
            reactions_info[reaction_name]['reactants'] = left_specie_names
            reactions_info[reaction_name]['products'] = right_specie_names
            if reversible:
                reactions_info[reaction_name]['forward'] = 'f'+reaction_name
                reactions_info[reaction_name]['reverse'] = 'r'+reaction_name
            else:
                reactions_info[reaction_name]['forward'] = 'f'+reaction_name
                reactions_info[reaction_name]['reverse'] = None

            reactions_info[reaction_name]['name'] = reaction_string

    return matrix, species, reactions, species_info, reactions_info
