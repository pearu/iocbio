#
# A script that computes flux relations for the yeast example.
#
import utils
flush = utils.flush
from sympycore import Matrix

from IO import load_stoic_from_sbml

import pprint
def pf(item):
    return pprint.pformat(item)

def pp(item):
    print pprint.pformat(item)


from yeast_example_info import external_fluxes, internal_fluxes, sbml_file

#sbml_file = 'generated/small_example'

#
# Load stoichiometry matrix and system information from an SBML file
#

matrix_data, species, reactions, species_info, reaction_info = \
             load_stoic_from_sbml(sbml_file + '.xml',
                                  discard_boundary_species = True)

positive_reactions = [r for r,info in reaction_info.iteritems () if not info['reversible']]

#
# Compute row-echelon form of a matrix using Gauss-Jordan elimination
#

# specify dependent fluxes as a list of leading column canditates
independent_cols = [reactions.index(reaction_id) for reaction_id in external_fluxes.keys() + internal_fluxes.keys()]
#independent_cols = []
leading_cols_canditates = [j for j in range (len (reactions)) if j not in independent_cols]

# make a sympycore matrix
matrix = Matrix (len (species), len(reactions), matrix_data)

# apply Gauss-Jordan elimination
print 'Computing GJE...',flush,
gj, row_operations, leading_rows, leading_cols, zero_rows = \
    matrix.get_gauss_jordan_elimination_operations(leading_cols=leading_cols_canditates,
                                                   leading_row_selection='sparsest first',
                                                   leading_column_selection='sparsest first',
                                                   verbose = False)
print 'done'
#
# Compute flux relation expressions
#
out = utils.compute_relation_matrix_from_gauss_jordan_data(leading_rows, leading_cols, reactions, gj.data)
dependent_flux_variables, independent_flux_variables, relation_matrix = out

save_matrix = True
if save_matrix:
    f = open(sbml_file + '_relation_matrix.py', 'w')
    f.write('from sympycore import mpq\n\n')
    f.write('relation_matrix = ' + pf(relation_matrix))
    f.write('\n\nindependent_flux_variables = ' + pf(independent_flux_variables))
    f.write('\n\ndependent_flux_variables = ' + pf(dependent_flux_variables))
    f.close()
 
ext_subs_dict = {}                                                
ext_subs_dict = external_fluxes.copy()
int_subs_dict = ext_subs_dict.copy()
int_subs_dict.update(internal_fluxes)

valid_python = False

if 1:
    relations = utils.make_relation_expressions(dependent_flux_variables,
                                                independent_flux_variables,
                                                relation_matrix,
                                                #subs_dict=subs_dict,
                                                valid_python=valid_python,
                                                format='%.3f')

    relations2 = utils.make_relation_expressions(dependent_flux_variables,
                                                 independent_flux_variables,
                                                 relation_matrix,
                                                 subs_dict=ext_subs_dict,
                                                 valid_python=valid_python,
                                                 format='%.3f')

    relations3 = utils.make_relation_expressions(dependent_flux_variables,
                                                 independent_flux_variables,
                                                 relation_matrix,
                                                 subs_dict=int_subs_dict,
                                                 valid_python=valid_python,
                                                 format='%.3f')

#
# Save results to text file.
#

utils.save_relations(sbml_file + '_relations.py',
                     dependent_flux_variables,
                     independent_flux_variables,
                     relations)

utils.save_relations(sbml_file + '_values.py',
                     dependent_flux_variables,
                     independent_flux_variables,
                     relations3)

    
if 1:
    print 'Making cdd matrix...',flush,
    cdd_mat, column_names = utils.make_cdd_matrix(dependent_flux_variables,
                                                  independent_flux_variables,
                                                  relation_matrix,
                                                  #subs_dict=int_subs_dict,
                                                  positive_flux_variables = positive_reactions,
                                                  )
    cdd_mat2, column_names2 = utils.make_cdd_matrix(dependent_flux_variables,
                                                    independent_flux_variables,
                                                    relation_matrix,
                                                    subs_dict=ext_subs_dict,
                                                    positive_flux_variables = positive_reactions,
                                                  )
    print 'done'

    print 'Reducing cdd matrix...',flush,
    reduced_poly, reduced_mat = utils.reduce_cdd_matrix(cdd_mat)
    reduced_poly2, reduced_mat2 = utils.reduce_cdd_matrix(cdd_mat2)
    print 'done'

    print 'Computing extremes..',flush,
    #extreme_flux_names, extremes = utils.extremes_from_cdd_generators (reduced_poly.get_generators(), column_names, independent_flux_variables)
    extreme_flux_names2, extremes2 = \
                         utils.extremes_from_cdd_generators (reduced_poly2.get_generators(),
                                                             column_names2,
                                                             independent_flux_variables,
                                                             number2str = lambda x: '%s' % (float (x)),
                                                             )
    print 'done'

    print 'Computing constraints...',flush,
    constraints = utils.relations_from_cdd_matrix (reduced_mat, column_names, discard_equalities=True,
                                                   positive_flux_variables = positive_reactions,
                                                   #number2str = lambda x: '%s' % (float (x))
                                                   )
    print '\n'.join (constraints)
    constraints2 = utils.relations_from_cdd_matrix (reduced_mat2, column_names2, discard_equalities=True,
                                                    positive_flux_variables = positive_reactions,
                                                    number2str = lambda x: '%s' % (float (x))
                                                    )
    print '\n'.join (constraints2)
    print 'done'


    print 'Nof positive reactions:',len (positive_reactions)
    print sorted([n for n in positive_reactions if n not in external_fluxes])

