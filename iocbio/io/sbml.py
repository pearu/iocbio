""" A simple SBML-XML file reader

References
----------
http://sbml.org/Documents/Specifications

Module content
--------------
"""
# Author: Pearu Peterson
# Created: December 2010

from __future__ import absolute_import
from collections import defaultdict
import re
from lxml import etree

__all__ = ['get_stoichiometry']

def str2num(s):
    f = eval(s)
    i = int (f)
    if i==f:
        return i
    import sympycore, gmpy
    r = gmpy.f2q(f)
    n,d = int(r.numer ()), int (r.denom ())
    return sympycore.mpq((n,d))

def get_stoichiometry(file_name,
                      discard_boundary_species=False,
                      introduce_boundary_fluxes=False
                      ):
    """ Return stoichiometry information of a network described in a SBML-XML file.

    Parameters
    ----------
    file_name : str
      Path to SMBL-XML file.

    discard_boundary_species : bool

      When True then discard species that are reactants or products of
      the full network. The corresponding stoichiometry system will be
      open. For example, in a reaction ``A -> B -> C`` the species A
      and C are reactant and product of the system and after
      discarding A and C, the system will be open: ``-> B ->`` .
      In the case of a reaction `` -> A -> B + C`` the system will be
      made open by adding new reactions `` B-> `` and `` C -> ``.

    introduce_boundary_fluxes : bool

      When True then introduce boundary fluxes to boundary species.
      The corresponding stoichiometry system will be open.  For
      example, in a reaction ``A -> B -> C`` the species A and C are
      reactant and product of the system and after introducing
      boundary fluxes, the system will be open: ``-> A -> B -> C ->``.
      New flux names start with prefix 'BR_'.

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
    reaction_info = defaultdict(lambda:dict(modifiers=[],reactants=[],products=[],
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
                    species_info[specie.attrib['id']]['name'] = specie.attrib['name']
            elif item.tag.endswith('listOfReactions'):
                for reaction in item:
                    reaction_id = reaction.attrib['id']
                    assert reaction_id not in reactions,`reaction_id`
                    reactions.append(reaction_id)
                    reaction_index = len(reactions)-1
                    reaction_info[reaction_id]['name'] = reaction.attrib['name']
                    reaction_info[reaction_id]['reversible'] = eval(reaction.attrib.get('reversible', 'False').title())
                    for part in reaction:
                        if part.tag.endswith ('listOfReactants'):
                            for reactant in part:
                                assert reactant.tag.endswith('speciesReference'), `reactant.tag`
                                specie_id = reactant.attrib['species']
                                stoichiometry = -str2num(reactant.attrib.get('stoichiometry', default_stoichiometry))
                                reaction_info[reaction_id]['reactants'].append(specie_id)
                                try:
                                    specie_index = species.index(specie_id)
                                except ValueError:
                                    species.append(specie_id)
                                    specie_index = len(species)-1
                                matrix[specie_index, reaction_index] = stoichiometry
                                species_reactions[specie_index].append(reaction_index)
                                reactions_species[reaction_index].append(specie_index)                                
                                reaction_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                        elif part.tag.endswith ('listOfProducts'):
                            for product in part:
                                assert product.tag.endswith('speciesReference'), `product.tag`
                                specie_id = product.attrib['species']
                                stoichiometry = str2num(product.attrib.get('stoichiometry', default_stoichiometry))
                                reaction_info[reaction_id]['products'].append(specie_id)
                                try:
                                    specie_index = species.index(specie_id)
                                except ValueError:
                                    species.append(specie_id)
                                    specie_index = len(species)-1
                                matrix[specie_index, reaction_index] = stoichiometry
                                species_reactions[specie_index].append(reaction_index)
                                reactions_species[reaction_index].append(specie_index)
                                reaction_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                        elif part.tag.endswith ('listOfModifiers'):
                            for modifier in part:
                                assert modifier.tag.endswith('modifierSpeciesReference'), `modifier.tag`
                                specie_id = product.attrib['species']
                                reaction_info[reaction_id]['modifiers'].append(specie_id)
                                reaction_info[reaction_id]['compartments'].add(species_info[specie_id]['compartment'])
                            continue
                        elif part.tag.endswith ('annotation'):
                            reaction_info[reaction_id]['annotation'].append(part.text)
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
    stoichiometry_matrix = matrix

    if discard_boundary_species:
        # make the network open by removing boundary species, i.e. species that are reactants or products of the network
        boundary_species = []
        extra_reactions = []
        for specie_index in range(len(species)):
            specie_reactions = species_reactions[specie_index]
            if len(specie_reactions)>1:
                continue
            reaction_index = specie_reactions[0]
            reaction_id = reactions[reaction_index]
            stoichiometry = matrix[specie_index, reaction_index]
            if 1:
                # check that the specie has different stoichiometry sign from other species in the reaction
                other_species = [i for i in reactions_species[reaction_index] if i!=specie_index]
                flag = False
                for other_specie_index in reactions_species[reaction_index]:
                    if other_specie_index == specie_index:
                        continue
                    if len(species_reactions[other_specie_index])>1:
                        flag = True
                if not flag:
                    continue
                other_stoichiometries = [matrix[i, reaction_index] for i in other_species]
                if [s for s in other_stoichiometries if s * stoichiometry > 0]:
                    continue
                    # need additional flux
                    if stoichiometry>0:
                        new_reaction_id = '%s_output' % (species[specie_index])
                    else:
                        new_reaction_id = '%s_input' % (species[specie_index])
                    assert new_reaction_id not in reactions,`new_reaction_id`
                    new_reaction_index = len (reactions)
                    reactions.append(new_reaction_id)
                    specie_reactions.append(new_reaction_index)
                    if stoichiometry>0:
                        reaction_info[new_reaction_id]['reactants'].append(species[specie_index])
                    else:
                        reaction_info[new_reaction_id]['products'].append(species[specie_index])
                    matrix[specie_index, new_reaction_index] = -stoichiometry
                    print 'created new reaction', new_reaction_id
                    #print reaction_id, specie_index, other_species, stoichiometry, other_stoichiometries
                    continue
            boundary_species.append(specie_index)
            reaction_id = reactions[reaction_index]
            specie_id = species[specie_index]
            reaction_info[reaction_id]['boundary_specie_stoichiometry'][specie_id] = stoichiometry
        # discard rows corresponding to boundary species
        i = 0
        stoichiometry_matrix = {}
        new_species = []
        for specie_index in range(len(species)):
            if specie_index in boundary_species:
                continue
            new_species.append(species[specie_index])
            for reaction_index in species_reactions[specie_index]:
                stoichiometry_matrix[i, reaction_index] = matrix[specie_index, reaction_index]
            i += 1
        species = new_species

    if introduce_boundary_fluxes:
        for specie_index in range(len(species)):
            specie_reactions = species_reactions[specie_index]
            if len(specie_reactions)>1:
                continue
            reaction_index = specie_reactions[0]
            reaction_id = reactions[reaction_index]
            stoichiometry = matrix[specie_index, reaction_index]
            specie_id = species[specie_index]
            new_reaction_id = 'BR_%s' % (specie_id)
            new_reaction_index = len(reactions)
            reactions.append(new_reaction_id)
            new_stoichiometry = -1 if stoichiometry>0 else 1
            stoichiometry_matrix[specie_index, new_reaction_index] = new_stoichiometry
            if new_stoichiometry>0:
                reaction_info[new_reaction_id]['products'].append(specie_id)
            else:
                reaction_info[new_reaction_id]['reactants'].append(specie_id)
            
    return stoichiometry_matrix, species, reactions, species_info, reaction_info
