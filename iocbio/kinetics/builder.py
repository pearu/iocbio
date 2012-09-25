"""
builder - provides IsotopeModel, a base class for generating kinetic
    equations for reactions with isotopologues.

For more information, see

  http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator

"""

from __future__ import division

__all__ = ['IsotopeModel']

import sys
import pprint
import itertools
from fractions import Fraction
from collections import defaultdict

make_indices = lambda repeat: map(''.join,itertools.product('01', repeat=repeat))

class Flush:
    def __str__ (self):
        sys.stdout.flush ()
        return ''
flush = Flush ()
 
def round_float(float, precision=0, return_string=True):
    new = round(float, precision)
    if return_string:
        if precision < 0:
            p = '0'
        else:
            p = str(precision)
        exec('s = "%0.' + p + 'f" %new')
        return str(int(s)) 
    else:
        return new  
    
############################################
########                          ##########
########  Start of symbolic code  ##########
########                          ##########
############################################

class StrContext:
    context = 'str'
    name_map = {}
    def __init__(self, context='str', name_map={}):
        self.context = context
        self.name_map = name_map
    def __enter__ (self):
        self.previous_context = StrContext.context
        self.previous_name_map = StrContext.name_map
        StrContext.context = self.context
        StrContext.name_map = self.name_map
    def __exit__ (self, *args):
        StrContext.context = self.previous_context
        StrContext.name_map = self.previous_name_map

    @staticmethod
    def map(name, default=None):
        if StrContext.context == 'latex':
            latex = getattr(name, 'latex', None)
            if latex is not None:
                return latex
            n = StrContext.name_map.get(name)
            if n is not None:
                return n
            if default is not None:
                return default
            if isinstance(name, symbol):
                return str.__str__(name)
            return name

        elif StrContext.context=='ccode':
            ccode = getattr (name, 'ccode', None)
            if ccode is not None:
                return ccode
            n = StrContext.name_map.get(name)
            if n is not None:
                return n
            if default is not None:
                return default
            if isinstance (name, symbol):
                return str.__str__(name)
            return name
        
        return name

class symbol(str):

    def __new__ (cls, prefix, index='', latex=None):
        if prefix is None:
            obj = str.__new__(cls, index)
        else:
            if isinstance(prefix, symbol):
                prefix = prefix.prefix + prefix.index
            obj = str.__new__(cls, prefix+index)
        assert not isinstance(prefix, symbol),`prefix`
        obj.prefix = prefix
        obj.index = index
        obj.latex = latex
        return obj

    def normal (self):
        return self

    def collect(self, **options):
        return self

    def subs(self, relations, **options):
        for terms_list, name in relations:
            if len(terms_list) != 1:
                continue
            if terms_list[0]==self:
                return name
        return self

    @property
    def symbols(self):
        return set([self])

    @property
    def coeff_term(self):
        return number(1), self

    def __str__(self):
        return StrContext.map(self)
        if StrContext.context=='latex':
            if self.latex is not None:
                return self.latex
            return StrContext.map(self)
        return str.__str__ (self)

def symbol_latex(s, i, skip_replace=[]):
    if s[-1]=='_':
        ln = '\\'+s[:-1]+'name'
        for index, i1 in enumerate(i.split('_')):
            if index in skip_replace:
                ln = ln + '\\Oba' + ('r'*len (i1)) +' ' + i1
            else:
                ln = ln + '\\Oba' + ('r'*len (i1)) + ' ' + i1.replace('X','*').replace('0', r'\circ').replace ('1',r'\bullet')
        return symbol(s,i,latex=ln)
    else:
        ln = '\\'+s+'name'
        for i1 in i.split('_'):
            ln = ln + '\\Oba' + ('r'*len (i1)) + i1.replace('0',r'\circ').replace ('1',r'\bullet')
        return symbol(s,i,latex=ln)
    return symbol(s,i,latex=ln)

class number(Fraction):

    @property
    def symbols(self):
        return set()

    def subs(self, relations, **options):
        return self

    def normal(self):
        return self

    def collect(self, **options):
        return self

    @property
    def coeff_term (self):
        return self, number (1)

    def __str__ (self):
        if StrContext.context=='latex':
            if self.denominator != 1:
                return '\\tfrac{%s}{%s}' % (self.numerator, self.denominator)
        return Fraction.__str__(self)

class Sum(object):
    """ Holds terms of a kinetic equation.
    """
    def __init__ (self):
        self.data = defaultdict(lambda : Terms(normalize=False))

    def set_parent (self, parent):
        self.parent = parent

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.data == other.data
        return False

    def __ne__(self, other):
        return not (self==other)

    def append(self, prefix, *factors):
        self.data[prefix].data.append(Factors(*factors))

    def extend(self, other):
        for prefix, terms in other.data.iteritems():
            self.data[prefix].data.extend(terms.data)

    def collect(self, **options):
        new_sum = Sum()
        for prefix, terms in self.data.iteritems():
            new_sum.data[prefix] = terms.collect(**options)
        return new_sum

    def subs(self, relations, **options):
        new_sum = Sum()
        for prefix, terms in self.data.iteritems():
            new_sum.data[prefix] = terms.subs(relations, **options)
        return new_sum

    def normal(self):
        new_sum = Sum()
        for prefix, terms in self.data.iteritems():
            new_sum.data[prefix] = terms.normal()
        return new_sum

    def __str__(self):
        mul_op = '*'
        if StrContext.context=='latex':
            mul_op = ''
        l = []
        for k in sorted(self.data):
            rate = k[0]+StrContext.map(k[1:])
            l.append('%s%s(%s)' % (rate, mul_op, self.data[k]))

        if StrContext.context == 'latex':
            return ''.join(l)[1:]
        else:
            return ''.join(l)

    __repr__ = __str__
    
    @property
    def symbols(self):
        s = set()
        for prefix, terms in self.data.iteritems():
            s = s.union(terms.symbols)
        return s

class Factors(object):

    def __new__(cls, *data, **options):
        if options.get('normalize', True):
            new_data = []
            for d in data:
                if isinstance(d, number) and d==number(1):
                    continue
                new_data.append(d)
            data = new_data
            if not data:
                return number(1)
            if len(data)==1:
                return data[0]

        obj = object.__new__(cls)
        obj.data = list(data)
        return obj

    def __repr__(self):
        return self.__class__.__name__ + repr(tuple(self.data))

    def __str__(self):
        mul_op = '*'
        if StrContext.context=='latex':
            mul_op = ''
        l = []
        numbers = []
        for f in self.data:
            if isinstance (f, number):
                numbers.append(str (f))
            else:
                f = str(f)
                if '+' in f:
                    f = '(%s)' % f
                l.append(f)
        l = numbers + sorted(l)
        return mul_op.join(l)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return sorted(self.data)==sorted(other.data)
        return False

    def __ne__(self, other):
        return not (self==other)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(self.data)))

    def append(self, factor):
        if isinstance(factor, Factors):
            self.data.extend(factor.data)
        elif isinstance(factor, number) and factor==number(1):
            pass
        else:
            self.data.append(factor)

    def normal(self):
        data = []
        c = 1
        for f in self.data:
            f = f.normal()
            if isinstance(f, number):
                if f==number(1):
                    continue
                c = c * f
            elif isinstance(f, Factors):
                for f1 in f.data:
                    if isinstance(f1, number):
                        if f1==number(1):
                            continue
                        c = c * f1
                    else:
                        data.append(f1)
            else:
                data.append(f)
        if c!=1:
            data.insert(0, number(c))
        return Factors(*data)

    def subs(self, relations, **options):
        factors = Factors(normalize=False)
        numbers = []
        for f in self.data:
            f = f.subs(relations, **options)
            factors.append(f)            
        return factors.normal()

    def collect(self, **options):
        factors = Factors(normalize=False)
        for f in self.data:
            f = f.collect(**options)
            factors.append(f)
        return factors.normal()

    @property
    def symbols(self):
        s = set()
        for f in self.data:
            s = s.union(f.symbols)
        return s

    @property
    def coeff_term(self):
        r = self.normal()
        if isinstance(r, self.__class__):
            if isinstance(r.data[0], number):
                return r.data[0], Factors(*r.data[1:])
            return number(1), r
        return r.coeff_term()
        
class Terms(object):

    def __new__(cls, *data, **options):
        if options.get('normalize', True):
            if not data:
                return number(0)
            if len(data)==1:
                return data[0]
        obj = object.__new__(cls)
        obj.data = list(data)
        return obj

    def __repr__(self):
        return self.__class__.__name__ + repr(tuple(self.data))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return sorted(self.data)==sorted(other.data)
        return False

    def __ne__(self, other):
        return not (self==other)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(self.data)))

    def append(self, term):
        if isinstance(term, Terms):
            self.data.extend(term.data)
        elif isinstance(term, number) and term==number(0):
            pass
        else:
            self.data.append(term)

    collect_counter = 0

    def collect(self, **options):
        symbols_only = options.get('symbols_only', True)

        terms = Terms(*[term.collect() for term in self.data]).normal()
        if not isinstance(terms, Terms):
            return terms

        Terms.collect_counter += 1
        show_counters = [] 
        collect_counter = Terms.collect_counter
        if collect_counter in show_counters:
            print 'IN[%s]\n  ' % (collect_counter),
            print terms

        common_factors = None

        factor_map = defaultdict(set)
        for i, term in enumerate(terms.data):
            if isinstance(term, Factors):
                for f in term.data:
                    factor_map[f].add(i)
            else:
                factor_map[term].add(i)
            if common_factors is None:
                if isinstance(term, Factors):
                    common_factors = set(term.data)
                elif isinstance(term, Fraction):
                    common_factors = None
                    break
                else:
                    common_factors = set(term)
            else:
                if isinstance(term, Factors):
                    common_factors = common_factors.intersection(set(term.data))
                else:
                    common_factors = common_factors.intersection(set([term]))
        if common_factors:
            rest = []
            for term in terms.data:
                if isinstance(term, Factors):
                    factors = term.data[:]
                    for f in common_factors:
                        factors.remove(f)
                    rest.append(Factors(*factors))
                else:
                    assert term in common_factors
                    rest.append(number(1))

            if collect_counter in show_counters:
                print 'OUT0[%s]\n  ' % (collect_counter),
                print Factors(Terms(*rest), *common_factors)

            return Factors(Terms(*rest), *common_factors)

        common_terms = {}

        if collect_counter in show_counters:
            print terms
            print factor_map

        while factor_map:
            mxlen = max(map(len, factor_map.values()))
            if mxlen==0:
                break
            for k, l in factor_map.items():
                if len(l)==mxlen:
                    common_terms[k] = l.copy ()
                    for k1 in factor_map:
                        if k!=k1:
                            for i in l:
                                try:
                                    factor_map[k1].remove(i)
                                except (ValueError, KeyError):
                                    pass
                    del factor_map[k]
                    break

        if not common_terms:
            if collect_counter in show_counters:
                print 'OUT1[%s]\n  ' % (collect_counter),
                print terms
            return terms

        lst = []

        for f, indices in common_terms.iteritems():
            tlst = []
            for i in indices:
                if isinstance(terms.data[i], Factors):
                    factors = terms.data[i].data[:]
                    factors.remove(f)
                else:
                    assert terms.data[i]==f
                    factors = [number(1)]
                term = Factors(*factors)
                tlst.append(term)
            lst.append(Factors(f, Terms(*tlst)))

        terms = Terms(*lst)

        if collect_counter in show_counters:
            print 'OUT2[%s]\n  ' % (collect_counter),
            print terms
            print common_terms

        return terms


    def subs(self, relations, **options):
        terms = Terms(normalize=False)
        data = terms.data
        for d in self.data:
            terms.append(d.subs(relations, **options))

        for terms_list, name in relations:
            flag = True
            for t in terms_list:
                if t not in data:
                    flag = False
                    break
            if flag:
                map(data.remove, terms_list)
                data.append(name)

        return terms.normal()

    def normal (self):
        data = []
        d = {}
        for f in self.data:
            f = f.normal()
            if f==number(0):
                continue
            coeff, term = f.coeff_term
            if term in d:
                d[term] = d[term] + coeff
            else:
                d[term] = coeff
        data = [Factors(number(c),t).normal() for t,c in d.iteritems()]
        return Terms(*data, **dict(normalize=True))

    def __str__(self):
        if not self.data:
            return '0'
        return '+'.join(sorted(map (str, self.data)))

    @property
    def symbols(self):
        s = set()
        for f in self.data:
            s = s.union(f.symbols)
        return s

    @property
    def coeff_term(self):
        r = self.normal()
        if isinstance(r, self.__class__):
            coeffs, terms = [], []
            denoms = []
            for t in r.data:
                coeff, term = t.coeff_term
                coeffs.append(coeff)
                terms.append(term)
            if len(set(coeffs))==1:
                return coeffs[0], Terms(*terms)
            return number(1), r
        return r.coeff_term

def load_stoic_from_text(text, split_bidirectional_fluxes=False):
    """ Parse stoichiometry matrix from a string.

    Parameters
    ----------
    text : str
      A multiline string where each line contains a chemical reaction
      description. The description must be given in the following
      form: ``<sum of reactants> (=> | <=) <sum of producats>``. For example,
      ``A + 2 B => C``. Lines starting with ``#`` are ignored. To
      assign a name to reaction, start the line with the name following
      a colon. For example, ``f : A + 2 B => C``.

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

class Labels(dict):

    def __init__(self, parent, reaction):
        dict.__init__(self)
        self.parent = parent
        self.reaction = reaction

    def match (self, reaction_pattern, transport=False):
        if transport and not (len (self.reaction[1]) == len (self.reaction[2]) == 1):
            return False
        self.parent.get_labels(self, reaction_pattern, self.reaction)
        return not not self

    def __getattr__(self, specie):
        return self[specie]

    def __setitem__ (self, item, value):
        current_value = self.get (item)
        if isinstance (current_value, tuple):
            value = current_value + (value,)
        elif current_value is not None:
            value = (current_value, value)
        return dict.__setitem__ (self, item, value)

class IsotopeModel(object):
    """ Base class for generating kinetic equations for reactions with
    isotopologues.

    See oxygen_isotope_model.py for example usage.
    """
    system_string = None
    species = {} 
    reactions = [] 
    latex_name_map = {}

    def __init__(self,
                 system_string=None, 
                 defined_labeling = {},
                 model_name = None,
                 verbose=True,
                 ):
        """ Construct a model builder.

        Parameters
        ----------
        system_string : str
          Specify system string containing reaction definitions and
          atom labeling information.  See parse_system_string
          documentation for syntax of the system string.  When
          system_string is not specified then system_string attribute
          of the derived class is used.

        defined_labeling : dict
          Defines a map of isotope species and their defined values as
          C/C++ expressions.

        model_name : str
          Specify the name of the model that defines a part of the
          generated files.

        verbose : bool
          When True then show information about model building
          progress.

        See oxygen_isotope_model.py for example usage.
        """
        if system_string is not None:
            self.system_string = system_string
        if self.system_string is None:
            raise ValueError ('Must specify system_string as class attribute or constructor argument')
        raw_species, self.reactions = self.parse_system_string(self.system_string)

        self.species = {}
        self.c_name_map = {}
        for sp in raw_species:
            self.species[sp] = self.index_dic[sp]
            self.c_name_map[sp] = sp

        self.defined_labeling = defined_labeling
        self.use_sum = False
        self.use_mass = True
        self.replace_total_sum_with_one = False

        self.import_dir = 'local'

        if model_name is None:
            model_name = self.__class__.__name__
            
        self.model_name = model_name

        if self.use_mass:
            self.model_name += '_mass'
        if self.use_sum:
            self.model_name += '_sum'
        if self.replace_total_sum_with_one:
            self.model_name += '_repl1'

        self.verbose = verbose
        self._pools = None
        self._pools2 = None
        self._kinetic_equations = None
        self._kinetic_terms = None
        self._relations = None
        self._pool_relations = None
        self._pool_relations_quad = None

    def parse_system_string(self, system_string):
        """ Parse a string of chemical reactions.

        Parameters
        ----------
        system_string : str
          A multiline string where each line contains a chemical reaction
          description. The description must be given in the following
          form: ``<sum of reactants> (=> | <=) <sum of producats>``. For example,
          ``A + 2 B => C``. Lines starting with ``#`` are ignored. To
          assign a name to reaction, start the line with the name following
          a colon. For example, ``f : A + 2 B => C``.

        Returns
        -------
        species : list
          A list of species names.

        reactions : list
          A list of reaction data. Reaction data is a dictionary::

            dict(reactants=<list of reactants names>,
                 products=<list of products names>,
                 reversible=<bool>,
                 forward=<name of forward flux>
                 reverse=<name of reverse flux>)
        """
        matrix, raw_species, rxns, species_info, reactions_info = \
            load_stoic_from_text(system_string)
        reactions = []
        species = set ()
        self.index_dic = index_dic = {}
        for n, d in reactions_info.iteritems():
            r = {}
            for k,v in d.iteritems ():
                if k in ['reversible','forward', 'reverse']:
                    r[k] = v
            ri = rxns.index(n)
            reactants = []
            products = []
            for (si,j), stoic in matrix.iteritems ():
                if j==ri and stoic:
                    specie = raw_species[si]
                    i = specie.find('[')
                    if i>=0:
                        assert specie.endswith (']'), `specie`
                        labeling_pattern = specie[i+1:-1]
                        specie = specie[:i]
                        all_indices = []
                        for l in labeling_pattern.split('_'):
                            indices = make_indices(len(l))
                            all_indices.append (indices)
                        specie_indices = []
                        for indices in itertools.product(*all_indices):
                            specie_indices.append('_'.join (indices))
                        if specie in self.index_dic:
                            assert specie_indices==self.index_dic[specie]
                        if specie not in index_dic:
                            index_dic[specie] = specie_indices
                        else:
                            if index_dic[specie] != specie_indices:
                                raise ValueError ('Mismatch of specie indices: %s != %s' % (index_dic, specie_indices))
                    species.add (specie)
                    if stoic > 0:
                        products.extend([specie]*stoic)
                    else:
                        reactants.extend([specie]*abs(stoic))
            if set(reactants)==set(products):
                # artificial reaction used for labeling definition
                continue
            r['reactants'] = reactants
            r['products'] = products

            reactions.append(r)
        return species, reactions

    def normalize_reaction(self, reaction):
        left, right = reaction.split('=')
        left = left.strip ()
        right = right.strip ()
        op = '='
        if left.endswith('<'):
            op = '<' + op
            left = left[:-1].rstrip()
        if right.startswith ('>'):
            op = op + '>'
            right = right[1:].lstrip()
        reactants = map(str.strip, left.split('+'))
        products = map(str.strip, right.split('+'))

        return reactants, op, products

    def labels(self, reaction):
        """ Return Labels instance to be used in check_reaction as
        label matcher object.
        """
        return Labels(self, reaction)

    def get_labels(self, labels, pattern, (reaction, rindices, pindices)):
        """ If reaction matches with reaction pattern then return
        a Labels object containing the labeling information of
        species as attributes to the Labels object.
        """
        reactants, op, products = self.normalize_reaction(reaction)
        if op=='<=':
            op = '=>'
            reactants, products = products, reactants
            rindices, pindices = pindices, rindices

        pattern_reactants, pattern_op, pattern_products = self.normalize_reaction(pattern)
        if pattern_op=='<=':
            pattern_op = '=>'
            pattern_reactants, pattern_products = pattern_products, pattern_reactants
        if pattern_op=='<=>':
            if set(pattern_reactants)==set(products) and set(pattern_products)==set (reactants):
                pattern_reactants, pattern_products = pattern_products, pattern_reactants
        if set(pattern_reactants)==set (reactants) and set (pattern_products)==set (products):
            #print pattern_reactants, pattern_op, pattern_products
            for i,r in enumerate(reactants):
                labels[r] = rindices[i]
            for i,p in enumerate(products):
                labels[p] = pindices[i]

    def write_ccode(self, stream=sys.stdout):
        """ Generate a C source code of the model.
        """
        use_sum = self.use_sum
        use_mass = self.use_mass
        pool_relations = self.pool_relations
        it_eqns = self.isotopomer_equations
        eqns = self.mass_isotopomer_equations
        reactions = self.reactions

        c_variables = dict(input_list=None,
                           flux_list=None,
                           pool_list=None,
                           solver_time=None,
                           out=None)

        in_list = str()
        for variable in c_variables.keys():
            if variable == 'out':
                continue
            in_list += 'double* %s, ' %(variable)
        in_list = in_list[:-2]
        var_list = in_list + ', double* out'

        c_header = 'void c_equations(%s)\n{\n' %(var_list)

        def get_name (r):
            r_copy = r
            while r and not r[0].isdigit ():
                r = r[1:]
            return r_copy[:len(r_copy)-len(r)]
        def is_unlabeled (r):
            while not r[0].isdigit ():
                r = r[1:]
            return sum(map(int,r.split ('_'))) == 0

        if use_mass:
            c_eqns = eqns
        else:
            c_eqns = it_eqns

        with StrContext ('ccode', self.c_name_map):
            stream.write ('/* See http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator */\n')
            stream.write ('/*\n%s\n*/\n' % ('\n'.join(line for line in self.system_string.splitlines () if line.strip() and not line.strip().startswith ('#'))))
            stream.write (c_header)
            
            # Assign the input_list to their isotopologue names.
            initial_value_list = []
            derivatives_list = []
            index = 0
            for k in sorted (c_eqns):
                it_key = StrContext.map(k)
                if it_key in self.defined_labeling:
                    value = self.defined_labeling[it_key]
                    stream.write('double %s = %s ;\n' %(it_key, value))
                elif use_sum and is_unlabeled (k):
                    count = -1

                    for k1 in c_eqns:
                        if get_name (k1) == get_name (k):
                            count += 1
                    s = ['input_list[%s]' % (index+c) for c in range (count)]
                    stream.write('double %s = 1.0-(%s);\n' %(it_key, '+'.join(s)))
                else:
                    stream.write('double %s = input_list[%s] ;\n' %(it_key, index))
                    initial_value_list.append(it_key)
                    derivatives_list.append(it_key)
                    index += 1
            c_variables['input_list'] = initial_value_list
            c_variables['out'] = derivatives_list
            stream.write('\n')

            # Assign the fluxes to their input list.
            flux_list = []
            flux_index = 0
            for rxn in sorted(reactions):
                r_key = rxn['forward']
                stream.write('double %s = flux_list[%s] ;\n' %(r_key, flux_index))
                flux_list.append(r_key)
                flux_index += 1
                r_key = rxn['reverse']
                if r_key is None:
                    continue
                stream.write('double %s = flux_list[%s] ;\n' %(r_key, flux_index))
                flux_list.append(r_key)
                flux_index += 1
            c_variables['flux_list'] = flux_list

            stream.write('\n')

            # Assign the pools to their input list.
            pool_set = set()
            for k in sorted(eqns):
                if k.split('_')[0] == 'Wo':
                    continue
                pool_set.add(StrContext.map(k).split('_')[0])
            pool_list = []
            for index, pool in enumerate(pool_set):
                stream.write('double pool_%s = pool_list[%s] ;\n' %(pool, index))
                pool_list.append (pool)
            c_variables['pool_list'] = pool_list
            stream.write('\n')

            # Write the equations.
            index = 0
            for k in sorted (c_eqns):
                if use_sum and is_unlabeled (k):
                    continue
                if k.startswith ('Wo'):
                    continue
                expr = c_eqns[k]
                replacements = [('/2', '/2.0'), ('/3', '/3.0'), ('/4', '/4.0')]
                e = str(expr)
                for rt in replacements:
                    e = e.replace(rt[0], rt[1])
                pool = StrContext.map(k).split('_')[0]
                stream.write('/*d%s/dt=*/ out[%s] = ( %s )/ pool_%s ;\n\n' %(k, index, e, get_name(pool)))
                index += 1

            # End block.
            stream.write('} \n')

        self.c_variables = c_variables

    def compile_ccode(self, debug=False, stage=None):
        """
        Compile C source file containing the kinetic model.
        """
        file_names = dict(c='{0}.c'.format(self.model_name),
                          c_variables='{0}_c_variables.py'.format(self.model_name),
                          )
        if not debug:
            f = open(file_names['c'], 'w')
            self.write_ccode(f)
            f.close()
            print '\n\nWrote: ',file_names['c']
        else:
            self.write_ccode()

        if stage==1:
            return

        cv_string = pprint.pformat(self.c_variables)
        if not debug:
            f = open(file_names['c_variables'], 'w')
            f.write('c_variables = %s\n' %(cv_string))
            f.close()
            print 'Wrote: ',file_names['c_variables']
        else:
            print cv_string

        if debug:
            return True

    def check_reaction(self, (reaction_pattern, rindices, pindices)):
        """ Validate reaction.

        This method must be implemented in an user derived class.
        
        Parameters
        ----------
        reaction_pattern : str
          The reaction pattern is composed by the IsotopeModel class
          and has the following form: R1+R2+...RM(<=|=>)P1+P2+...PN
          where Ri, i=1..M, are the names of M reactants, and Pi,
          i=1..N, are the names of N products.  For example, reaction
          pattern could be 'ADP+P=>ATP+W'.

        rindices, pindices : tuple
          Reactant and product labeling indices are also composed by
          the IsotopeModel class. Labeling indices consists of
          strings representing labeling states where '0' refers to
          ublabeled state and '1' to labeled state. If a reactant or product
          has several groups of labeling states then they are separated
          with underscore ('_').
          For example, reactant indices could be ('010','0011') and
          product indices could be ('010_001', '1').

        Returns
        -------
        is_valid : bool
          Return True when reaction is possible. For examples above, a
          reaction ADP[010]+P[0011]=>ATP[010_001]+W[1] is possible and
          True should be returned while in the case of
          ADP[010]+P[0011]=>ATP[011_001]+W[0] False should be
          returned.
        """
        raise NotImplementedError('check_reaction for reaction_pattern=%r' % (reaction_pattern))

    def check_equivalence(self, s1, s2):
        if s1.prefix != s2.prefix:
            return False
        if s1==s2:
            return True
        parts1 = s1.index.split('_')
        parts2 = s2.index.split('_')
        for p1, p2 in zip (parts1, parts2):
            if p1.count ('1')!=p2.count ('1'):
                return False
        return True    

    def get_pool_name(self, specie):
        """
        Return the name of a pool where specie belongs.
        """
        l = []
        for i in specie.index.split('_'):
            l.append(str(i.count ('1')))
        return symbol_latex(specie.prefix+'_', '_'.join(l), skip_replace=range(len(l)))

    @property
    def kinetic_equations(self):
        """
        Generate kinetic equations.
        """
        if self._kinetic_equations is not None:
            return self._kinetic_equations
        verbose = self.verbose
        equations = defaultdict(Sum)

        kinetic_terms = []
        for reaction in self.reactions:
            reactants = reaction['reactants']
            products = reaction['products']
            forward_reaction_pattern = '%s=>%s' % ('+'.join(reactants), '+'.join(products))
            reverse_reaction_pattern = '%s<=%s' % ('+'.join(reactants), '+'.join(products))
            forward = reaction.get('forward')
            reverse = reaction.get('reverse')
            
            reactant_indices = [self.species[s] for s in reaction['reactants']]
            product_indices = [self.species[s] for s in reaction['products']]

            reactions = defaultdict(list)
            for rindices in itertools.product(*map(self.species.__getitem__, reactants)):
                rspecies = tuple([symbol_latex(p,i) for p,i in zip(reactants, rindices)])
                for pindices in itertools.product(*map(self.species.__getitem__, products)):
                    pspecies = tuple([symbol_latex(p,i) for p,i in zip(products, pindices)])

                    if forward and self.check_reaction((forward_reaction_pattern, rindices, pindices)):
                        reactions[forward, rspecies].append(pspecies)
                    if reverse and self.check_reaction((reverse_reaction_pattern, rindices, pindices)):
                        reactions[reverse, pspecies].append(rspecies)
            kinetic_terms.append(reactions)

            for (rate, reactants), all_products in reactions.iteritems():
                for r in reactants:
                    equations[r].append('-%s' % (rate), *reactants)
                for products in all_products:
                    for p in products:
                        equations[p].append('+%s' % (rate), number(1, len(all_products)), *reactants)

        for v in equations.itervalues():
            v.set_parent(self)

        self._kinetic_equations = equations
        self._kinetic_terms = kinetic_terms
        return equations

    def demo(self):
        eqns0 = self.kinetic_equations
        print 'Model kinetic equations:'
        for sp in sorted(eqns0):
            print '  d%s/dt = %s' % (sp, eqns0[sp])

        print 'Pool relations:'
        #pool = self.pool_relations
        #for ispecies, specie in sorted (pool):
        #    print '  %s <- %s' % ('+'.join(ispecies), specie)

        for mspecie in sorted(self.pools):
            print '  %s = %s' % (mspecie, '+'.join(self.pools[mspecie]))

        print 'Applying pool relations:'
        eqns = self.apply(self.pools, self.kinetic_equations)
        for sp in sorted(eqns):
            print '  d%s/dt = %s' % (sp, eqns[sp])

        print 'Substituting pool relations to RHS..',
        for i in range(4):
            # repeate until all isotopomer species are resolved
            eqns = self.subs(eqns, self.pool_relations)
            eqns = self.collect(eqns)
        print 'done'
        
        print 'Mass isotopomer kinetic equations:'
        for sp in sorted(eqns):
            print '  d%s/dt = %s' % (sp, eqns[sp])

    @property
    def isotopomer_equations(self):
        verbose = self.verbose
        if verbose:
            print 'Generating isotopologue equations:'
        self.eqn_count = -1    
        eqns0 = self.kinetic_equations

        if verbose:
            print 'Performing first term collection.'
        eqns0 = self.collect(eqns0)

        if verbose:
            self.eqn_count = 0
            print 'Performing second term collection.'
        return self.collect(eqns0)
        
    @property
    def mass_isotopomer_equations(self):
        verbose = self.verbose
        if verbose:
            print 'Generating mass isotopologue equations:'
        self.eqn_count = -1
        pr = self.pool_relations
        pr_quad = self.pool_relations_quad
        if verbose:
            print 'Applying pool relations.'
        eqns = self.apply(self.pools, self.kinetic_equations)

        if verbose:
            print 'Performing first term collection.'
        eqns = self.subs(eqns, pr_quad)
        eqns = self.collect(eqns)

        if verbose:
            print 'Performing second term collection.'
        eqns = self.subs(eqns, pr)
        eqns = self.collect(eqns)

        if verbose:
            print 'Performing third term collection.'
        eqns = self.subs(eqns, pr)        
        eqns = self.collect(eqns)

        if verbose:
            print 'Performing final term collection.'
        eqns = self.subs(eqns, pr)
        self.eqn_count = 0
        return  self.collect(eqns)

    @property
    def pool_relations(self):
        if self._pool_relations is not None:
            return self._pool_relations
        relations = []
        d = defaultdict(set)
        for poolname, species in self.pools.iteritems ():
            d[symbol(poolname.prefix,'')].add(poolname)
            for s in species:
                d[symbol(s.prefix,'')].add(s)
            n = len(species) - len(poolname)/100.
            relations.append((n,species, poolname))
        for prefix in d:
            species = list(d[prefix])
            n = len(species) - len(prefix)/100.
            if prefix[-1]=='_':
                if self.replace_total_sum_with_one:
                    relations.append((n, species, number(1)))
            else:
                relations.append((n, species, prefix))
        relations.sort(reverse=True)
        relations = [item[1:] for item in relations if item[1] != [item[2]]]
        self._pool_relations = relations        
        return relations
    
    @property
    def pool_relations_quad(self):
        if self._pool_relations_quad is not None:
            return self._pool_relations_quad
        relations = []

        for lst, name in self.pool_relations:
            if len(lst)>1 and isinstance (name, str) and name.startswith('ADP'): # HACK
                l = [Factors(*sorted([a,b])) for a in lst for b in lst]
                relations.append((l, Factors(name, name)))
                l = [Factors(*sorted([number(2),a,b])) for a in lst for b in lst]
                relations.append((l, Factors(number(2), name, name)))

        return relations

    @property
    def relations(self):
        """
        Return a list of relations (<terms>, <name>) that can be used
        for simplifying the expressions in the system of kinetic
        equations.
        """
        if self._relations is not None:
            return self._relations
        relations = []
        d = defaultdict(set)
        for poolname, species in self.pools.iteritems ():
            for s in species:
                d[symbol(s.prefix,'')].add(s)
                if '_' in poolname.index:
                    t1, t2 = s.index.split('_')
                    d[symbol_latex(s.prefix+'_',t1+'_'+('X'*len (t2)))].add(s)

                    d[symbol_latex(s.prefix+'_',t1+'_'+str(t2.count('1')), skip_replace=[1])].add(s)
                    d[symbol_latex(s.prefix+'_',('X'*len (t1))+'_'+t2)].add(s)
                    d[symbol_latex(s.prefix+'_',str(t1.count('1'))+'_'+t2, skip_replace=[0])].add(s)
                    d[symbol_latex(s.prefix+'_',str(t1.count('1'))+'_'+('X'*len (t2)), skip_replace=[0])].add(s)
                    d[symbol_latex(s.prefix+'_',('X'*len (t1))+'_'+str(t2.count('1')), skip_replace=[1])].add(s)
            n = len(species) - len(poolname)/100.
            relations.append((n,species, poolname))

        for prefix in d:
            species = list(d[prefix])
            n = len(species) - len(prefix)/100.
            relations.append((n, species, prefix))
        relations.sort(reverse=True)
        relations = [item[1:] for item in relations if item[1] != [item[2]]]
        self._relations = relations        
        return relations

    @property
    def pools(self):
        if self._pools is not None:
            return self._pools
        pools = defaultdict (set)
        for s, indices in self.species.iteritems():
            if s is None:
                continue
            for i in indices:
                si = symbol_latex(s, i)
                pools[self.get_pool_name (si)].add(si)

        for poolname in pools:
            species = pools[poolname]
            for si in species:
                for sj in species:
                    assert self.check_equivalence(si, sj),`si,sj,poolname`
            pools[poolname] = sorted (species)
        self._pools = pools
        return pools

    @property
    def pools2(self):
        if self._pools2 is not None:
            return self._pools2
        pools = defaultdict (set)
        for s, indices in self.species.iteritems():
            if s is None:
                continue
            for i in indices:
                si = symbol_latex(s,i)
                pools[symbol(s)].add(si)

        for poolname in pools:
            species = pools[poolname]
            pools[poolname] = sorted (species)
        self._pools2 = pools
        return pools


    def apply(self, pools, kinetic_equations):
        new_kinetic_equations = defaultdict(Sum)
        for poolname, pool in pools.iteritems():
            for s in pool:
                new_kinetic_equations[poolname].extend(kinetic_equations[s])
        for poolname, eqns in new_kinetic_equations.iteritems():
            new_kinetic_equations[poolname] = eqns.normal()
        return new_kinetic_equations

    def show(self, obj, collect=False, stream=sys.stdout):

        if isinstance(obj, list):
            l = [(name, terms) for terms, name in obj]
            for name, terms in sorted(l):
                stream.write('%s = %s\n' % (StrContext.map(name), '+'.join(sorted(terms))))
        else:
            for k in sorted(obj):
                expr = obj[k]
                if collect:
                    expr = expr.collect()
                if StrContext.context=='latex':
                    stream.write('\\tfrac{d%s}{dt}=%s\n' % (StrContext.map(k), expr))
                else:
                    stream.write('d%s/dt=%s\n' % (StrContext.map(k), expr))

    def subs(self, kinetic_equations, relations, **options):
        new_kinetic_equations = {}
        for k in kinetic_equations:
            new_kinetic_equations[k] = kinetic_equations[k].subs(relations, **options)
        return new_kinetic_equations

    def collect(self, kinetic_equations, **options):
        verbose = self.verbose
        new_kinetic_equations = {}
        for k in kinetic_equations:
            #if verbose:
            #    if self.eqn_count != -1:
            #        self.eqn_count += 1
            #        print self.eqn_count, flush,
            #    else:
            #        print '.', flush,
            new_kinetic_equations[k] = kinetic_equations[k].collect(**options)
        return new_kinetic_equations
