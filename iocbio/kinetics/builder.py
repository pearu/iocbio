from __future__ import division

import os
import re
import sys
import itertools
import subprocess
from fractions import Fraction
from collections import defaultdict

from mytools.tools import drop_to_ipython as dti

from sympycore import Symbol, Calculus

from steadystate import SteadyFluxAnalyzer

import pprint

def pp(item):
    if isinstance(item, dict):
        item = dict(item)
    print pprint.pformat(item)

def pf(item):
    if isinstance(item, dict):
        item = dict(item)
    return pprint.pformat(item)

class Flush:
    def __str__ (self):
        sys.stdout.flush()
        return ''
flush = Flush()

make_indices = lambda repeat: map(''.join,itertools.product('01', repeat=repeat)) if repeat else ['']

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
            #r = None
            #if self.index:
            #    r = '\\%s%s' % (self.prefix, self.index.replace('1','x').replace('0','o').replace('_','I'))
            #
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

    #def __new__(cls, value, denom='1'):
    #    if str(denom)=='1':
    #        return symbol.__new__(cls, str(value), '')
    #    return symbol.__new__(cls, '%s/%s' % (value, denom), '')

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
            return sorted(self.data) == sorted(other.data)
        return False

    def __ne__(self, other):
        return not (self==other)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(self.data)))

    def append(self, term):
        if isinstance(term, Terms):
            self.data.extend(term.data)
        elif isinstance(term, number) and term == number(0):
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
        show_counters = [] #[55] #range (100)
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
                    common_terms[k] = l.copy()
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

class IsotopologueModel:
    """ Base class for generating kinetic equations for reactions with isotopologues.

    See Example below for usage.
    """
    species = {} 
    reactions = [] 
    latex_name_map = {}

    replace_total_sum_with_one = False

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._pools = None
        self._pools2 = None
        self._kinetic_equations = None
        self._kinetic_terms = None
        self._relations = None
        self._pool_relations = None
        self._total_one_relations = None
        self._pool_relations_quad = None

    def check_reaction(self, reaction_pattern, rindices, pindices):
        """
        Return True when reaction is possible for given tuples of
        reactant and product indices. Reaction pattern is a string in
        a form 'R1+R2(<-|->)P1-P2' where reactants R1, R2 and products
        P1, P2 are keys of the species dictionary.
        """
        raise NotImplementedError('check_reaction for reaction_pattern=%r' % (reaction_pattern))

    def check_equivalence(self, s1, s2):
        """
        Return True when species s1 and s2 are equivalent.
        """
        return False

    def get_pool_name(self, specie):
        """
        Return the name of a pool where a species belongs.
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
            forward_reaction_pattern = '%s->%s' % ('+'.join(reactants), '+'.join(products))
            reverse_reaction_pattern = '%s<-%s' % ('+'.join(reactants), '+'.join(products))
            forward = reaction.get('forward')
            reverse = reaction.get('reverse')
            
            reactant_indices = [self.species[s] for s in reaction['reactants']]
            product_indices = [self.species[s] for s in reaction['products']]

            reactions = defaultdict(list)
            for rindices in itertools.product(*map(self.species.__getitem__, reactants)):
                rspecies = tuple([symbol_latex(p,i) for p,i in zip(reactants, rindices)])
                for pindices in itertools.product(*map(self.species.__getitem__, products)):
                    pspecies = tuple([symbol_latex(p,i) for p,i in zip(products, pindices)])

                    if forward and self.check_reaction(forward_reaction_pattern, rindices, pindices):
                        reactions[forward, rspecies].append(pspecies)
                    if reverse and self.check_reaction(reverse_reaction_pattern, rindices, pindices):
                        #print reverse_reaction_pattern
                        reactions[reverse, pspecies].append(rspecies)
            kinetic_terms.append(reactions)

            for (rate, reactants), all_products in reactions.iteritems():
                for r in reactants:
                    equations[r].append('-%s' % (rate), *reactants)
                for products in all_products:
                    for p in products:
                        equations[p].append('+%s' % (rate), number(1, len(all_products)), *reactants)
        #pp(equations)

        for v in equations.itervalues():
            v.set_parent(self)

        self._kinetic_equations = equations
        self._kinetic_terms = kinetic_terms
        return equations

    @property
    def isotopomer_equations(self):
        eqns0 = self.kinetic_equations
        eqns0 = self.collect(eqns0)
        eqns0 = self.collect(eqns0)
        if self.replace_total_sum_with_one:
            eqns0 = self.subs(eqns0, self.total_one_relations)
        return eqns0
    
    @property
    def mass_isotopomer_equations(self):
        pr = self.pool_relations
        pr_quad = self.pool_relations_quad
        eqns = self.apply(self.pools, self.kinetic_equations)
        #pp (pr_quad)
        #pp (pr)
        eqns = self.subs(eqns, pr_quad)
        #pp (eqns)
        #sys.exit ()
        eqns = self.collect(eqns)

        eqns = self.subs(eqns, pr)
        eqns = self.collect(eqns)

        eqns = self.subs(eqns, pr)        
        eqns = self.collect(eqns)


        eqns = self.subs(eqns, pr)
        return  self.collect(eqns)

    @property
    def total_one_relations(self):
        if self._total_one_relations is not None:
            return self._total_one_relations
        #pp(self.species)
        pools = defaultdict (set)
        relations = []
        for s, indices in self.species.iteritems():
            if not self.replace_total_sum_with_one:
                break
            if s is None:
                continue
            l = []
            for i in indices:
                si = symbol_latex(s, i)
                l.append(si)
            relations.append((l,number(1)))
        self._total_one_relations = relations        
        return relations

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
            relations.append((n, species, poolname))
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
            if len (lst)>1 and isinstance (name, str) and name.startswith('ADP'): # HACK
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
            # relations
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
        new_kinetic_equations = {}
        for k in kinetic_equations:
            new_kinetic_equations[k] = kinetic_equations[k].collect(**options)
        return new_kinetic_equations

class IsotopologueModelBuilder(IsotopologueModel):

    def __init__(self, system=None, system_name=None, labeled_species=None, options=dict()):
        
        IsotopologueModel.__init__(self)

        if not isinstance(options, dict):
            try:
                options = dict(options)
            except ValueError, msg:
                raise ValueError('Cannot create options dictionary.  dict(options) failed with {0}'.format(msg))

        required_options = dict(use_mass=False,
                                use_sum=False,
                                import_dir='generated',
                                add_boundary_fluxes=False,
                                discard_boundary_species=True,
                                replace_total_sum_with_one=False)
        
        for k, v in required_options.items():
            if not options.has_key(k):
                options[k] = v
        self.options = options

        labeled_tag = ''.join(sorted(labeled_species.keys()))
        model_name = 'model_' + system_name + labeled_tag
        if options['use_mass']:
            model_name += '_mass'
        if options['use_sum']:
            model_name += '_sum'
        if options['replace_total_sum_with_one']:
            model_name += '_repl1'

        print 'labeled_species : ', labeled_species
        print 'System name     : ', system_name    
        print 'Model name      : ', model_name

        self.flux_analyzer = a = SteadyFluxAnalyzer(system,
                                                    add_boundary_fluxes=options['add_boundary_fluxes'],
                                                    discard_boundary_species=options['discard_boundary_species'],
                                                    split_bidirectional_fluxes=False)
        self.model_name = model_name
        self.system_name = system_name
        self.labeled_species = labeled_species
        self.system_str = a.system_string
        
        print 'Steady-state system:'
        mat = a.label_matrix(a.stoichiometry, a.species, a.reactions)
        print mat.__str__(max_nrows=20, max_ncols=21)
        self.reactions = a.reactions_info.values()
        reaction_info = a.species_info

        self.metabolite_lengths = reaction_info.pop('metabolite_lengths')
        #print self.metabolite_lengths
        #print a.discarded_species
        #print labeled_species
        self.reaction_pairs = reaction_info

        index_dic = {}
        for met, length in self.metabolite_lengths.items():
            index_dic[met] = make_indices(length)

        self.species = {}
        lnm = dict()
        for sp in a.species + a.discarded_species:
            self.species[sp] = index_dic[sp]
            lnm[sp] = r'\{0}name'.format(sp) + '{}'
        
        self.latex_name_map = lnm
        
        self.c_name_map = {}
        for k in self.latex_name_map.keys():
            self.c_name_map[k] = k

        self._make_options_attributes()


    def _make_options_attributes(self):
        for k, v in self.options.items():
            setattr(self, k, v)

    @property
    def labeled_isotopologues(self):
        k = []
        for met_key, it_code_dic in self.labeled_species.items():
            for it_code in it_code_dic.keys():
                k.append(met_key + it_code)
        return k

    @property
    def system_jacobian(self):
        ie = self.isotopomer_equations
        rxns = self.reactions
        
        symbols = []
        for it_key in ie.keys():
            symbols.append(Symbol(it_key))
        for rxn in rxns:
            symbols.append(Symbol(rxn['forward']))
            symbols.append(Symbol(rxn['reverse']))

        for s in symbols:
            st = s.__str__()
            locals()[st] = s # HACK

        data = {}
        for eqn, eq in ie.items():
            inner_dic = defaultdict(dict)
            exec('ceq = Calculus({0})'.format(eq.__str__()))
            for it_key in ie.keys():
                inner_dic[it_key] = ceq.diff(it_key)
            data[eqn] = dict(inner_dic)

        # Remove labeled isotopologues from jacobian.
        for it_key in self.labeled_isotopologues:
            del data[it_key]
        for k, vd in data.items():
            for it_key in self.labeled_isotopologues:
                del vd[it_key]
        return data
    
    def check_equivalence(self, s1, s2):
        if s1.prefix != s2.prefix:
            return False
        if s1==s2:
            return True
        return False

    def check_reaction(self, reaction_pattern, rindices, pindices):
        #print reaction_pattern, rindices, pindices

        met_lists = reaction_pattern.split('->')
        if len(met_lists) < 2:
            met_lists = reaction_pattern.split('<-')
        assert len(met_lists) == 2, `met_lists`
    
        str_reactants = met_lists[0]
        str_products = met_lists[1]
        if '+' in str_reactants:
            reactants = str_reactants.split('+')
        else:
            reactants = [str_reactants]
            
        if '+' in str_products:
            products = str_products.split('+')
        else:
            products = [str_products]

        assert len(reactants) == len(rindices), `(reactants, rindices)`
        assert len(products) == len(pindices), `(products, pindices)`

        for rindex, rpat in enumerate(rindices):
            reactant = reactants[rindex]
            rlen = self.metabolite_lengths[reactant]
            assert rlen == len(rpat), `rlen, rpat`

        for pindex, ppat in enumerate(pindices):
            product = products[pindex]
            plen = self.metabolite_lengths[product]
            assert plen == len(ppat), `plen, ppat`

        if len(products) == 1 and len(reactants) == 1 and reactants[0] == products[0]:
            return rindices == pindices

        rlcount = 0
        for ri in rindices:
            for i in ri:
                if i == '1':
                    rlcount += 1
        plcount = 0
        for pi in pindices:
            for i in pi:
                if i == '1':
                    plcount += 1

        if rlcount != plcount: return False

        rxn_ok = True
        for rindex, reactant in enumerate(reactants):
            rpat = rindices[rindex]
            len_r = self.metabolite_lengths[reactant]
            r_mappings = self.reaction_pairs[reactant]
            for r_mapping in r_mappings:
                assert len(r_mapping.keys()) == 1, `r_mapping`
                p_key, atom_dic = r_mapping.items()[0]
                for pindex, product in enumerate(products):
                    if product == p_key:
                        ppat = pindices[pindex]
                        for ra, pa in atom_dic.items():
                            if rpat[ra - 1] != ppat[pa -1]:
                                return False
        return rxn_ok

        return IsotopeModel.check_reaction(self, reaction_pattern, rindices, pindices)

    def write_pyf(self, stream=sys.stdout, package_name='c_package'):

        if not hasattr(self, 'c_variables'):
            msg = "The routine write_ccode must be called before write_pyf since this routine defines c_variables."
            raise UserWarning(msg)
        else:
            c_variables = self.c_variables
        
        indent = ' '*6
        
        in_list = str()
        for variable in c_variables.keys():
            if variable == 'out':
                continue
            in_list += '%s, ' %(variable)
        in_list = in_list[:-2]
        var_list = in_list + ', out'
        
        pyf_header = '''
python module %s
  interface
    subroutine c_equations(%s)\n''' %(package_name, var_list)

        pyf_header = pyf_header[:-2] + ')\n%sintent(c)\n%sintent(c) c_equations\n\n' %(indent, indent)

        for variable, var_list in c_variables.items():
            if variable == 'solver_time' and var_list is None:
                length = 1
            else:
                length = len(var_list)
            pyf_header += '%sreal*8 %s (%s)\n' %(indent, variable, length)
        pyf_header += '\n%sintent(out) out\n%sintent(in) %s\n' %(indent, indent, in_list)
        pyf_header += '''
    end subroutine c_equations
  end interface
end python module %s\n\n''' % (package_name)

        stream.write(pyf_header)

    def write_ccode(self, stream=sys.stdout):
        use_sum = self.options['use_sum']
        use_mass = self.options['use_mass']
        it_eqns = self.isotopomer_equations
        #pp(it_eqns)
        eqns = it_eqns #self.mass_isotopomer_equations
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
            # r = ATPi0_0
            while not r[0].isdigit ():
                r = r[1:]
            return sum(map(int,r.split ('_')))==0
        #is_unlabeled = lambda r: sum(map(int,r.split ('_')[1:]))==0

        if use_mass:
            c_eqns = eqns
        else:
            c_eqns = it_eqns

        with StrContext ('ccode', self.c_name_map):
            stream.write('/*\n')
            for line in self.system_str.splitlines():
                if line.strip() and not line.strip().startswith('#'):
                    stream.write('{0}\n'.format(line))            
            stream.write('*/\n' + c_header)

            # Write input labeling species with values.
            labeled_it_keys = []
            for s_key, l_dic in self.labeled_species.items():
                for l_key, value in l_dic.items():
                    it_key = '{0}{1}'.format(s_key, l_key)
                    labeled_it_keys.append(it_key)
                    stream.write('double {0} = {1} ;\n'.format(it_key, value))

            # Assign the input_list to their isotopomer names.
            initial_value_list = []
            derivatives_list = []
            index = 0
            for k in sorted (c_eqns):
                it_key = StrContext.map(k)

                if it_key in labeled_it_keys:
                    continue
                elif use_sum and is_unlabeled (k):
                    count = -1

                    for k1 in c_eqns:
                        if get_name (k1)==get_name (k):
                            #if k1.startswith(k.split ('_')[0]):
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
            for met in self.metabolite_lengths.keys():
                if met in self.labeled_species:
                    continue
                pool_set.add(StrContext.map(met))
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
                if k in labeled_it_keys:
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

    def make_terms(self):
        td = []
        for r in self._kinetic_terms:
            for (k, reactants), vl in r.items():
                rs = ''
                for reactant in reactants:
                    #rs += perform_latex_substitution(reactant) + '+'
                    rs += reactant.__str__() + '+'
                rs = rs[:-1]
                for products in vl:
                    ps = ''
                    for product in products:
                        #ps += perform_latex_substitution(product) + '+'
                        ps += product.__str__() + '+'
                    ps = ps[:-1]
                    td.append((k, rs, ps))
        return sorted(td)

    def make_kinetic_terms(self):
        terms = sorted(self.make_terms())

        bi_dic = dict()
        for t in terms:
            k = t[0][1:]
            v = t[0][0]
            l = bi_dic.get(k, [])
            l.append(v)
            bi_dic[k] = l

        b_dic = dict()
        for k, vl in bi_dic.items():
            fcount = 0
            rcount = 0
            for v in vl:
                if v == 'f':
                    fcount += 1
                elif v == 'r':
                    rcount += 1
            if rcount == 0:
                b_dic[k] = False
            else:
                assert fcount == rcount, `fcount == rcount`
                b_dic[k] = True

        #print b_dic

        s = r'\subsection{Individual isotopic transformations}\label{sec:isotopomer_reactions}' + '\n' + r'\begin{align*}'
        lnm = self.latex_name_map2
        for index, t in enumerate(terms):
            wrap = index%4
            k = t[0][1:]
            v = t[0][0]
            if b_dic[k]:
                es = '<=>'
            else:
                es = '->'
            if v == 'r':
                continue
            if wrap == 0:
                if index != 0:
                    eq = r'\\' + '\n'
                else:
                    eq = '\n'
                eq += r'\cee{' + ' {0} &{1}C[${2}$] {3} &'.format(t[1], es, lnm[t[0]], t[2])
            if wrap == 1 or wrap == 2:
                eq += ' {0} &{1}C[${2}$] {3} & '.format(t[1], es, lnm[t[0]], t[2])
            if wrap == 3:
                #if index == len(terms)-1:
                eq += ' {0} &{1}C[${2}$] {3} }} \n'.format(t[1], es, lnm[t[0]], t[2])
                #else:
                #    eq += ' {0} &->C[{1}] {2} \\\\ \n'.format(t[1], t[0], t[2])
                s += eq
        s +=  r'\end{align*}' + '\n'
        return s
       
                
    def compile_ccode(self, debug=False, stage=None):
        build_dir = self.options['import_dir']
        mn = self.model_name
        file_names = dict(c='%s/%s.c' % (build_dir, mn),
                          pyf='%s/%s.pyf' % (build_dir, mn),
                          c_variables='%s/%s_c_variables.py' % (build_dir, mn),
                          so='%s.so' % (mn),
                          )
        if not debug:
            print 'Writing',file_names['c']
            f = open(file_names['c'], 'w')
            self.write_ccode(f)
            f.close()
        else:
            self.write_ccode()

        if stage==1:
            return

        if not debug:
            f = open(file_names['pyf'], 'w')
            self.write_pyf(f, package_name=mn)
            f.close()
        else:
            self.write_pyf(package_name=mn)

        cv_string = pf(self.c_variables)
        if not debug:
            print 'Writing',file_names['c_variables']
            f = open(file_names['c_variables'], 'w')
            f.write('c_variables = %s\n' %(cv_string))
            f.close()
        else:
            print cv_string

        if debug:
            return True
        
        Popen_kwds = dict(shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=os.environ)

        commands = ["f2py -c %s %s" %(file_names['pyf'], file_names['c']),
                    "cp %s %s/" %(file_names['so'], build_dir),
                    "rm %s" %(file_names['so']),
                    ]
        for cmd in commands:
            print "Executing command: %s" %(cmd)
            cmd_proc = subprocess.Popen(cmd, **Popen_kwds)
            cmd_out, cmd_err = cmd_proc.communicate()
            if cmd_err.strip() != str():
                msg = "Error executing command. %s" %(cmd)
                print cmd_out
                print cmd_err
                raise SystemError(msg)#, cmd_out, cmd_err)




class Example(IsotopologueModel):

    species = dict(
        T = ['00', '01', '10', '11'],
        C = ['000', '001', '010', '100', '011', '110', '101', '111'],
        #A = ['001', '010', '100','000'],
        A = ['00','01','10','11'],
        B = ['0','1']
        #B = ['01', '10'],
        #C = ['0001','0010','0100','1000'],
        #D= ['1'],
        )
    reactions = [dict(forward='f', reactants=['T','B'], products=['C'],
                      reverse='r')]
    #reactions = [dict(reactants=['A'], products=['B', 'B'], forward='f')]

    def check_reaction(self, reaction_pattern, rindices, pindices):
        if reaction_pattern in ['T+B->C','T+B<-C']:
            return ''.join(rindices).count('1') == ''.join(pindices).count('1')
        if reaction_pattern in ['A->B+B']:
            return ''.join(rindices).count('1') == ''.join(pindices).count('1')
        return IsotopeModel.check_reaction(self, reaction_pattern, rindices, pindices)

    def check_equivalence(self, s1, s2):
        if s1.prefix==s2.prefix:
            if s1.prefix in 'ABCDCT':
                return s1.index.count('1') == s2.index.count('1')
        return False

if __name__ == '__main__':
    

    ex = Example()
    ex.show(ex.relations)
    print ex.show(ex.subs(ex.kinetic_equations, ex.relations))
