

from __future__ import division
import sys
import re
import itertools
from fractions import Fraction
from collections import defaultdict

import pprint
pp = pprint.pprint
def pp (item):
    if isinstance (item, dict):
        item = dict (item)
    print pprint.pformat (item)

class Flush:
    def __str__ (self):
        sys.stdout.flush ()
        return ''
flush = Flush ()

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

class IsotopeModel:
    """ Base class for generating kinetic equations for reactions with
    isotopomers.

    See Example below for usage.
    """
    species = {} # {A = ['1']}
    reactions = [] # [dict(forward='f', reactants=['A'], products=['B'])]
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
        #pp (equations)

        for v in equations.itervalues():
            v.set_parent(self)

        self._kinetic_equations = equations
        self._kinetic_terms = kinetic_terms
        return equations

    @property
    def isotopomer_equations(self):
        eqns0 = self.kinetic_equations
        eqns0 = self.collect(eqns0)
        return self.collect(eqns0)
        
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

class Example(IsotopeModel):

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
            return ''.join(rindices).count('1')==''.join(pindices).count('1')
        if reaction_pattern in ['A->B+B']:
            return ''.join(rindices).count('1')==''.join(pindices).count('1')
        return IsotopeModel.check_reaction(self, reaction_pattern, rindices, pindices)

    def check_equivalence(self, s1, s2):
        if s1.prefix==s2.prefix:
            if s1.prefix in 'ABCDCT':
                return s1.index.count('1')==s2.index.count('1')
        return False

if __name__ == '__main__':
    

    ex = Example()
    ex.show(ex.relations)
    print ex.show(ex.subs(ex.kinetic_equations, ex.relations))
