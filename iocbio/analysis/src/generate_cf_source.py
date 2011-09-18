"""
Generate code for evaluating correlation functions.

"""

import sys
from sympycore import Symbol, Calculus, PolynomialRing

class Indexed:
    def __init__(self, name, index):
        self.name = name
        self.index = index
    def __str__(self):
        return '%s[%s]' % (self.name, self.index)
    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.name, self.index)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name==other.name and self.index==other.index
        return False
    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self.index))

class IndexedGenerator:
    def __init__(self, ring, name):
        self.ring = ring
        self.name = name
    def __getitem__(self, index):
        return self.ring.Number(Symbol(Indexed(self.name, index)))

R = PolynomialRing[('s','r')]

s = R('s')
r = R('r')


#r = Symbol('r')
i = Symbol('i')
j = Symbol('j')
N = Symbol('N')
f = IndexedGenerator(R, 'f')

def pwf(i, s): # piecewise f
    return f[i] + s*(f[i+1]-f[i])


integrand1 = pwf(i,s)*pwf(i+j,s+r)+pwf(N-2-j, s)*pwf(N-2,s+r)
integrand2 = pwf(i,s)*pwf(i+j+1,s-(1-r))

undefintegral1 = integrand1.variable_integrate(s, 0, 1-r)
undefintegral2 = integrand2.variable_integrate(s, 1-r, 1)
print undefintegral1
print undefintegral2

sys.exit()
integral1 = undefintegral1.variable_subs(s,1-r) - undefintegral1.variable_subs(s,0)
integral2 = undefintegral2.variable_subs(s,1) - undefintegral2.variable_subs(s,1-r)
for exps, coeff in integral1.data.iteritems():
    print exps, coeff.expand()



