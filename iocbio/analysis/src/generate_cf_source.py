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

class Generator:
    def __init__(self, pwf):
        """
        Parameters
        ----------
        pwf : {callable, 'constant', 'linear', 'catmulrom'}
          Specify piecewise polynomial function pwf(f, i, s) where f
          denotes a sequence of nodal values, i denotes i-th piece and
          s denotes local variable of the polynomial. For example,
          use `pwf = lambda f,i,s: f[i] + s*(f[i+1]-f[i])` for piecewise
          linear function. Note that when pwf is evaluated, f will be
          instance of IndexGenerator, i will be Symbol and s will be
          instance of PolynomialRing['s', 'r'].
        """
        if isinstance(pwf, str):
            if pwf=='constant':
                pwf = lambda f,i,s: f[i]
            elif pwf=='linear':
                pwf = lambda f,i,s: f[i] + s*(f[i+1]-f[i])
            else:
                raise NotImplementedError(`pwf`)
        self.pwf = pwf
        self.ring = R = PolynomialRing[('s','r')]
        self.namespace = dict(s = R('s'), r=R('r'), i=Symbol('i'),
                              j = Symbol('j'), N=Symbol('N'),
                              f = IndexedGenerator(R, 'f'),
                              pwf = pwf)

    def convolution(self):
        for k,v in self.namespace.iteritems():
            exec k+' = v'

        # int(f(x)*f(x+y), x=0..N-1)
        integrand1 = pwf(f,i,s)*pwf(f,i+j,s+r)+pwf(f,N-2-j, s)*pwf(f,N-2,s+r)
        integrand2 = pwf(f,i,s)*pwf(f,i+j+1,s-(1-r))
        integral1 = integrand1.variable_integrate(s, 0, 1-r)
        integral2 = integrand2.variable_integrate(s, 1-r, 1)
        integral_fx_fxy = (integral1 + integral2).expand() # sum(intergal_fx_fxy, i=0..N-2-j)
    
        return integral_fx_fxy

    def show_convolution(self):
        poly = self.convolution()
        for k in sorted(poly.data):
            expr = poly.data[k]
            expr = expr.head.to_ADD(type(expr), expr.data, expr)
            assert str(expr.head)=='ADD',`expr.head`
            loop_terms = []
            nonloop_terms = []
            for term in expr.data:
                if 'i' in str(term):
                    loop_terms.append(term)
                else:
                    nonloop_terms.append(term)
                
            print '%s: %s + {%s}' % (self.ring({k:1}), self.ring.ring.Add(*loop_terms), self.ring.ring.Add(*nonloop_terms))

g = Generator('linear')
g.show_convolution()



