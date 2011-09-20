"""
Generate code for evaluating correlation functions.

"""

import sys
from sympycore import Symbol, Calculus, PolynomialRing, Expr, heads

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
                              pwf = pwf, R=R)

    def integrate(self, integrand='f(x)*f(x+y)'):
        """ Integrate integrand of piecewise polynomial functions
        within bounds [0, N-1-y] where N is the number of functions
        nodal values, y=j+r, 0<=r<1, j is integer in [0, N-1].

        Parameters
        ----------
        integrand : str
          Specify integrand as string expression. The expression can contain
          substrings `f(x)`, `f(x+y)`, `x` that are treated specially.
          The expression may not contain substring `i`.

        Returns
        -------
        integral_i, integral_r : PolynomialRing['s','r']
          Integral is sum(integral_i, i=0..N-3-j) + integral_r
          where integral_(i|r) is a polynomial of r.
        """
        for k,v in self.namespace.iteritems():
            exec k+' = v'

        integrand1 = eval(integrand.replace('f(x)','pwf(f,i,s)').replace('f(x+y)','pwf(f,i+j,s+r)').replace('x','(R.Number(i)+s)'))
        integrand2 = eval(integrand.replace('f(x)','pwf(f,N-2-j, s)').replace('f(x+y)','pwf(f,N-2,s+r)').replace('x','(R.Number(N-2-j)+s)'))
        integrand3 = eval(integrand.replace('f(x)','pwf(f,i,s)').replace('f(x+y)','pwf(f,i+j+1,s+r-1)').replace('x','(R.Number(i)+s)'))

        integral1 = integrand1.variable_integrate(s, 0, 1-r)
        integral2 = integrand2.variable_integrate(s, 0, 1-r)
        integral3 = integrand3.variable_integrate(s, 1-r, 1)
        integral_i1 = (integral1 + integral3).expand()
        integral_r = (integral2).expand()

        integral_i = R({})
        for e,c in integral_i1.data.iteritems():
            if isinstance(c, Expr):
                c = c.head.to_ADD(type(c), c.data, c)
                if c.head is heads.ADD:
                    data = c.data
                else:
                    data = [c]
            else:
                data = [c]
            for c in data:
                if 'i' in str(c):
                    integral_i += R({e:c})
                else:
                    integral_r += R({e:c*(N-2-j)})

        return integral_i, integral_r

    def show_convolution(self):
        poly_i, poly_r = self.integrate('f(x)*f(x+y)')
        for k in sorted(set(poly_i.data.keys() + poly_r.data.keys())):
            expr_i = poly_i.data.get(k, 0)
            expr_r = poly_r.data.get(k, 0)
                
            print '%s: sum(%s,i=0..N-3-j) + {%s}' % (self.ring({k:1}), expr_i, expr_r)

g = Generator('linear')
g.show_convolution()



