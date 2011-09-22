"""
Generate code for evaluating correlation functions.

"""
# Author: Pearu Peterson
# Created: September 2011

import re
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
        #return self.ring.Number(index)
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
            elif pwf=='qint':
                pwf = lambda f,i,s: f[i-1]*(s-1)*s/2 + f[i]*(1-s*s) + f[i+1]*(1+s)*s/2
            elif pwf=='cint':
                pwf = lambda f,i,s: (f[i-1]*s*((2-s)*s-1) + f[i]*(2+s*s*(3*s-5)) + f[i+1]*s*((4-3*s)*s+1) + f[i+2]*s*s*(s-1))/2
            else:
                raise NotImplementedError(`pwf`)
        self.pwf = pwf
        self.ring = R = PolynomialRing[('s','r')]
        self.namespace = dict(s = R('s'), r=R('r'), i=Symbol('i'),
                              o=Symbol('o'),
                              j = Symbol('j'), N=Symbol('n'),
                              f = IndexedGenerator(R, 'f'),
                              pwf = pwf, R=R)

    def integrate(self, integrand='f(x)*f(x+y)', extension='cutoff'):
        """ Integrate integrand of piecewise polynomial functions
        within bounds [0, N-1-y] where N is the number of functions
        nodal values, y=j+r, 0<=r<1, j is integer in [0, N-1].

        Parameters
        ----------
        integrand : str
          Specify integrand as string expression. The expression can
          contain substrings `f(x)`, `f(x+y)`, `x` that are treated
          specially.  The expression may not contain substrings `i`,
          `s` and `r`.
          
        extension : {'cutoff', 'periodic'}
          Specify f(x) extension outside its support interval. For
          cutoff extension ``f(x)=0`` and for periodic
          ``f(x)=f(x-(N-1))``, when ``x > N-1``.

        Returns
        -------
        integral_i, integral_r : PolynomialRing['s','r']
          Integral is ``sum(integral_i, i=0..N-3-j) + integral_r``
          where integral_(i|r) is a polynomial of r.
        """
        for k,v in self.namespace.iteritems():
            exec k+' = v'

        if extension=='cutoff':
            integrand1 = eval(integrand.replace('f(x)','pwf(f,o+i,s)').replace('f(x+y)','pwf(f,o+i+j,s+r)').replace('x','(R.Number(i)+s)').replace('f(0)','f[o]'))
            integrand2 = eval(integrand.replace('f(x)','pwf(f,o+N-2-j, s)').replace('f(x+y)','pwf(f,o+N-2,s+r)').replace('x','(R.Number(N-2-j)+s)').replace('f(0)','f[o]'))
            integrand3 = eval(integrand.replace('f(x)','pwf(f,o+i,s)').replace('f(x+y)','pwf(f,o+i+j+1,s+r-1)').replace('x','(R.Number(i)+s)').replace('f(0)','f[o]'))

            integral1 = integrand1.variable_integrate(s, 0, 1-r)
            integral2 = integrand2.variable_integrate(s, 0, 1-r)
            integral3 = integrand3.variable_integrate(s, 1-r, 1)
            integral_i1 = (integral1 + integral3).expand()
            integral_r = (integral2).expand()
        elif extension=='periodic' or 1:
            raise NotImplementedError(`extension`)

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

    def show_convolution(self, integrand='f(x)*f(x+y)'):
        poly_i, poly_r = self.integrate(integrand)

        for k in sorted(set(poly_i.data.keys() + poly_r.data.keys())):
            expr_i = poly_i.data.get(k, 0)
            expr_r = poly_r.data.get(k, 0)
                
            print '%s: sum(%s,i=0..N-3-j) + {%s}' % (self.ring({k:1}), expr_i, expr_r)

    def generate_source(self,
                        name = 'mcf1',
                        integrand = '(f(x)-f(0))*(2*f(x+y)-f(x)-f(0))',
                        extension='cutoff'):
        #self.show_convolution(integrand)
        poly_i, poly_r = self.integrate(integrand)
        exps = sorted(set(poly_i.data.keys() + poly_r.data.keys()))

        decl_coeffs = ', '.join('double* a%s' % (i) for i in range(len(exps)))
        init_coeffs = '\n  '.join('double b%s = 0.0;' % (i) for i in range(len(exps)))
        set_coeffs = '\n  '.join('*a%s = b%s;' % (i, i) for i in range(len(exps)))

        cf_source_template = '''
%(cf_proto)s
{
  /* %(cf_def)s */
  int p, i, q;
  int k = n - 2 - j;
  %(init_coeffs)s
  if (k>=0)
  {
    for(p=0, o=0; p<m; ++p, o+=n)
    {
      for (i=0; i<k; ++i)
      {
        %(update_loop_coeffs)s
      }
      %(update_nonloop_coeffs)s
    }
  }
  %(set_coeffs)s
}
        '''

        for order in range(4):
            poly_i_diff = poly_i.variable_diff(self.namespace['r'], order)
            poly_r_diff = poly_r.variable_diff(self.namespace['r'], order)
            diff_exps = sorted(set(poly_i_diff.data.keys() + poly_r_diff.data.keys()))

            update_loop_coeffs = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, 0)) for e in diff_exps)
            update_nonloop_coeffs = '\n      '.join('b%s += %s;' % (e[0], poly_r_diff.data.get(e, 0)) for e in diff_exps)
            cf_proto = 'void cf_%(name)s_compute_coeffs_diff%(order)s(int j, double *f, int n, int m, %(decl_coeffs)s)' % (locals())

            if order:
                cf_def = 'diff(int(%s, x=0..L-y), y, order=%s) = sum(a_k*r^k, k=0..%s) where y=j+r' % (integrand, order, len(exps)-1)
            else:
                cf_def = 'int(%s, x=0..L-y) = sum(a_k*r^k, k=0..%s) where y=j+r' % (integrand, len(exps)-1)
            cf_def += '\n     f(x)=sum([0<=s<1]*(%s), i=0..N-1) where s=x-i' % (eval('pwf(f,i,s)', self.namespace).evalf())

            cf_source = cf_source_template % (locals())
            cf_source = re.sub(r'(\(f\[(?P<index>[^\]]+)\]\)[*]{2,2}2)', r'(f[\g<index>]*f[\g<index>])', cf_source)
            cf_source = re.sub(r'(\(f\[(?P<index>[^\]]+)\]\)[*]{2,2}(?P<exp>\d+))', r'pow(f[\g<index>], \g<exp>)', cf_source)
            cf_source = re.sub(r'(?P<numer>\d+)[/](?P<denom>\d+)', r'\g<numer>.0/\g<denom>.0', cf_source)
            yield cf_proto, cf_source

g = Generator('qint')
for proto, source in g.generate_source('mcf1_pw1'):
    print source
    pass




