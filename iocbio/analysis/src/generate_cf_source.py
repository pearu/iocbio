"""
Generate code for evaluating correlation functions.

"""
# Author: Pearu Peterson
# Created: September 2011
#from __future__ import division
import re
import os
import sys

from sympycore import Symbol, Calculus, PolynomialRing, Expr, heads


indexed_str = 'direct'
class Indexed:
    def __init__(self, name, index, indexed_subs):
        self.name = name
        self.index = index
        self.indexed_subs = indexed_subs
    def __str__(self):
        global indexed_str
        if indexed_str=='direct':
            return '%s[%s]' % (self.name, str(self.index).replace(' ',''))
        elif indexed_str=='macro':
            return '%s(%s)' % (self.name.upper(), str(self.index).replace(' ',''))
        raise NotImplementedError(`indexed_str`)
    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.name, self.index)
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name==other.name and self.index==other.index
        return False
    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self.index))

class IndexedGenerator:
    def __init__(self, ring, name, indexed_subs):
        self.ring = ring
        self.name = name
        self.indexed_subs = indexed_subs
    def __getitem__(self, index):
        index = Indexed(self.name, index, self.indexed_subs)
        return self.ring.Number(Symbol(index))

def str_replace(text, repl_list):
    for k, v in repl_list:
        text = text.replace (k, v)
    return text

class Generator:
    def __init__(self, pwf, extension='cutoff'):
        """
        Parameters
        ----------
        pwf : {'constant', 'linear', 'qint', 'cint'}
          Specify piecewise polynomial function pwf(f, i, s) where f
          denotes a sequence of nodal values, i denotes i-th piece and
          s denotes local variable of the polynomial. For example,
          use `pwf = lambda f,i,s: f[i] + s*(f[i+1]-f[i])` for piecewise
          linear function. Note that when pwf is evaluated, f will be
          instance of IndexGenerator, i will be Symbol and s will be
          instance of PolynomialRing['s', 'r'].
        """
        offsets = 0,0
        if isinstance(pwf, str):
            if pwf=='constant':
                pwf1 = pwf2 = lambda f,i,s: f[i]
            elif pwf=='linear':
                pwf1 = pwf2 = lambda f,i,s: f[i] + s*(f[i+1]-f[i])
                if 0:
                    pwf = lambda f,i,s,d=1: f[i-d] + s*(f[i+1+d]-f[i-d])
                    offsets = 1,1
            elif pwf=='linear_constant':
                pwf1 = lambda f,i,s: f[i] + s*(f[i+1]-f[i])
                pwf2 = lambda f,i,s: f[i]
            elif pwf=='constant_linear':
                pwf1 = lambda f,i,s: f[i]
                pwf2 = lambda f,i,s: f[i] + s*(f[i+1]-f[i])
            elif pwf=='qint':
                pwf1 = pwf2 = lambda f,i,s: f[i-1]*(s-1)*s/2 + f[i]*(1-s*s) + f[i+1]*(1+s)*s/2
                #offsets = 1,1
            elif pwf=='cint':
                pwf1 = pwf2 = lambda f,i,s: (f[i-1]*s*((2-s)*s-1) + f[i]*(2+s*s*(3*s-5)) + f[i+1]*s*((4-3*s)*s+1) + f[i+2]*s*s*(s-1))/2
                #offsets = 1,2
            else:
                raise NotImplementedError(`pwf`)
        self.offsets = offsets
        self.ring = R = PolynomialRing[('s','r')]
        if extension=='cutoff':
            def indexed_subs(expr, *subs_args):
                assert isinstance (expr, Calculus) and expr.head is heads.SYMBOL,`expr.pair`
                index = expr.data.index.subs(*subs_args)
                index1 = index #.subs('o', 0)
                if 1 and isinstance(index1, Calculus) and index1.head is heads.NUMBER:
                    if index1.data < 0:
                        return Calculus(0)
                return Symbol(Indexed(expr.data.name, index, expr.data.indexed_subs))
        else:
            raise NotImplementedError (`extension`)
        self.extension = extension

        self.namespace = dict(s = R('s'), r=R('r'), i=Symbol('i'),
                              o=Symbol('o'),
                              j = Symbol('j'), N=Symbol('n'),
                              f = IndexedGenerator(R, 'f', indexed_subs),
                              pwf1 = pwf1,
                              pwf2 = pwf2,
                              R=R)

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
            integrand1 = eval(str_replace(integrand,[
                        ('f(x)','pwf1(f,i,s)'),
                        ('f(x+y)','pwf1(f,i+j,s+r)'),
                        ('f1(x)','pwf1(f,i,s)'),
                        ('f1(x+y)','pwf1(f,i+j,s+r)'),
                        ('f2(x)','pwf2(f,i,s)'),
                        ('f2(x+y)','pwf2(f,i+j,s+r)'),
                        ('x','(R.Number(i)+s)')])
                              )
            integrand2 = eval(str_replace(integrand,[
                        ('f(x)','pwf1(f,N-2-j, s)'),
                        ('f(x+y)','pwf1(f,N-2,s+r)'),
                        ('f1(x)','pwf1(f,N-2-j, s)'),
                        ('f1(x+y)','pwf1(f,N-2,s+r)'),
                        ('f2(x)','pwf2(f,N-2-j, s)'),
                        ('f2(x+y)','pwf2(f,N-2,s+r)'),
                        ('x','(R.Number(N-2-j)+s)')]))
            integrand3 = eval(str_replace(integrand,[
                        ('f(x)','pwf1(f,i,s)'),
                        ('f(x+y)','pwf1(f,i+j+1,s+r-1)'),
                        ('f1(x)','pwf1(f,i,s)'),
                        ('f1(x+y)','pwf1(f,i+j+1,s+r-1)'),
                        ('f2(x)','pwf2(f,i,s)'),
                        ('f2(x+y)','pwf2(f,i+j+1,s+r-1)'),
                        ('x','(R.Number(i)+s)')]))

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
                        extension='cutoff',
                        max_diff_order=3):
        global indexed_str
        #self.show_convolution(integrand)
        poly_i, poly_r = self.integrate(integrand)
        exps = sorted(set(poly_i.data.keys() + poly_r.data.keys()))
        poly_order = max([e[0] for e in exps])
        coeffs = ', '.join('a%s' % (i) for i in range(poly_order+1))
        refcoeffs = ', '.join('&a%s' % (i) for i in range(poly_order+1))
        decl_coeffs = ', '.join('double* a%s' % (i) for i in range(poly_order+1))
        init_coeffs_ref = '\n  '.join('double a%s = 0.0;' % (i) for i in range(poly_order+1))
        init_coeffs = '\n  '.join('double b%s = 0.0;' % (i) for i in range(poly_order+1))
        set_coeffs = '\n  '.join('*a%s = b%s;' % (i, i) for i in range(poly_order+1))
        set_coeffs0 = '\n      '.join('*a%s = 0.0;' % (i,) for i in range(poly_order+1))

        cf_source_template = '''
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?0.0:((I)>=n?0.0:f[(I)]))

%(cf_proto)s
{
  /* %(cf_def)s */
  int p, i;
  int k = n - 2 - j;
  int k0 = (%(start_offset)s>k ? k : %(start_offset)s);
  int k1 = k-%(end_offset)s;
  double *f = fm;
  %(init_coeffs)s
  if (k>=0)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      i = 0;
      //printf("k0=%%d, k=%%d, k1=%%d\\n", k0, k, k1);
      for (;i<k0; ++i)
      {
      //printf("i0=%%d\\n", i);
        %(update_loop_coeffs_start)s
      }
      for (; i<k1; ++i)
      {
        //printf("i1=%%d\\n", i);
        %(update_loop_coeffs)s
      }
      for (; i<k; ++i)
      {
        //printf("i2=%%d\\n", i);
        %(update_loop_coeffs_end)s
      }
      %(update_nonloop_coeffs)s
    }
  }
  %(set_coeffs)s
}
        '''

        start_offset, end_offset = self.offsets
        order_cases = []
        for order in range(poly_order+1):
            poly_i_diff = poly_i.variable_diff(self.namespace['r'], order)
            poly_r_diff = poly_r.variable_diff(self.namespace['r'], order)
            diff_exps = sorted(set(poly_i_diff.data.keys() + poly_r_diff.data.keys()))
            
            indexed_str = 'macro'
            update_loop_coeffs_start = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, 0)) for e in diff_exps)
            update_loop_coeffs_end = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, 0)) for e in diff_exps)
            update_nonloop_coeffs = '\n      '.join('b%s += %s;' % (e[0], poly_r_diff.data.get(e, 0)) for e in diff_exps)
            #indexed_str = 'direct'
            update_loop_coeffs = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, 0)) for e in diff_exps)


            cf_proto = 'void cf_%(name)s_compute_coeffs_diff%(order)s(int j, double *fm, int n, int m, %(decl_coeffs)s)' % (locals())

            if order:
                cf_def = 'diff(int(%s, x=0..L-y), y, order=%s) = sum(a_k*r^k, k=0..%s) where y=j+r' % (integrand, order, poly_order)
            else:
                cf_def = 'int(%s, x=0..L-y) = sum(a_k*r^k, k=0..%s) where y=j+r' % (integrand, poly_order)
            cf_def += '\n     f1(x)=sum([0<=s<1]*(%s), i=0..N-1) where s=x-i' % (eval('pwf1(f,i,s)', self.namespace).evalf())
            cf_def += '\n     f2(x)=sum([0<=s<1]*(%s), i=0..N-1) where s=x-i' % (eval('pwf2(f,i,s)', self.namespace).evalf())

            order_cases.append('case %(order)s: cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(coeffs)s); break;' % (locals()))

            cf_source = cf_source_template % (locals())
            cf_source = re.sub(r'(\(f\[(?P<index>[^\]]+)\]\)[*]{2,2}2)', r'(f[\g<index>]*f[\g<index>])', cf_source)
            cf_source = re.sub(r'(\(f\[(?P<index>[^\]]+)\]\)[*]{2,2}(?P<exp>\d+))', r'pow(f[\g<index>], \g<exp>)', cf_source)
            cf_source = re.sub(r'(?P<numer>\d+)[/](?P<denom>\d+)', r'\g<numer>.0/\g<denom>.0', cf_source)
            yield cf_proto, cf_source, ''

        cf_proto = 'void cf_%(name)s_compute_coeffs(int j, double *fm, int n, int m, int order, %(decl_coeffs)s)' % (locals())
        order_cases = '\n    '.join(order_cases)
        cf_def = 'diff(int(%s, x=0..L-y), y, order) = sum(a_k*r^k, k=0..%s) where y=j+r' % (integrand, poly_order)
        cf_def += '\n     f1(x)=sum([0<=s<1]*(%s), i=0..N-1) where s=x-i' % (eval('pwf1(f,i,s)', self.namespace).evalf())
        cf_def += '\n     f2(x)=sum([0<=s<1]*(%s), i=0..N-1) where s=x-i' % (eval('pwf2(f,i,s)', self.namespace).evalf())
        cf_source_template = '''
%(cf_proto)s
{
  /* %(cf_def)s */
  switch (order)
  {
    %(order_cases)s
    default:
      %(set_coeffs0)s
  }
}
        '''
        yield cf_proto, cf_source_template % (locals()), ''

        cf_proto = 'double cf_%(name)s_evaluate(double y, double *fm, int n, int m, int order)' % (locals())
        horner = 'a%s' % (poly_order)
        for e in reversed(range(poly_order)):
            horner = 'a%s+(%s)*r' % (e, horner)
        cf_source_template = '''
%(cf_proto)s
{
  %(init_coeffs_ref)s
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  cf_%(name)s_compute_coeffs(j, fm, n, m, order, %(refcoeffs)s);
  return %(horner)s;
}
        '''
        pyf_template = '''
    function %(name)s_evaluate(y, f, n, m, order) result (value)
       intent(c) %(name)s_evaluate
       fortranname cf_%(name)s_evaluate
       double precision intent(in, c) :: y
       double precision dimension (m, n), intent(in,c):: f
       integer, depend(f), intent(c,hide) :: n = (shape(f,1)==1?shape (f,0):shape(f,1))
       integer, depend(f), intent(c,hide) :: m = (shape(f,1)==1?1:shape(f,0))
       integer intent(c), optional :: order = 0
       double precision :: value
    end function  %(name)s_evaluate
        '''
        
        yield cf_proto, cf_source_template % (locals()), pyf_template % (locals())

        cf_proto = 'double cf_%(name)s_find_piecewise_linear_zero(int j0, int j1, double *fm, int n, int m)' % (locals())
        order = poly_order-1
        cf_source_template = '''
%(cf_proto)s
{
  %(init_coeffs_ref)s
  int j;
  for (j=j0; j<j1; ++j)
  {
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs)s);  
  }
  return 0;
}
        '''

        pyf_template = '''
    subroutine %(name)s_find_piecewise_linear_zero(f, n, m)
       intent(c) %(name)s_find_piecewise_linear_zero
       fortranname cf_%(name)s_find_piecewise_linear_zero
       double precision dimension (m, n), intent(in,c):: f
       integer, depend(f), intent(c,hide) :: n = (shape(f,1)==1?shape (f,0):shape(f,1))
       integer, depend(f), intent(c,hide) :: m = (shape(f,1)==1?1:shape(f,0))
    end subroutine  %(name)s_find_piecewise_linear_zero
        '''
        
        yield cf_proto, cf_source_template % (locals()), pyf_template % (locals())

        cf_proto = 'double cf_%(name)s_find_piecewise_quadratic_zero(int j0, int j1, double *fm, int n, int m)' % (locals())
        order = poly_order-2
        cf_source_template = '''
%(cf_proto)s
{
  %(init_coeffs_ref)s
  int j;
  for (j=j0; j<j1; ++j)
  {
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs)s);  
  }
  return 0;
}
        '''
        yield cf_proto, cf_source_template % (locals()), ''
        
def generate():
    this_file = __file__
    source_name = os.path.join(os.path.dirname(this_file), 'cf.c')
    header_name = os.path.join(os.path.dirname(this_file), 'cf.h')
    pyf_name = os.path.join(os.path.dirname(this_file), 'cf.pyf')

    print 'Creating files:'
    print '\t', header_name
    print '\t', source_name
    print '\t', pyf_name
    source_file = open(source_name, 'w')
    header_file = open(header_name, 'w')
    pyf_file = open(pyf_name, 'w')

    header_header = '''
/* This file is generated using %(this_file)s.

  Author: Pearu Peterson
  Created: Oct 2011
*/
#ifndef CF_H
#define CF_H

#ifdef __cplusplus
extern "C" {
#endif
''' % (locals())
    header_footer = '''
#ifdef __cplusplus
}
#endif
#endif
'''

    source_header = '''
/* This file is generated using %(this_file)s.

  Author: Pearu Peterson
  Created: Oct 2011
*/
#include <math.h>
#include <stdio.h>
#include "cf.h"
    ''' % (locals())
    source_footer = '''
'''
    pyf_header = '''

python module cf
  interface
    '''
    pyf_footer = '''
  end interface
end python module
    '''
    header_file.write(header_header)
    source_file.write(source_header)
    pyf_file.write(pyf_header)

    for name, (pwf, integrand) in dict(
        a11 = ('linear', 'f1(x)*f2(x+y)'),
        a10 = ('linear_constant', 'f1(x)*f2(x+y)'),
        a22 = ('qint', 'f1(x)*f2(x+y)'),
        a33 = ('cint', 'f1(x)*f2(x+y)'),
        ).iteritems():
        g = Generator(pwf)
        for proto, source, interface in g.generate_source(name,
                                               integrand=integrand):
            source_file.write(source)
            header_file.write('extern %s;\n' % (proto))
            if interface:
                pyf_file.write(interface)

    header_file.write(header_footer)
    source_file.write(source_footer)
    pyf_file.write(pyf_footer)

    source_file.close()
    header_file.close()
    pyf_file.close()

if __name__=='__main__':
    generate()


