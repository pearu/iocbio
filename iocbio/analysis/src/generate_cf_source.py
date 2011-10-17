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
indexed_map = {}
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
        elif indexed_str=='variable':
            v = '%s_%s' % (self.name, str(self.index).replace(' ','').replace('+','p').replace('-','m'))
            e = '%s(%s)' % (self.name.upper(), str(self.index).replace(' ',''))
            indexed_map[v] = e
            return v
        elif indexed_str=='latex':
            return '%s_{%s}' % (self.name, self.index)
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
                if 1:
                    pwf1 = pwf2 = lambda f,i,s: f[i-1]*(s-1)*s/2 + f[i]*(1-s*s) + f[i+1]*(1+s)*s/2
                    offsets = 1,1
                elif 1:
                    pwf1 = pwf2 = lambda f,i,s: f[i-2]*4*(s-s**2)/3+f[i-1]*(1-10*s+9*s*s)/6+f[i]*(4+4*s-7*s**2)/6+f[i+1]*(1+2*s+s**2)/6+f[i+2]*s**2/6
                    offsets = 2,2
                elif 1:
                    pwf1 = pwf2 = lambda f,i,s: f[i-2]*(-s+s*s)/12+f[i-1]*2*(s-s*s)/3+f[i]*(1-s*s)+f[i+1]*(-2*s+5*s*s)/3+f[i+2]*(s-s*s)/12
                    #f[-2]*(-1/12*s+1/12*s^2)+f[-1]*(2/3*s-2/3*s^2)+f[0]*(1-s^2)+f[1]*(-2/3*s+5/3*s^2)+f[2]*(1/12*s-1/12*s^2)
                    offsets = 2,2
            elif pwf=='cint':
                if 1:
                    # catmull-rom
                    pwf1 = pwf2 = lambda f,i,s: (f[i-1]*s*((2-s)*s-1) + f[i]*(2+s*s*(3*s-5)) + f[i+1]*s*((4-3*s)*s+1) + f[i+2]*s*s*(s-1))/2
                    offsets = 1,2
                elif 1:
                    pwf1 = pwf2 = lambda f,i,s: f[i-2]*(-s+2*s*s-s*s*s)/12 + f[i-1]*(2*s/3-5*s*s/4+7*s**3/12)+f[i]*(1-11*s*s/3+8*s**3/3)+f[i+1]*(-2*s/3+13*s**2/3-8*s**3/3)+f[i+2]*(s/12+s**2/2-7*s**3/12)+f[i+3]*(-s**2/12+s**3/12)
                    offsets = 2,3
                elif 1:
                    pwf1 = pwf2 = lambda f,i,s:f[i]*(1-3*s*s+2*s*s*s)+f[i+1]*(3*s*s-2*s*s*s)
                    offsets = 0,0
            else:
                raise NotImplementedError(`pwf`)
        else:
            raise NotImplementedError(`pwf`)
        self.offsets = offsets
        self.ring = R = PolynomialRing[('s','r')]
        if extension=='cutoff':
            def indexed_subs(expr, *subs_args):
                assert isinstance (expr, Calculus) and expr.head is heads.SYMBOL,`expr.pair`
                index = expr.data.index.subs(*subs_args)
                if isinstance(index, Calculus) and index.head is heads.NUMBER:
                    if index.data < 0:
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
                        ('f1(0)','f[0]'),
                        ('f2(0)','f[0]'),
                        ('f1(L)','f[N-1]'),
                        ('f2(L)','f[N-1]'),
                        ('f(x)','pwf1(f,i,s)'),
                        ('f(x+y)','pwf1(f,i+j,s+r)'),
                        ('f1(x)','pwf1(f,i,s)'),
                        ('f1(x+y)','pwf1(f,i+j,s+r)'),
                        ('f2(x)','pwf2(f,i,s)'),
                        ('f2(x+y)','pwf2(f,i+j,s+r)'),
                        ('x','(R.Number(i)+s)')])
                              )
            integrand2 = eval(str_replace(integrand,[
                        ('f1(0)','f[0]'),
                        ('f2(0)','f[0]'),
                        ('f1(L)','f[N-1]'),
                        ('f2(L)','f[N-1]'),
                        ('f(x)','pwf1(f,N-2-j, s)'),
                        ('f(x+y)','pwf1(f,N-2,s+r)'),
                        ('f1(x)','pwf1(f,N-2-j, s)'),
                        ('f1(x+y)','pwf1(f,N-2,s+r)'),
                        ('f2(x)','pwf2(f,N-2-j, s)'),
                        ('f2(x+y)','pwf2(f,N-2,s+r)'),
                        ('x','(R.Number(N-2-j)+s)')]))
            integrand3 = eval(str_replace(integrand,[
                        ('f1(0)','f[0]'),
                        ('f2(0)','f[0]'),
                        ('f1(L)','f[N-1]'),
                        ('f2(L)','f[N-1]'),
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

    def generate_find_real_zero_in_01_source(self):
        codes = {2:\
'''
/* Code translated from http://www.netlib.org/toms/493, subroutine QUAD, with modifications. */
#define ABS(X) ((X)<0.0?-(X):(X))
double b, e, d, lr, sr;
//printf("a_0,a_1,a_2, e=%f, %f, %f\\n", a_0, a_1, a_2);
if (a_2==0.0)
  {
    if (a_1!=0.0) return -a_0/a_1;
    return -1.0;
  }
else
  {
    if (a_0==0.0)
      return 0.0;
    b = a_1*0.5;
    if (ABS(b) < ABS(a_0))
    {
      e = a_2;
      if (a_0<0.0)
        e = -a_2;
      e = b*(b/ABS(a_0)) - e;
      d = sqrt(ABS(e))*sqrt(ABS(a_0));
    }
    else
    {
      e = 1.0 - (a_2/b)*(a_0/b);
      d = sqrt(ABS(e))*ABS(b);
    }
    if (e>=0)
    {
      if (b>=0.0) d=-d;
      lr = (-b+d)/a_2;
      if (lr==0.0)
        return 0.0;
      sr = (a_0/lr)/a_2;
      //printf("p(lr=%f)=%f\\n", lr,a_0+lr*(a_1+lr*a_2));
      //printf("p(sr=%f)=%f\\n", sr,a_0+sr*(a_1+sr*a_2));
      if (lr>=0 && lr<=1.0)
        return lr;
      return sr;
    }
  }
'''
                 }
        
        for poly_order in [2]:
            decl_coeffs = ', '.join('double a_%s' % (i) for i in range(poly_order+1))
            cf_proto = 'double cf_find_real_zero_in_01_%(poly_order)s(%(decl_coeffs)s)' % (locals ())
            code = codes.get(poly_order, 'printf("Not implemented: %s\\n");' % (cf_proto))
            cf_source_template = '''
%(cf_proto)s
{
  %(code)s
  return -1.0;
}
            '''
            yield cf_proto, cf_source_template %(locals ()), ''

    def generate_approximation_source(self):
        # the following coefficient tables are computed with the following maple program:
        """
        max_order:=7:
        ff:=(i,s)->sum(a[i,k]*s^k,k=0..max_order):
        p:=s->p0+p1*s+p2*s^2:
        f:=int((ff(1,1+s)-p(s))^2,s=-1..0)+int((ff(2,s)-p(s))^2,s=0..1)+int((ff(3,s-1)-p(s))^2,s=1..2);
        sol:=solve({diff(f,p0), diff(f,p1), diff(f,p2)}, {p0,p1,p2});
        'a[i]=['evalf(coeff(subs(sol,p0),a[i,j]),16)'$j=0..7]'$i=1..3; # where pp=p0,p1,p2
        """
        p0_coeffs_table3 = dict (
            a1 = [.4320987654320988, .2592592592592593, .1851851851851852, .1438271604938272, .1174603174603175, 0.9920634920634921e-1, 0.8583186360964139e-1, 0.7561728395061728e-1], 
            a2 = [.5802469135802469, .2716049382716049, .1728395061728395, .1253086419753086, 0.9770723104056437e-1, 0.7980599647266314e-1, 0.6731334509112287e-1, 0.5812757201646091e-1], 
            a3 = [-0.1234567901234568e-1, -0.8641975308641975e-1, -0.8641975308641975e-1, -0.7839506172839506e-1, -0.7019400352733686e-1, -0.6305114638447972e-1, -0.5702527924750147e-1, -0.5195473251028807e-1]
            )            
        p1_coeffs_table3 = dict (
            a1 = [-.6913580246913580, -.1851851851851852, -0.7407407407407407e-1, -0.3456790123456790e-1, -0.1693121693121693e-1, -0.7936507936507936e-2, -0.2939447383891828e-2, 0.], 
            a2 = [.4938271604938272, .2839506172839506, .1975308641975309, .1506172839506173, .1213403880070547, .1014109347442681, 0.8700764256319812e-1, 0.7613168724279835e-1], 
            a3 = [.1975308641975309, 0.1234567901234568e-1, -0.2469135802469136e-1, -0.3456790123456790e-1, -0.3668430335097002e-1, -0.3615520282186949e-1, -0.3468547912992357e-1, -0.3292181069958848e-1]
            )
        p2_coeffs_table3 = dict (
            a1 = [.2469135802469136, 0., -0.3703703703703704e-1, -0.4320987654320988e-1, -0.4232804232804233e-1, -0.3968253968253968e-1, -0.3674309229864785e-1, -0.3395061728395062e-1], 
            a2 = [-.4938271604938272, -.2469135802469136, -.1604938271604938, -.1172839506172840, -0.9171075837742504e-1, -0.7495590828924162e-1, -0.6319811875367431e-1, -0.5452674897119342e-1],
            a3 = [.2469135802469136, .2469135802469136, .2098765432098765, .1790123456790123, .1552028218694885, .1366843033509700, .1219870664315109, .1100823045267490]
            )

        l0_coeffs_table3 = dict (
            a1 = [.5555555555555556, .2592592592592593, .1666666666666667, .1222222222222222, 0.9629629629629630e-1, 0.7936507936507936e-1, 0.6746031746031746e-1, 0.5864197530864198e-1], 
            a2 = [.3333333333333333, .1481481481481481, 0.9259259259259259e-1, 0.6666666666666667e-1, 0.5185185185185185e-1, 0.4232804232804233e-1, 0.3571428571428571e-1, 0.3086419753086420e-1],
            a3 = [.1111111111111111, 0.3703703703703704e-1, 0.1851851851851852e-1, 0.1111111111111111e-1, 0.7407407407407407e-2, 0.5291005291005291e-2, 0.3968253968253968e-2, 0.3086419753086420e-2]
            )
        l1_coeffs_table3 = dict (
            a1 = [-.4444444444444444, -.1851851851851852, -.1111111111111111, -0.7777777777777778e-1, -0.5925925925925926e-1, -0.4761904761904762e-1, -0.3968253968253968e-1, -0.3395061728395062e-1],
            a2 = [0., 0.3703703703703704e-1, 0.3703703703703704e-1, 0.3333333333333333e-1, 0.2962962962962963e-1, 0.2645502645502646e-1, 0.2380952380952381e-1, 0.2160493827160494e-1],
            a3 = [.4444444444444444, .2592592592592593, .1851851851851852, .1444444444444444, .1185185185185185, .1005291005291005, 0.8730158730158730e-1, 0.7716049382716049e-1]
            )

        p0_coeffs_table1 = dict (
            a1 = [1., 0., 0., 0.5000000000000000e-1, 0.8571428571428571e-1, .1071428571428571, .1190476190476190, .1250000000000000]
            )
        p1_coeffs_table1 = dict (
            a1 = [0., 1., 0., -.6000000000000000, -.9142857142857143, -1.071428571428571, -1.142857142857143, -1.166666666666667]
            )
        p2_coeffs_table1 = dict (
            a1 = [0., 0., 1., 1.500000000000000, 1.714285714285714, 1.785714285714286, 1.785714285714286, 1.750000000000000]
            )

        l0_coeffs_table1 = dict (
            a1 = [1., 0., -.1666666666666667, -.2000000000000000, -.2000000000000000, -.1904761904761905, -.1785714285714286, -.1666666666666667],
            )
        l1_coeffs_table1 = dict (
            a1 = [0., 1., 1., .9000000000000000, .8000000000000000, .7142857142857143, .6428571428571429, .5833333333333333]
            )

        for poly_order in range (8):
            decl_coeffs1 = ', '.join('double a1_%s' % (i) for i in range(poly_order+1))
            decl_coeffs2 = ', '.join('double a2_%s' % (i) for i in range(poly_order+1))
            decl_coeffs3 = ', '.join('double a3_%s' % (i) for i in range(poly_order+1))

            cf_proto = 'void cf_quadratic_approximation_3_%(poly_order)s(%(decl_coeffs1)s, %(decl_coeffs2)s, %(decl_coeffs3)s, double* p0, double* p1, double* p2)' % (locals ())

            p0 = '+'.join(['%.16e*%s_%s' % (p0_coeffs_table3[a][i],a,i) for a in ['a1','a2', 'a3'] for i in range(poly_order+1)])
            p1 = '+'.join(['%.16e*%s_%s' % (p1_coeffs_table3[a][i],a,i) for a in ['a1','a2', 'a3'] for i in range(poly_order+1)])
            p2 = '+'.join(['%.16e*%s_%s' % (p2_coeffs_table3[a][i],a,i) for a in ['a1','a2', 'a3'] for i in range(poly_order+1)])

            cf_source_template = '''
%(cf_proto)s
{
  *p0 = %(p0)s;
  *p1 = %(p1)s;
  *p2 = %(p2)s;
}
            '''
            yield cf_proto, cf_source_template %(locals ()), ''            

            cf_proto = 'void cf_quadratic_approximation_1_%(poly_order)s(%(decl_coeffs1)s, double* p0, double* p1, double* p2)' % (locals ())

            p0 = '+'.join(['%.16e*%s_%s' % (p0_coeffs_table1[a][i],a,i) for a in ['a1'] for i in range(poly_order+1)])
            p1 = '+'.join(['%.16e*%s_%s' % (p1_coeffs_table1[a][i],a,i) for a in ['a1'] for i in range(poly_order+1)])
            p2 = '+'.join(['%.16e*%s_%s' % (p2_coeffs_table1[a][i],a,i) for a in ['a1'] for i in range(poly_order+1)])

            cf_source_template = '''
%(cf_proto)s
{
  *p0 = %(p0)s;
  *p1 = %(p1)s;
  *p2 = %(p2)s;
}
            '''
            yield cf_proto, cf_source_template %(locals ()), ''

            cf_proto = 'void cf_linear_approximation_3_%(poly_order)s(%(decl_coeffs1)s, %(decl_coeffs2)s, %(decl_coeffs3)s, double* p0, double* p1)' % (locals ())

            p0 = '+'.join(['%.16e*%s_%s' % (l0_coeffs_table3[a][i],a,i) for a in ['a1','a2', 'a3'] for i in range(poly_order+1)])
            p1 = '+'.join(['%.16e*%s_%s' % (l1_coeffs_table3[a][i],a,i) for a in ['a1','a2', 'a3'] for i in range(poly_order+1)])

            cf_source_template = '''
%(cf_proto)s
{
  *p0 = %(p0)s;
  *p1 = %(p1)s;
}
            '''
            yield cf_proto, cf_source_template %(locals ()), ''            
            
            cf_proto = 'void cf_linear_approximation_1_%(poly_order)s(%(decl_coeffs1)s, double* p0, double* p1)' % (locals ())

            p0 = '+'.join(['%.16e*%s_%s' % (l0_coeffs_table1[a][i],a,i) for a in ['a1'] for i in range(poly_order+1)])
            p1 = '+'.join(['%.16e*%s_%s' % (l1_coeffs_table1[a][i],a,i) for a in ['a1'] for i in range(poly_order+1)])

            cf_source_template = '''
%(cf_proto)s
{
  *p0 = %(p0)s;
  *p1 = %(p1)s;
}
            '''
            yield cf_proto, cf_source_template %(locals ()), ''            

    def generate_source(self,
                        name = 'mcf1',
                        integrand = '(f(x)-f(0))*(2*f(x+y)-f(x)-f(0))',
                        extension='cutoff',
                        max_diff_order=3):
        global indexed_str, indexed_map
        #self.show_convolution(integrand)
        poly_i, poly_r = self.integrate(integrand)
        exps = sorted(set(poly_i.data.keys() + poly_r.data.keys()))
        poly_order = max([e[0] for e in exps])
        if 1 or name=='e00':
            indexed_str = 'latex'
            print 'name=',name
            if name[0]=='a':
                print '\\acf_f(y)=',
            else:
                print '\\dn_f(y)=',
            print '+'.join(['%s_%s(\\flo{y}) %s%s' % (name[0],e[0], ('\\rem{y}' if e[0] else ''), ('^%s' % (e[0]) if e[0]>1 else '')) for e in exps])
            for e in exps:
                print '%s_%s(\\flo{y})=' % (name[0],e[0]), ('\sum_{i=0}^{N-3-j}%s'%(poly_i.data[e])).replace ('(','').replace (')','').replace ('**','^').replace ('*','').replace (' ',''),
                print '\\\\\n+',('%s'%(poly_r.data[e])).replace ('(','').replace (')','').replace ('**','^').replace ('*','').replace (' ','')

            print '-'*10
        coeffs = ', '.join('a%s' % (i) for i in range(poly_order+1))
        coeffs1 = ', '.join('a1_%s' % (i) for i in range(poly_order+1))
        coeffs2 = ', '.join('a2_%s' % (i) for i in range(poly_order+1))
        coeffs3 = ', '.join('a3_%s' % (i) for i in range(poly_order+1))
        refcoeffs = ', '.join('&a%s' % (i) for i in range(poly_order+1))
        refcoeffs1 = ', '.join('&a1_%s' % (i) for i in range(poly_order+1))
        refcoeffs2 = ', '.join('&a2_%s' % (i) for i in range(poly_order+1))
        refcoeffs3 = ', '.join('&a3_%s' % (i) for i in range(poly_order+1))
        decl_coeffs = ', '.join('double* a%s' % (i) for i in range(poly_order+1))
        init_coeffs_ref = '\n  '.join('double a%s = 0.0;' % (i) for i in range(poly_order+1))
        init_coeffs_ref1 = '\n  '.join('double a1_%s = 0.0;' % (i) for i in range(poly_order+1))
        init_coeffs_ref2 = '\n  '.join('double a2_%s = 0.0;' % (i) for i in range(poly_order+1))
        init_coeffs_ref3 = '\n  '.join('double a3_%s = 0.0;' % (i) for i in range(poly_order+1))
        set_coeffs_ref21 = '\n    '.join('a1_%s = a2_%s;' % (i,i) for i in range(poly_order+1))
        set_coeffs_ref32 = '\n    '.join('a2_%s = a3_%s;' % (i,i) for i in range(poly_order+1))
        init_coeffs = '\n  '.join('double b%s = 0.0;' % (i) for i in range(poly_order+1))
        set_coeffs = '\n  '.join('*a%s = b%s;' % (i, i) for i in range(poly_order+1))
        set_coeffs0 = '\n      '.join('*a%s = 0.0;' % (i,) for i in range(poly_order+1))

        cf_source_template = '''
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

%(cf_proto)s
{
  /* %(cf_def)s */
  int p, i;
  int k = n - 3 - j;
  double *f = fm;
  %(init_coeffs)s
  %(decl_vars)s
  if (j>=0 && j<=n-2)
  {
    for(p=0; p<m; ++p, f+=n)
    {
      %(init_vars)s
      for(i=0;i<=k;++i)
      {
        %(init_vars_i)s
        %(update_loop_coeffs)s
      }
      %(update_nonloop_coeffs)s
    }
  }
  %(set_coeffs)s
}
        '''

        start_offset, end_offset = self.offsets
        order_cases = []
        order_cases_extreme = []
        order_cases_zero = []
        for order in range(poly_order+1):
            poly_i_diff = poly_i.variable_diff(self.namespace['r'], order)
            poly_r_diff = poly_r.variable_diff(self.namespace['r'], order)
            diff_exps = sorted(set(poly_i_diff.data.keys() + poly_r_diff.data.keys()))
            indexed_map.clear()
            indexed_str = 'variable'
            update_nonloop_coeffs = '\n      '.join('b%s += %s;' % (e[0], poly_r_diff.data.get(e, Calculus(0)).evalf(16)) for e in diff_exps)
            #update_loop_coeffs_macro = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, Calculus(0)).evalf()) for e in diff_exps)
            update_loop_coeffs = '\n        '.join('b%s += %s;' % (e[0], poly_i_diff.data.get(e, Calculus(0)).evalf(16)) for e in diff_exps)
            indexed_str = 'macro'

            decl_vars = 'double ' + ', '.join(indexed_map) + ';'
            init_vars = '\n      '.join(['%s = %s;' % (v,e) for v,e in indexed_map.iteritems() if 'i' not in v])
            init_vars_i = '\n        '.join(['%s = %s;' % (v,e) for v,e in indexed_map.iteritems() if 'i' in v])

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
            cf_source = re.sub(r'(f_(?P<index>[\w\d]+)[*]{2,2}2)', r'(f_\g<index>*f_\g<index>)', cf_source)
            cf_source = re.sub(r'(\(f\[(?P<index>[^\]]+)\]\)[*]{2,2}(?P<exp>\d+))', r'pow(f[\g<index>], \g<exp>)', cf_source)
            cf_source = re.sub(r'(?P<numer>\d+)[/](?P<denom>\d+)', r'\g<numer>.0/\g<denom>.0', cf_source)
            yield cf_proto, cf_source, ''

            cf_proto2 = 'int cf_%(name)s_find_extreme_diff%(order)s(int j0, int j1, double *fm, int n, int m, double* result)' % (locals())
            poly_order2 = poly_order-order
            coeffs1_2 = ','.join(coeffs1.split(',')[:poly_order2+1])
            coeffs2_2 = ','.join(coeffs2.split(',')[:poly_order2+1])
            coeffs3_2 = ','.join(coeffs3.split(',')[:poly_order2+1])
            order_cases_extreme.append('case %(order)s: return cf_%(name)s_find_extreme_diff%(order)s(j0, j1, fm, n, m, result);' % (locals()))
            cf_source_template2_3 = '''
%(cf_proto2)s
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  %(init_coeffs_ref1)s
  %(init_coeffs_ref2)s
  %(init_coeffs_ref3)s
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    %(set_coeffs_ref21)s
    %(set_coeffs_ref32)s
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs3)s);
    count ++;
    if (count<3)
      continue;
    cf_quadratic_approximation_3_%(poly_order2)s(%(coeffs1_2)s, %(coeffs2_2)s, %(coeffs3_2)s, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            '''

            cf_source_template2_1 = '''
%(cf_proto2)s
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  %(init_coeffs_ref1)s
  int start_j = (j0>0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs1)s);
    cf_quadratic_approximation_1_%(poly_order2)s(%(coeffs1_2)s, &p0, &p1, &p2);
    if (p2!=0.0)
    {
       s = -0.5*p1/p2;
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            '''
            
            if poly_order2<2:
                yield cf_proto2, cf_source_template2_3 %(locals ()), ''
            else:
                yield cf_proto2, cf_source_template2_1 %(locals ()), ''

            cf_proto2 = 'int cf_%(name)s_find_zero_diff%(order)s(int j0, int j1, double *fm, int n, int m, double* result)' % (locals())
            poly_order2 = poly_order-order
            coeffs1_2 = ','.join(coeffs1.split(',')[:poly_order2+1])
            coeffs2_2 = ','.join(coeffs2.split(',')[:poly_order2+1])
            coeffs3_2 = ','.join(coeffs3.split(',')[:poly_order2+1])
            order_cases_zero.append('case %(order)s: return cf_%(name)s_find_zero_diff%(order)s(j0, j1, fm, n, m, result);' % (locals()))
            cf_source_template2_3 = '''
%(cf_proto2)s
{
  int count = 0;
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  %(init_coeffs_ref1)s
  %(init_coeffs_ref2)s
  %(init_coeffs_ref3)s
  int start_j = (j0>0?j0-1:0);
  int end_j = (j1<n-1?j1+1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    %(set_coeffs_ref21)s
    %(set_coeffs_ref32)s
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs3)s);
    count ++;
    if (count<3)
      continue;
    cf_linear_approximation_3_%(poly_order2)s(%(coeffs1_2)s, %(coeffs2_2)s, %(coeffs3_2)s, &p0, &p1);
    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_%(name)s_find_zero_diff%(order)s: j=%%d, p0=%%f, p1=%%f, s=%%f\\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j-1) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            '''

            cf_source_template2_1 = '''
%(cf_proto2)s
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  %(init_coeffs_ref1)s
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs1)s);
    cf_linear_approximation_1_%(poly_order2)s(%(coeffs1_2)s, &p0, &p1);

    if (p1!=0.0)
    {
       s = -p0/p1;
       //printf("cf_%(name)s_find_zero_diff%(order)s: j=%%d, p0=%%f, p1=%%f, s=%%f\\n",j,p0,p1, s);
       if (s>=0.0 && s<=1.0)
         {
            *result = (double) (j) + s;
            status = 0;
            break;
         }
    }
  }
  return status;
}
            '''

            cf_source_template2_d = '''
%(cf_proto2)s
{
  int j;
  double s;
  double p0 = 0.0;
  double p1 = 0.0;
  %(init_coeffs_ref1)s
  int start_j = (j0>=0?j0:0);
  int end_j = (j1<n?j1:n-1);
  int status = -1;
  for (j=start_j; j<end_j; ++j)
  {
    cf_%(name)s_compute_coeffs_diff%(order)s(j, fm, n, m, %(refcoeffs1)s);
    s = cf_find_real_zero_in_01_%(poly_order2)s(%(coeffs1_2)s);
    //printf("j,s=%%d, %%f\\n",j,s);
    if (s>=0.0 && s<=1.0)
      {
        *result = (double) (j) + s;
        status = 0;
        break;
      }
  }
  return status;
}
            '''

            if poly_order2==2 and order==1:
                yield cf_proto2, cf_source_template2_d %(locals ()), ''
            elif poly_order2<1:
                yield cf_proto2, cf_source_template2_3 %(locals ()), ''
            else:
                yield cf_proto2, cf_source_template2_1 %(locals ()), ''

            

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

        cf_proto = 'int cf_%(name)s_find_extreme(int j0, int j1, double *fm, int n, int m, int order, double* result)' % (locals())
        order_cases = '\n    '.join(order_cases_extreme)
        cf_source_template = '''
%(cf_proto)s
{
  switch (order)
  {
    %(order_cases)s
    default:
      *result = 0.0;
  }
  return -2;
}
        '''
        pyf_template = '''
    function %(name)s_find_extreme(j0, j1, f, n, m, order, res) result (status)
       intent(c) %(name)s_find_extreme
       fortranname cf_%(name)s_find_extreme
       integer intent(in, c) :: j0, j1
       double precision dimension (m, n), intent(in,c):: f
       integer, depend(f), intent(c,hide) :: n = (shape(f,1)==1?shape (f,0):shape(f,1))
       integer, depend(f), intent(c,hide) :: m = (shape(f,1)==1?1:shape(f,0))
       integer intent(c), optional :: order = 0
       double precision, intent(out) :: res
       integer :: status
    end function  %(name)s_find_extreme
        '''
        yield cf_proto, cf_source_template % (locals()), pyf_template % (locals ())

        cf_proto = 'int cf_%(name)s_find_zero(int j0, int j1, double *fm, int n, int m, int order, double* result)' % (locals())
        order_cases = '\n    '.join(order_cases_zero)
        cf_source_template = '''
%(cf_proto)s
{
  switch (order)
  {
    %(order_cases)s
    default:
      *result = 0.0;
  }
  return -2;
}
        '''
        pyf_template = '''
    function %(name)s_find_zero(j0, j1, f, n, m, order, res) result (status)
       intent(c) %(name)s_find_zero
       fortranname cf_%(name)s_find_zero
       integer intent(in, c) :: j0, j1
       double precision dimension (m, n), intent(in,c):: f
       integer, depend(f), intent(c,hide) :: n = (shape(f,1)==1?shape (f,0):shape(f,1))
       integer, depend(f), intent(c,hide) :: m = (shape(f,1)==1?1:shape(f,0))
       integer intent(c), optional :: order = 0
       double precision, intent(out) :: res
       integer :: status
    end function  %(name)s_find_zero
        '''
        yield cf_proto, cf_source_template % (locals()), pyf_template % (locals ())

        cf_proto = 'double cf_%(name)s_evaluate(double y, double *fm, int n, int m, int order)' % (locals())
        horner = 'a%s' % (poly_order)
        for e in reversed(range(poly_order)):
            horner = 'a%s+(%s)*r' % (e, horner)
        cf_source_template = '''
%(cf_proto)s
{
  %(init_coeffs_ref)s
  /*
  int j = floor((y<0?-y:y));
  double r = (y<0?-y:y) - j;
  */
  int j = floor (y);
  double r = y - j;
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



        for k in [1,2]:
            cf_proto = 'double cf_%(name)s_f%(k)s_evaluate(double x, double *f, int n, int order)' % (locals ())
            cases = []
            for order in range(3):
                f_expr = eval('pwf%s(f,i,s)'%(k), self.namespace).variable_diff(self.namespace['s'], order).evalf ();
                f_exps = sorted (f_expr.data)[::-1]
                if not f_exps:
                    break
                horner = f_expr.data[f_exps[0]]
                for e in f_exps[1:]:
                    horner = '%s + (%s)*s' % (f_expr.data[e], horner)
                cases.append('case %s: return %s;' % (order, horner))
            cases = '\n    '.join (cases)
            cf_source_template = '''
#ifdef F
#undef F
#endif
#define F(I) ((I)<0?((1-(I))*f[0]+(I)*f[1]):((I)>=n?(((I)-n+2)*f[n-1]-((I)-n+1)*f[n-2]):f[(I)]))

%(cf_proto)s
{
  int i = floor(x);
  double s = x - floor(x);
  switch (order)
  {
    %(cases)s
  }
  return 0.0;
}
        '''
            pyf_template = '''
    function %(name)s_f%(k)s_evaluate (x, f, n, order) result (value)
      intent (c) %(name)s_f%(k)s_evaluate
      fortranname cf_%(name)s_f%(k)s_evaluate
      double precision intent(in, c) :: x
      double precision dimension (n), intent(in,c):: f
      integer, depend(f), intent(c,hide) :: n = shape (f,0)
      integer intent(c), optional :: order = 0
      double precision :: value
    end function %(name)s_f%(k)s_evaluate
'''

            yield cf_proto, cf_source_template % (locals()), pyf_template % (locals())
        
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
        a00 = ('constant', 'f1(x)*f2(x+y)'),
        a11 = ('linear', 'f1(x)*f2(x+y)'),
        a22 = ('qint', 'f1(x)*f2(x+y)'),
        a33 = ('cint', 'f1(x)*f2(x+y)'),
        #b00 = ('constant','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)-f2(L))'),
        #b11 = ('linear','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)-f2(L))'),
        #b22 = ('qint','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)-f2(L))'),
        #b33 = ('cint','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)-f2(L))'),
        #c00 = ('constant','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)+f(x)/2+3*f[0]/2-9*f2(L)/4)'),
        #c11 = ('linear','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)+f(x)/2+3*f[0]/2-9*f2(L)/4)'),
        #c22 = ('qint','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)+f(x)/2+3*f[0]/2-9*f2(L)/4)'),
        #c33 = ('cint','(f1(x)-(f1(0)+f1(L))/2)*(f2(x+y)+f(x)/2+3*f[0]/2-9*f2(L)/4)'),
        e00 = ('constant', '(f1(x)-f1(x+y))*(f2(x)-f2(x+y))'),
        e11 = ('linear', '(f1(x)-f1(x+y))*(f2(x)-f2(x+y))'),
        e22 = ('qint', '(f1(x)-f1(x+y))*(f2(x)-f2(x+y))'),
        e33 = ('cint', '(f1(x)-f1(x+y))*(f2(x)-f2(x+y))'),
        ).iteritems():
        #if not name.startswith ('e'):
        #    continue
        #if name not in ['a11','a22','a33']:
        #    continue
        g = Generator(pwf)
        for proto, source, interface in g.generate_source(name,
                                                          integrand=integrand):
            source_file.write(source)
            header_file.write('extern %s;\n' % (proto))
            if interface:
                pyf_file.write(interface)

    for proto, source, interface in g.generate_approximation_source ():
        source_file.write(source)
        header_file.write('extern %s;\n' % (proto))
        if interface:
            pyf_file.write(interface)

    for proto, source, interface in g.generate_find_real_zero_in_01_source ():
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


