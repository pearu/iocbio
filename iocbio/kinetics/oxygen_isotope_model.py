#
# Author: Pearu Peterson
# Created: February 2011

import os
import sys
import pprint
import tempfile
import itertools
import subprocess

from isotope import IsotopeModel, StrContext, symbol, Terms, number

full_system_str = '''

ADPms : ADPm <=> ADPs
Pms   : Pm <=> Ps
Wos   : Wo <=> Ws
ASs   : ADPs + Ps <=> ATPs + Ws
ATPsm : ATPs => ATPm

ATPoe : ATPo => ATPe
Peo   : Pe <=> Po
Weo   : We <=> Wo
ASe   : ATPe + We <=> ADPe + Pe
ADPeo : ADPe <=> ADPo

AKi  : ADPi + ADPi <=> ATPi
AKo  : ATPo <=> ADPo + ADPo

CKi   : ATPi <=> ADPi + CPi
CKo   : CPo + ADPo <=> ATPo 

ADPim : ADPi <=> ADPm
ADPoi : ADPo <=> ADPi 

ATPmi : ATPm <=> ATPi
ATPio : ATPi <=> ATPo

Cio   : CPi <=> CPo
Pom   : Po => Pm
'''

make_indices = lambda repeat: map(''.join,itertools.product('01', repeat=repeat)) if repeat else ['']

n = 3
T1indices = make_indices(n)
T2indices = make_indices(n)
W_indices = make_indices(1)
ADP_indices = make_indices(n)
CP_indices = make_indices(n)
P_indices = make_indices(n+1)

print T1indices, type(T1indices)

ATP_indices = []
for t1 in T1indices:
    for t2 in T2indices:
        ATP_indices.append (t1+'_'+t2)

class Model(IsotopeModel):

    m_species = dict(ATPm = ATP_indices,
                     ADPm = ADP_indices,
                     Pm   = P_indices,
                     )

    i_species = dict(ATPi = ATP_indices,
                     ADPi = ADP_indices,
                     CPi  = CP_indices,
                     )

    c_species = dict(ATPo = ATP_indices,
                     Wo    = W_indices,
                     ADPo = ADP_indices,
                     Po   = P_indices,
                     CPo  = CP_indices,
                     )
    
    e_species = dict(ATPe = ATP_indices,
                     We = W_indices,
                     ADPe = ADP_indices,
                     Pe = P_indices,
                     )
    
    s_species = dict(ATPs = ATP_indices,
                     Ws = W_indices,
                     ADPs = ADP_indices,
                     Ps = P_indices,
                     )

    index_dic = dict()
    index_dic.update(m_species)
    index_dic.update(i_species)
    index_dic.update(c_species)
    index_dic.update(e_species)
    index_dic.update(s_species)
    
    c_name_map = dict(ATPo=r'ATPo',
                      ATPm=r'ATPm',
                      ATPe=r'ATPe',
                      ATPs=r'ATPs',
                      ATPi=r'ATPi',
                      ADPo=r'ADPo',
                      ADPe=r'ADPe',
                      ADPs=r'ADPs',
                      ADPm=r'ADPm',
                      ADPi=r'ADPi',
                      Wo = r'Wo',
                      Ws = r'Ws',
                      We = r'We',
                      Po = r'Po',
                      Pe = r'Pe',
                      Ps = r'Ps',
                      Pm = r'Pm',
                      CPo = r'CPo',
                      CPi = r'CPi',
                      )


    system_str = full_system_str

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

    transport_reactions = []
    for a, b in [('ATPi', 'ATPo'), ('ATPm', 'ATPi'),
                 ('ADPo', 'ADPi'), ('ADPi', 'ADPm'), ('ADPm', 'ADPs'), ('ADPe', 'ADPo'),
                 ('Pm', 'Ps'), ('Pe', 'Po'),
                 ('Wo', 'Ws'), ('We', 'Wo'),
                 ('CPi', 'CPo'),        
                 ]:
        transport_reactions.append('%s->%s' % (a,b))
        transport_reactions.append('%s<-%s' % (a,b))
        
    for a, b in [('Po', 'Pm'), ('ATPo', 'ATPe'), ('ATPs', 'ATPm'),]:
        transport_reactions.append('%s->%s' % (a,b))

    def check_reaction(self, reaction_pattern, rindices, pindices):
        if reaction_pattern in self.transport_reactions:
            return rindices == pindices
        if reaction_pattern in ['CPo+ADPo->ATPo', 'CPo+ADPo<-ATPo']:
            atp, = pindices
            cp, adp = rindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == cp
        if reaction_pattern in ['ATPi->ADPi+CPi', 'ATPi<-ADPi+CPi']:
            atp, = rindices
            adp, cp = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == cp
        if reaction_pattern in ['ADPi+ADPi->ATPi','ADPi+ADPi<-ATPi']:
            adp, adp2 = rindices
            atp, = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == adp2
        if reaction_pattern in ['ATPo->ADPo+ADPo','ATPo<-ADPo+ADPo']:
            atp, = rindices
            adp, adp2 = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == adp2
        if reaction_pattern in ['ATPe+We->ADPe+Pe', 'ATPe+We<-ADPe+Pe']:
            atp, w = rindices
            adp, p = pindices
            t1, t2 = atp.split ('_')
            return t1==adp and (t2+w).count('1')==p.count ('1')
        if reaction_pattern in ['ADPs+Ps->ATPs+Ws', 'ADPs+Ps<-ATPs+Ws']:
            atp, w = pindices
            adp, p = rindices
            t1, t2 = atp.split ('_')
            return t1==adp and (t2+w).count('1')==p.count ('1')
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
            stream.write ('/*\n%s\n*/\n' % ('\n'.join(line for line in self.system_str.splitlines () if line.strip() and not line.strip().startswith ('#'))))
            stream.write (c_header)
            
            # Assign the input_list to their isotopomer names.
            initial_value_list = []
            derivatives_list = []
            index = 0
            for k in sorted (c_eqns):
                #print k, is_unlabeled (k)
                it_key = StrContext.map(k)
                if it_key.startswith ('Wo'):
                    value = self.water_labeling[it_key[-1]]
                    stream.write('double %s = %s ;\n' %(it_key, value))
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
            for k in sorted(eqns):
                if k.split('_')[0] == 'Wo':
                    continue
                pool_set.add(StrContext.map(k).split('_')[0])
            pool_list = []
            for index, pool in enumerate(pool_set):
                stream.write('double pool_%s = pool_list[%s] ;\n' %(pool, index))
                #pool_list.append({pool:None})
                pool_list.append (pool)
            c_variables['pool_list'] = pool_list
            stream.write('\n')

            # Write the equations.
            index = 0
            for k in sorted (c_eqns):
                if use_sum and is_unlabeled (k):
                    continue
                if k.startswith ('Wo'):# k.split('_')[0] == 'Wo':
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
                    rs += reactant.__str__() + '+'
                rs = rs[:-1]
                for products in vl:
                    ps = ''
                    for product in products:
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
        import_dir = '.'
        build_dir = tempfile.mkdtemp(dir='.')
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

        cv_string = pprint.pformat(self.c_variables)
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

        commands = ["f2py -c %s %s" %(file_names['pyf'], file_names['c']),]

        import_string = 'import {0} as c_package'.format(mn)
        print import_string
        try:
            exec(import_string)
        except ImportError as e:
            for cmd in commands:
                print "Executing command: %s" %(cmd)
                cmd_proc = subprocess.Popen(cmd, **Popen_kwds)
                cmd_out, cmd_err = cmd_proc.communicate()
                if cmd_err.strip() != str():
                    msg = "Error executing command. %s" %(cmd)
                    print cmd_out
                    print cmd_err
                    raise SystemError(msg)#, cmd_out, cmd_err)

        cmd = 'rm -rf {0}'.format(build_dir)
        print "Executing command: %s" %(cmd)
        cmd_proc = subprocess.Popen(cmd, **Popen_kwds)
        cmd_out, cmd_err = cmd_proc.communicate()
        if cmd_err.strip() != str():
            msg = "Error executing command. %s" %(cmd)
            print cmd_out
            print cmd_err
            raise SystemError(msg)#, cmd_out, cmd_err)

                              
if __name__ == '__main__':
    
    frac_W = 0.3
    water_exchange = 0.000001
    
    water_labeling = {'0':1 - frac_W,
                      '1':frac_W}
    
    model = Model(water_labeling=water_labeling, water_exchange=water_exchange)
    
    print model.model_name, water_labeling, frac_W

    model.compile_ccode(debug=False, stage=None)