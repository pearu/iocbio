#
# Author: Pearu Peterson
# Created: February 2011

import os
import sys
import itertools
import subprocess

from builder import IsotopeModel, StrContext, symbol, Terms, number, pp, pf
from steadystate import SteadyFluxAnalyzer

system_name = 'bi_loop'

if system_name == 'bi_loop':
    full_system_str = '''
C + A | {1:1}
C + B | {2:1}

D + B | {1:1}

C + D | {2:1}
C + E | {1:1}
A + E | {1:1}

A_E   : A     <=> E
AB_C  : A + B <=> C
C_DE  : C     <=> D + E
B_D   : D     <=> B
'''
elif system_name == 'stable_loop':
    full_system_str = '''
C + A | {1:1}
C + B | {2:1}
B + D | {2:1}
C + D | {2:1}
A + D | {1:1}

AB_CD : A + B <=> C + D
C_AD  : C     <=> A + D
A_D   : A     <=> D
'''

make_indices = lambda repeat: map(''.join,itertools.product('01', repeat=repeat)) if repeat else ['']

class Model (IsotopeModel):
    use_sum = 0
    use_mass = 0
    replace_total_sum_with_one = 0

    A_labeling = {'0':0, '1':1}
    system_name = system_name
    system_str = full_system_str

    import_dir = 'generated'
    model_name = 'model_' + system_name 

    if use_mass:
        model_name += '_mass'
    if use_sum:
        model_name += '_sum'
    if replace_total_sum_with_one:
        model_name += '_repl1'

    print 'A labeling : ', A_labeling
    print 'System name: ', system_name    
    print 'Model name : ', model_name

    a = flux_analyzer = SteadyFluxAnalyzer(system_str, split_bidirectional_fluxes=False)    
    print 'Steady-state system:'
    mat = a.label_matrix(a.stoichiometry, a.species, a.reactions)
    print mat.__str__(max_nrows=20, max_ncols=21)
    reactions = flux_analyzer.reactions_info.values()
    reaction_info = flux_analyzer.species_info
    metabolite_lengths = reaction_info.pop('metabolite_lengths')
    reaction_pairs = reaction_info

    index_dic = {}
    for met, length in metabolite_lengths.items():
        index_dic[met] = make_indices(length)

    #pp((reaction_pairs, metabolite_lengths))
    #pp((a.species, index_dic))

    species = {}
    for sp in flux_analyzer.species:
        species[sp] = index_dic[sp]

    latex_name_map = dict(A=r'\Aname{}',
                          B=r'\Bname{}',
                          C=r'\Cname{}',
                          D=r'\Dname{}',
                          E=r'\Ename{}',
                          Ain=r'\Ainname{}',
                          Eout=r'\Eoutname{}',
                          )

    c_name_map = {}
    for k in latex_name_map.keys():
        c_name_map[k] = k

    def check_equivalence(self, s1, s2):
        if s1.prefix != s2.prefix:
            return False
        if s1==s2:
            return True
        #parts1 = s1.index.split('_')
        #parts2 = s2.index.split('_')
        #for p1, p2 in zip (parts1, parts2):
        #    if p1.count ('1') != p2.count ('1'):
        #        return False
        pp((s1, s2))
        #return True
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
                        #print reactant, p_key, atom_dic, rpat, ppat
                        for ra, pa in atom_dic.items():
                            if rpat[ra - 1] != ppat[pa -1]:
                                #print 'rxn_ok: ', False
                                return False
            #print 'rxn_ok: ', rxn_ok
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
        use_sum = self.use_sum
        use_mass = self.use_mass
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
            stream.write ('/*\n%s\n*/\n' % ('\n'.join(line for line in self.system_str.splitlines () if line.strip() and not line.strip().startswith ('#'))))
            stream.write (c_header)
            
            # Assign the input_list to their isotopomer names.
            initial_value_list = []
            derivatives_list = []
            index = 0
            for k in sorted (c_eqns):
                #print k, is_unlabeled (k)
                it_key = StrContext.map(k)
                if it_key.startswith ('A'):
                    value = self.A_labeling[it_key[-1]]
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
            for met in self.metabolite_lengths.keys():
                if met == 'A':
                    continue
                pool_set.add(StrContext.map(met))
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
                if k.startswith ('A'):# k.split('_')[0] == 'Wo':
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
        build_dir = self.import_dir
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

model = Model()

if __name__ == '__main__':

    model.compile_ccode(debug=False, stage=2)

