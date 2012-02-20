#
# Author: David Schryer
# Created: February 2011

from collections import namedtuple, defaultdict

from mytools.tools import drop_to_ipython as dti

from builder import IsotopologueModelBuilder, pp
from solver import IsotopologueSolver
from utils import get_solution

SystemInput = namedtuple('SystemInput', 'independent_flux_dic, exchange_flux_dic, pool_dic')
System = namedtuple('System', 'name, string, labeled_species, input')
IntegratorParams = namedtuple('IntegratorParams', 'method, rtol, atol, nsteps, with_jacobian, order, max_step, min_step')
SolverInput = namedtuple('SolverInput', 'initial_time_step, end_time, integrator_parameters')
SolutionDef= namedtuple('SolutionDef', 'name, solver_input, system')

int_params = IntegratorParams('adams', 1e-12, 1e-12, 2000000, False, 0, None, None)._asdict()

bi_loop = System('bi_loop', '''
C + A | {1:1}
C + B | {2:1}

D + B | {1:1}
A + E | {1:1}

C + D | {2:1}
C + E | {1:1}

A_E   : A     <=> E
AB_C  : A + B <=> C
C_DE  : C     <=> D + E
B_D   : D     <=> B
''',
                 dict(A={'0':0, '1':1}),
                 SystemInput(dict(AB_C=1.1),
                             dict(AB_C=0.1, C_DE=0.2, B_D=0.3, A_E=0.4),
                             dict(A=4, B=5, C=6, D=7, E=8),
                             ))

bi_loop_dynamic = SolutionDef('dynamic', SolverInput(0.01, 30, int_params), bi_loop)
bi_loop_mid = SolutionDef('mid', SolverInput(1, 2800, int_params), bi_loop)
bi_loop_long = SolutionDef('long', SolverInput(10, 6000, int_params), bi_loop)

stable_loop = System('stable_loop', '''
B + A | {1:1}
C + B | {1:1}
A + C | {1:1}

A_B   : A <=> B
B_C   : B <=> C
C_A   : C <=> A
''',
                     dict(A={'0':0, '1':1}),
                     SystemInput(dict(A_B=1.1),
                                 dict(A_B=0.1, B_C=0.2, C_A=0.3),
                                 dict(A=4, B=5, C=6),
                                 ))

stable_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 60, int_params), stable_loop)
stable_loop_long = SolutionDef('long', SolverInput(30, 6000, int_params), stable_loop)


if __name__ == '__main__':
    
    P = bi_loop_dynamic
    #P = bi_loop_mid
    #P = bi_loop_long
    #P = stable_loop_dynamic
    #P = stable_loop_long

    P.solver_input.integrator_parameters['rtol'] = 1e-12
    P.solver_input.integrator_parameters['atol'] = 1e-12

    simplify_sums = True

    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=simplify_sums),
                                     )

    sys_hess = model.system_hessian

    model.compile_ccode(debug=False, stage=2)

    it_solver = IsotopologueSolver(model)
    it_solver.set_data(solution_name=P.name, **P.system.input._asdict())
    it_solver.solve(**P.solver_input._asdict())

    s_list, sp_list, time_list, fd = get_solution(model, P.name)

    lsd = model.labeled_species
    lab_dic = dict()
    for met_key, l_dic in lsd.items():
        for it_code, it_value in l_dic.items():
            lab_dic[met_key + it_code] = it_value

    pp(sp_list)
    one_list = []
    for i, s_point in enumerate(s_list):
        inner_dic = defaultdict(list)
        for j, v in enumerate(s_point):
            sp = sp_list[j][0]
            inner_dic[sp].append(v)
        inner_list = []
        for vl in inner_dic.values():
            inner_list.append(sum(vl))
        one_list.append(inner_list)

    hess_list = []
    for i, s_point in enumerate(s_list):
        t_point = time_list[i]
        s_dic = dict()
        for j, sp in enumerate(sp_list):
            s_dic[sp] = s_point[j]
        sub_sys_hess = dict()
        for ek, ed in sys_hess.items():
            inner_dic = dict()
            for tk, expr in ed.items():
                inner_dic[tk] = expr.subs(fd).subs(s_dic).subs(lab_dic).data
            sub_sys_hess[ek] = inner_dic
        hess_list.append(sub_sys_hess)

    p_keys = sp_list #['D0', 'D1', 'C11', 'E0']
    for p_key in p_keys:
        print
        print p_key
        pp(sys_hess[p_key])
        #print
        #pp(hess_list[0][p_key])
        #print
        #pp(hess_list[-1][p_key])

    import numpy 
    import numpy.linalg
    import scipy.linalg
    import matplotlib.pyplot as plt

    for i, t in enumerate(time_list):

        to_matA = []
        for k, innner_dic in hess_list[i].items():
            inner_list = []
            for ik in hess_list[i].keys():
                #pp((hess_list[i][k], i, k, ik))
                inner_list.append(float(hess_list[i][k][ik]))
            to_matA.append(inner_list)

        jac_array = numpy.array(to_matA)
        cond_num = numpy.linalg.cond(jac_array)
        
        print 'Time: {0}  Condition number: {1}'.format(t, cond_num)

        numpy.savetxt('generated/time_{0}_jac_array.txt'.format(t), jac_array)#, fmt='%.24e')

    tol = 1e-3
    for t in time_list:
        if abs(t - 1940) < tol:
            jac_array = numpy.loadtxt('generated/time_1950.0_jac_array.txt')
        else:
            jac_array = numpy.loadtxt('generated/time_{0}_jac_array.txt'.format(t))
        
        e_values, e_vectors = scipy.linalg.eig(jac_array)
        eva, eve = zip(*sorted(zip(e_values, e_vectors)))

        evas = []
        for i, ev in enumerate(eva):
            if abs(ev) < tol:
                evas.append(ev)

        #if t == 3710:
        #    print e_values
        #    print evas, eves
        #    exit()

        if evas == list():
            numpy.savetxt('generated/time_{0}_e_values.txt'.format(t), e_values)
            numpy.savetxt('generated/time_{0}_e_vectors.txt'.format(t), [1e999])
        else:
            numpy.savetxt('generated/time_{0}_e_values.txt'.format(t), e_values)
            numpy.savetxt('generated/time_{0}_e_vectors.txt'.format(t), e_vectors)

    new_time_list = []
    e_value_list = []
    zero_e_vector_list = []
    pos_e_vector_list = []
    previous_zero_e_values = []
    for t in time_list:

        e_values = numpy.loadtxt('generated/time_{0}_e_values.txt'.format(t))
        e_vectors = numpy.loadtxt('generated/time_{0}_e_vectors.txt'.format(t))
        
        #e_values = numpy.array([e_values])
        #e_vectors = numpy.array([e_vectors])

        real_e_values = []
        pos_e_values = []
        zero_e_values = []
        for k, c in enumerate(e_values):
            if c.real + tol > 0:
                if abs(c.real) < tol:
                    #print c
                    zero_e_values.append((k, c.real))
                else:
                    pos_e_values.append((k, c.real))
            real_e_values.append(c.real)
            
        e_value_list.append(sorted(real_e_values))

        if e_vectors.tolist() == 1e999:
            continue

        new_time_list.append(t)

        #print t, zero_e_values

        zero_e_vectors = []
        sp_set = set()
        for k, e_value in zero_e_values:
            #print k, e_value, e_vectors
            e_vec = e_vectors[:,k]
            sys_e_vec = e_vec * numpy.sign(e_vec[0].real)
            zero_e_vectors.append(sys_e_vec.real)

        pos_e_vectors = []
        for k, e_value in pos_e_values:
            e_vec = e_vectors[:,k]
            real_e_vec = []
            for c in e_vec:
                real_e_vec.append(c.real)
            pos_e_vectors.append(real_e_vec)

        pos_e_vector_list.append(pos_e_vectors)

        if len(zero_e_vectors):
            #pp((t, pos_e_values, zero_e_values, e_values))
            p,l,u = scipy.linalg.lu(zero_e_vectors)
            if len(zero_e_vectors) == 2:
                if not abs(u[0,0]) < tol:
                    u[0] /= (u[0,0])
                if not abs(u[1,1]) < tol:
                    u[1] /= (u[1,1])
                zero_e_vector_list.append([u[0].real - u[0].real, u[1].real])
            else:
                zero_e_vector_list.append(u.real)
        else:
            pp((t, pos_e_values, zero_e_values, e_values))
            print 'No zero e-values found.  Exiting...'
            exit()


    if simplify_sums:
        title_start = 'Unstable '
    else:
        title_start = 'Stable '
        
    fig = plt.figure(figsize=(8.5,11))
    ax = fig.add_subplot(311)
    ax.set_title(title_start + P.name + ' e-values')
    ax.plot(time_list, e_value_list, ':')

    ax = fig.add_subplot(312)
    ax.set_title(title_start + P.name + ' zero e-value e-vector')
    vl = numpy.array(zero_e_vector_list)

    avl = numpy.sort(vl, axis=2)

    vls = []
    for i in range(avl.shape[1]):
        vls.append(avl[:,i,:])

    for ivl in vls:
        for i in range(ivl.shape[1]):
            #ax.plot(time_list, ivl[:,i])
            
            if ivl[:,i].sum() < 0.0001:
                ax.plot(new_time_list, ivl[:, i], '-')
            else:
                no_zeros = []
                new_time = []
                for j, t in enumerate(new_time_list):
                    if abs(ivl[j,i]) < tol:
                        continue
                    no_zeros.append(ivl[j, i].real)
                    new_time.append(t)
                ax.plot(new_time, no_zeros, '-')

    ax = fig.add_subplot(313)
    ax.set_title(title_start + P.name + ' solution')
    ax.plot(time_list, s_list, ':')
    ax.plot(time_list, one_list, '--')

    fn = '_'.join((title_start.strip().lower(), P.name + '.pdf'))
    plt.subplots_adjust(hspace=0.4)

    print 'Saving plot: {0}'.format(fn)
    plt.savefig(fn)
    plt.show()
    
