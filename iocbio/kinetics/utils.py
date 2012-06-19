import numpy 
import numpy.linalg
import scipy.linalg
import matplotlib.pyplot as plt

from collections import defaultdict
from builder import flush

import sympycore

from mytools.tools import drop_to_ipython as dti

def get_solution(it_model, solution_name):    
    import_template = 'from {0.import_dir}.{0.model_name}_c_variables import c_variables'
    import_string = import_template.format(it_model)
    try:
        exec(import_string)
    except ImportError, msg:
        raise UserWarning(msg, it_model)
    species_list = c_variables['input_list']
    
    i_str = 'solution_list, time_list'
    import_string = 'from {0.import_dir}.{0.model_name}_solution_{1}_output import {2}'
    try:
        exec(import_string.format(it_model, solution_name, i_str))
    except ImportError, msg:
        raise UserWarning(msg)

    import_string = 'from {0.import_dir}.{0.model_name}_solution_{1}_input import input_variables'
    try:
        exec(import_string.format(it_model, solution_name))
    except ImportError, msg:
        raise UserWarning(msg)
    pool_dic = input_variables['pool_dic']
    fd = input_variables['flux_dic']

    assert len(species_list) == len(solution_list[0]), `len(species_list) == len(solution_list[0])`

    if hasattr(solution_list, 'tolist'):
        solution_list = solution_list.tolist()
    if hasattr(time_list, 'tolist'):
        time_list = time_list.tolist()

    return solution_list, species_list, time_list, fd


def analyze_solution(model, solution_name, end_time=60):

    solution_list, species_list, time_list, fd = get_solution(model, solution_name)
        
    lsd = model.labeled_species
    lab_dic = dict()
    for met_key, l_dic in lsd.items():
        for it_code, it_value in l_dic.items():
            lab_dic[met_key + it_code] = it_value

    one_list = []
    for i, s_point in enumerate(solution_list):
        inner_dic = defaultdict(list)
        for j, v in enumerate(s_point):
            sp = species_list[j][0]
            inner_dic[sp].append(v)
        inner_list = []
        for vl in inner_dic.values():
            inner_list.append(sum(vl))
        one_list.append(inner_list)

    jac_list = []
    for i, s_point in enumerate(solution_list):
        print 'Substitution is {0:0.1f}% complete.'.format(i / float(len(solution_list)) * 100)

        t_point = time_list[i]
        s_dic = dict()
        for j, sp in enumerate(species_list):
            s_dic[sp] = s_point[j]
        sub_sys_jac = dict()
        for ek, ed in model.system_jacobian.items():
            inner_dic = dict()
            for tk, expr in ed.items():
                value = expr.subs(fd).subs(s_dic).subs(lab_dic).data
                inner_dic[tk] = float(value)
            sub_sys_jac[ek] = inner_dic
        jac_list.append(sub_sys_jac)

    p_keys = species_list #['D0', 'D1', 'C11', 'E0']
    #for p_key in p_keys:
    #    print
    #    print p_key
    #    pp(sys_jac[p_key])
        #print
        #pp(jac_list[0][p_key])
        #print
        #pp(jac_list[-1][p_key])

    for i, t in enumerate(time_list):

        to_matA = []
        for k, innner_dic in jac_list[i].items():
            inner_list = []
            for ik in jac_list[i].keys():
                VAL = jac_list[i][k][ik]
                inner_list.append(VAL)
                    
            to_matA.append(inner_list)

        jac_array = numpy.array(to_matA)
        cond_num = numpy.linalg.cond(jac_array)

        #print 'Time: {0}  Condition number: {1}'.format(t, cond_num)

        fn = 'generated/time_{0}_jac_array.txt'.format(t)
        #print 'Writing {0}'.format(fn)
        numpy.savetxt(fn, jac_array)#, fmt='%.24e')

    tol = 1e-4
    for t in time_list:
        jac_array = numpy.loadtxt('generated/time_{0}_jac_array.txt'.format(t))

        e_values, e_vectors = scipy.linalg.eig(jac_array)

        eva = []
        eve = []
        index_dic = numpy.argsort(e_values)
        for j, ev in enumerate(e_values):
            for ind in index_dic:
                if j == ind:
                    eva.append(ev)
                    eve.append(e_vectors[j])

        #eva, eve = zip(*sorted(zip(e_values, e_vectors)))

        evas = []
        for i, ev in enumerate(eva):
            if abs(ev) < tol:
                evas.append(ev)

        #if t == 3710:
        #    print e_values
        #    print evas, eves
        #    exit()

        if evas == list():
            fn = 'generated/time_{0}_e_values.txt'.format(t)
            #print 'Writing {0}'.format(fn)
            numpy.savetxt(fn, e_values)

            fn = 'generated/time_{0}_e_vectors.txt'.format(t)
            #print 'Writing {0}'.format(fn)
            numpy.savetxt(fn, [1e999])
        else:
            fn = 'generated/time_{0}_e_values.txt'.format(t)
            #print 'Writing {0}'.format(fn)
            numpy.savetxt(fn, e_values)

            fn = 'generated/time_{0}_e_vectors.txt'.format(t)
            #print 'Writing {0}'.format(fn)
            numpy.savetxt(fn, e_vectors)

    new_time_list = []
    e_value_list = []
    zero_e_vector_list = []
    pos_e_vector_list = []
    previous_zero_e_values = []
    for t in time_list:

        e_values = numpy.loadtxt('generated/time_{0}_e_values.txt'.format(t))
        e_vectors = numpy.loadtxt('generated/time_{0}_e_vectors.txt'.format(t))

        real_e_values = []
        pos_e_values = []
        zero_e_values = []
        for k, c in enumerate(e_values):
            if c.real + tol > 0:
                if abs(c.real) < tol:
                    zero_e_values.append((k, c.real))
                else:
                    pos_e_values.append((k, c.real))
            real_e_values.append(c.real)

        e_value_list.append(sorted(real_e_values))

        if e_vectors.tolist() == 1e999:
            continue

        new_time_list.append(t)

        zero_e_vectors = []
        sp_set = set()
        for k, e_value in zero_e_values:
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


    if model.options['replace_total_sum_with_one']:
        title_start = 'Unstable '
    else:
        title_start = 'Stable '

    if zero_e_vector_list != []:
        fig = plt.figure(figsize=(8.5,11))
        ax = fig.add_subplot(311)
        ax.set_xlim(0, end_time)
        ax.set_title(title_start + solution_name + ' e-values')
        ax.plot(time_list, e_value_list, ':')

        xlim = ax.get_xlim()

        ax = fig.add_subplot(312)
        ax.set_title(title_start + solution_name + ' zero e-value e-vector')
        ax.set_xlim(*xlim)

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
        ax.set_title(title_start + solution_name + ' solution')
        ax.plot(time_list, solution_list, ':')
        ax.plot(time_list, one_list, '--')
        ax.set_xlim(*xlim)
    else:
        fig = plt.figure(figsize=(8.5,11))
        ax = fig.add_subplot(211)
        ax.set_xlim(0, end_time)
        ax.set_title(title_start + solution_name + ' e-values')
        ax.plot(time_list, e_value_list, ':')

        xlim = ax.get_xlim()

        ax = fig.add_subplot(212)
        ax.set_title(title_start + solution_name + ' solution')
        ax.plot(time_list, solution_list, ':')
        ax.plot(time_list, one_list, '--')
        ax.set_xlim(*xlim)


    fn = '_'.join((title_start.strip().lower(), solution_name + '.pdf'))
    plt.subplots_adjust(hspace=0.4)

    print 'Saving plot: {0}'.format(fn)
    plt.savefig(fn)
    plt.show()

