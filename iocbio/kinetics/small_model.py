#
# Author: David Schryer
# Created: February 2011

from collections import namedtuple

from builder import IsotopologueModelBuilder, pp
from solver import IsotopologueSolver
from utils import get_solution

SystemInput = namedtuple('SystemInput', 'independent_flux_dic, exchange_flux_dic, pool_dic')
System = namedtuple('System', 'name, string, labeled_species, input')
SolverInput = namedtuple('SolverInput', 'initial_time_step, end_time, integrator_params')
SolutionDef= namedtuple('SolutionDef', 'name, solver_input, system')

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

bi_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 60, dict()), bi_loop)
bi_loop_long = SolutionDef('long', SolverInput(30, 6000, dict()), bi_loop)

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

stable_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 60, dict()), stable_loop)
stable_loop_long = SolutionDef('long', SolverInput(30, 6000, dict()), stable_loop)


if __name__ == '__main__':
    
    P = bi_loop_dynamic
    P = stable_loop_dynamic
    P = bi_loop_long
    #P = stable_loop_long

    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=True),
                                     )

    sys_hess = model.system_hessian()
    pp(sys_hess)

    model.compile_ccode(debug=False, stage=2)

    it_solver = IsotopologueSolver(model)
    it_solver.set_data(solution_name=P.name, **P.system.input._asdict())
    it_solver.solve(**P.solver_input._asdict())

    s_list, sp_list, time_list, fd = get_solution(it_solver.model, P.name)

    print len(s_list), len(sp_list), len(time_list)
    pp(fd)

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_list, s_list, 'o')
    plt.show()

