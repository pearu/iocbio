#
# Author: David Schryer
# Created: February 2011

from collections import namedtuple

from builder import IsotopologueModelBuilder, pp
from solver import IsotopologueSolver
from utils import analyze_solution

SystemInput = namedtuple('SystemInput', 'independent_flux_dic, exchange_flux_dic, pool_dic')
System = namedtuple('System', 'name, string, labeled_species, input')
IntegratorParams = namedtuple('IntegratorParams', 'method, rtol, atol, nsteps, with_jacobian, order, max_step, min_step')
SolverInput = namedtuple('SolverInput', 'initial_time_step, end_time, integrator_parameters')
SolutionDef= namedtuple('SolutionDef', 'name, solver_input, system')

int_params = IntegratorParams('adams', 1e-12, 1e-12, 2000000, False, 0, None, None)._asdict()

m_loop = System('m_loop', '''
A + C | {1:2}
B + C | {1:1}

D + A | {1:1}
E + B | {1:1}

D + C | {1:2}
E + C | {1:1}

AB_C  : A + B <=> C
C_DE  : C     <=> D + E
DE_AB : D + E <=> A + B
''',
                 dict(A={'0':0, '1':1}),
                 SystemInput(dict(AB_C=0),
                             dict(AB_C=0.1, C_DE=0.2, DE_AB=0.3),
                             dict(A=4, B=5, C=6, D=7, E=8),
                             ))

m_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 100, int_params), m_loop)
m_loop_mid = SolutionDef('mid', SolverInput(10, 2800, int_params), m_loop)
m_loop_long = SolutionDef('long', SolverInput(20, 10000, int_params), m_loop)

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
                 SystemInput(dict(AB_C=0),
                             dict(AB_C=0.1, C_DE=0.2, B_D=0.3, A_E=0.4),
                             dict(A=4, B=5, C=6, D=7, E=8),
                             ))

bi_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 30, int_params), bi_loop)
bi_loop_mid = SolutionDef('mid', SolverInput(1, 2800, int_params), bi_loop)
bi_loop_long = SolutionDef('long', SolverInput(10, 10000, int_params), bi_loop)

bi_loop_flow = System('bi_loop_flow', '''
C + A | {1:1}
C + B | {2:1}

D + B | {1:1}
A + E | {1:1}

C + D | {2:1}
C + E | {1:1}

XA + A | {1:1}
XB + B | {1:1}
XD + D | {1:1}
XE + E | {1:1}

A_E   : A     <=> E
AB_C  : A + B <=> C
C_DE  : C     <=> D + E
B_D   : D     <=> B

BR_A  : XA     => A
BR_B  : XB     => B
BR_E  : E      => XE
BR_D  : D      => XD

''',
                 dict(XA={'0':0, '1':1},
                      XB={'0':1, '1':0},
                      XE={'0':1, '1':0},
                      XD={'0':1, '1':0}),
                 SystemInput(dict(A_E=0.1, B_D=0.9, AB_C=1.1),
                             dict(AB_C=0.1, C_DE=0.2, B_D=0.3, A_E=0.4),
                             dict(A=4, B=5, C=6, D=7, E=8),
                             ))

bi_loop_flow_dynamic = SolutionDef('dynamic', SolverInput(1, 30, int_params), bi_loop_flow)
bi_loop_flow_mid = SolutionDef('mid', SolverInput(1, 2800, int_params), bi_loop_flow)
bi_loop_flow_long = SolutionDef('long', SolverInput(10, 8000, int_params), bi_loop_flow)

w_loop = System('w_loop', '''
B + A | {1:1, 2:2}
B + K | {1:1, 2:2}
B + E | {1:1, 2:2}
E + H | {1:1, 2:2}

B + C | {1:1, 2:2}
E + C | {1:3, 2:4}

F + C | {1:1}
D + C | {1:2, 2:3, 3:4}

G + D | {1:3}
E + D | {1:1, 2:2}

XA + A | {1:1, 2:2}
XK + K | {1:1, 2:2}
XH + H | {1:1, 2:2}
XG + G | {1:1}
XF + F | {1:1}

BR_A  : XA => A
BR_K  : K => XK
BR_H  : H => XH
BR_F  : F => XF
BR_G  : G => XG

A_B   : A => B
B_K   : B => K
B_E   : B <=> E
E_H   : E => H

BE_C  : B + E <=> C
C_DF  : C => D + F
D_GE  : D => G + E
''',
                dict(XA={'00':0, '11':1, '01':0, '10':0},
                     XK={'00':1, '11':0, '01':0, '10':0},
                     XH={'00':1, '11':0, '01':0, '10':0},
                     XF={'0':1, '1':0},
                     XG={'0':1, '1':0}),
                     SystemInput(dict(A_B=3.0, B_E=0, BE_C=2.3),
                                 dict(B_E=0.3, BE_C=0.5),
                                 dict(A=4, B=5, C=6, D=7, E=8, F=9, G=10, H=2, K=1.1),
                                 ))

w_loop_dynamic = SolutionDef('dynamic', SolverInput(1, 60, int_params), w_loop)
w_loop_long = SolutionDef('long', SolverInput(30, 10000, int_params), w_loop)



if __name__ == '__main__':

    P = m_loop_dynamic
    #P = m_loop_mid
    #P = m_loop_long
    
    #P = bi_loop_dynamic
    #P = bi_loop_mid
    #P = bi_loop_long
    
    #P = bi_loop_flow_dynamic
    #P = bi_loop_flow_mid
    #P = bi_loop_flow_long

    #P = w_loop_dynamic
    #P = w_loop_long

    P.solver_input.integrator_parameters['rtol'] = 1e-10
    P.solver_input.integrator_parameters['atol'] = 1e-10

    simplify_sums = True

    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=simplify_sums),
                                     )

    sys_jac = model.system_jacobian

    model.compile_ccode(debug=False, stage=2)

    it_solver = IsotopologueSolver(model)
    it_solver.set_data(solution_name=P.name, **P.system.input._asdict())
    it_solver.solve(**P.solver_input._asdict())

    analyze_solution(model, P.name, P.solver_input.end_time)
