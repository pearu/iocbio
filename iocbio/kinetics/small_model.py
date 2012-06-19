#
# Author: David Schryer
# Created: February 2011

from builder import IsotopologueModelBuilder, pp
from solver import IsotopologueSolver
from utils import analyze_solution
from models import get_model

if __name__ == '__main__':

    P = get_model('bi_loop_dynamic')

    #P = m_loop_dynamic
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
    #simplify_sums = False

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
