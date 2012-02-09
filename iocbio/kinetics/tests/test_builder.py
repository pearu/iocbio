#
# Author: David Schryer
# Created: February 2011
import inspect
import StringIO

from sympycore import Calculus

from collections import namedtuple

from iocbio.kinetics.builder import IsotopologueModelBuilder, pp, pf
from iocbio.kinetics.solver import IsotopologueSolver

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

def fname():
    return inspect.stack()[1][3]
        
def assert_same_lines(output, expected_output, function_name, sub_lines=None):
    expected_lines = expected_output.split('\n')
    lines = output.split('\n')

    for line_number, line in enumerate(lines):
        line = line.strip()
        eline = expected_lines[line_number].strip()
        
        if sub_lines is not None:
            for ln, sub_line in sub_lines:
                if line_number == ln:
                    eline = sub_line.strip()

        assert line == eline, `(line, eline, line_number, function_name)`


def test_model_construction_A():
    P = stable_loop_dynamic
    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=False),
                                     )

    f = StringIO.StringIO()
    model.write_ccode(stream=f)
    f.seek(0)
    output = f.read()
    f.close()
    
    expected_output = '''/*
B + A | {1:1}
C + B | {1:1}
A + C | {1:1}
A_B   : A <=> B
B_C   : B <=> C
C_A   : C <=> A
*/
void c_equations(double* pool_list, double* flux_list, double* solver_time, double* input_list, double* out)
{
double A1 = 1 ;
double A0 = 0 ;
double B0 = input_list[0] ;
double B1 = input_list[1] ;
double C0 = input_list[2] ;
double C1 = input_list[3] ;

double fA_B = flux_list[0] ;
double rA_B = flux_list[1] ;
double fB_C = flux_list[2] ;
double rB_C = flux_list[3] ;
double fC_A = flux_list[4] ;
double rC_A = flux_list[5] ;

double pool_C = pool_list[0] ;
double pool_B = pool_list[1] ;

/*dB0/dt=*/ out[0] = ( +fA_B*(A0)+rB_C*(C0)-fB_C*(B0)-rA_B*(B0) )/ pool_B ;

/*dB1/dt=*/ out[1] = ( +fA_B*(A1)+rB_C*(C1)-fB_C*(B1)-rA_B*(B1) )/ pool_B ;

/*dC0/dt=*/ out[2] = ( +fB_C*(B0)+rC_A*(A0)-fC_A*(C0)-rB_C*(C0) )/ pool_C ;

/*dC1/dt=*/ out[3] = ( +fB_C*(B1)+rC_A*(A1)-fC_A*(C1)-rB_C*(C1) )/ pool_C ;

}
'''
    assert_same_lines(output, expected_output, fname())

    return model, P, output, expected_output

def test_model_construction_AA():
    model, P, output, expected_output = test_model_construction_A()
    
    # This simple loop should give the same output 
    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=True),
                                     )
    

    f = StringIO.StringIO()
    model.write_ccode(stream=f)
    f.seek(0)
    output = f.read()
    f.close()

    assert_same_lines(output, expected_output, fname())

    return model, P, output, expected_output

def test_model_construction_B():
    P = bi_loop_dynamic
    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=False),
                                     )

    f = StringIO.StringIO()
    model.write_ccode(stream=f)
    f.seek(0)
    output = f.read()
    f.close()
    
    expected_output = '''/*
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
*/
void c_equations(double* pool_list, double* flux_list, double* solver_time, double* input_list, double* out)
{
double A1 = 1 ;
double A0 = 0 ;
double B0 = input_list[0] ;
double B1 = input_list[1] ;
double C00 = input_list[2] ;
double C01 = input_list[3] ;
double C10 = input_list[4] ;
double C11 = input_list[5] ;
double D0 = input_list[6] ;
double D1 = input_list[7] ;
double E0 = input_list[8] ;
double E1 = input_list[9] ;

double fAB_C = flux_list[0] ;
double rAB_C = flux_list[1] ;
double fA_E = flux_list[2] ;
double rA_E = flux_list[3] ;
double fB_D = flux_list[4] ;
double rB_D = flux_list[5] ;
double fC_DE = flux_list[6] ;
double rC_DE = flux_list[7] ;

double pool_C = pool_list[0] ;
double pool_B = pool_list[1] ;
double pool_E = pool_list[2] ;
double pool_D = pool_list[3] ;

/*dB0/dt=*/ out[0] = ( +fB_D*(D0)+rAB_C*(C00+C10)-fAB_C*((A0+A1)*B0)-rB_D*(B0) )/ pool_B ;

/*dB1/dt=*/ out[1] = ( +fB_D*(D1)+rAB_C*(C01+C11)-fAB_C*((A0+A1)*B1)-rB_D*(B1) )/ pool_B ;

/*dC00/dt=*/ out[2] = ( +fAB_C*(A0*B0)+rC_DE*(D0*E0)-fC_DE*(C00)-rAB_C*(C00) )/ pool_C ;

/*dC01/dt=*/ out[3] = ( +fAB_C*(A0*B1)+rC_DE*(D1*E0)-fC_DE*(C01)-rAB_C*(C01) )/ pool_C ;

/*dC10/dt=*/ out[4] = ( +fAB_C*(A1*B0)+rC_DE*(D0*E1)-fC_DE*(C10)-rAB_C*(C10) )/ pool_C ;

/*dC11/dt=*/ out[5] = ( +fAB_C*(A1*B1)+rC_DE*(D1*E1)-fC_DE*(C11)-rAB_C*(C11) )/ pool_C ;

/*dD0/dt=*/ out[6] = ( +fC_DE*(C00+C10)+rB_D*(B0)-fB_D*(D0)-rC_DE*((E0+E1)*D0) )/ pool_D ;

/*dD1/dt=*/ out[7] = ( +fC_DE*(C01+C11)+rB_D*(B1)-fB_D*(D1)-rC_DE*((E0+E1)*D1) )/ pool_D ;

/*dE0/dt=*/ out[8] = ( +fA_E*(A0)+fC_DE*(C00+C01)-rA_E*(E0)-rC_DE*((D0+D1)*E0) )/ pool_E ;

/*dE1/dt=*/ out[9] = ( +fA_E*(A1)+fC_DE*(C10+C11)-rA_E*(E1)-rC_DE*((D0+D1)*E1) )/ pool_E ;

}
'''

    assert_same_lines(output, expected_output, fname())

    return model, P, output, expected_output

def test_model_construction_BB():
    model, P, output, expected_output = test_model_construction_B()

    model = IsotopologueModelBuilder(system=P.system.string,
                                     system_name=P.system.name,
                                     labeled_species=P.system.labeled_species,
                                     options=dict(replace_total_sum_with_one=True),
                                     )

    f = StringIO.StringIO()
    model.write_ccode(stream=f)
    f.seek(0)
    output = f.read()
    f.close()

    sub_lines = [(41, '/*dB0/dt=*/ out[0] = ( +fB_D*(D0)+rAB_C*(C00+C10)-fAB_C*(B0)-rB_D*(B0) )/ pool_B ;'),
                 (43, '/*dB1/dt=*/ out[1] = ( +fB_D*(D1)+rAB_C*(C01+C11)-fAB_C*(B1)-rB_D*(B1) )/ pool_B ;'),
                 (53, '/*dD0/dt=*/ out[6] = ( +fC_DE*(C00+C10)+rB_D*(B0)-fB_D*(D0)-rC_DE*(D0) )/ pool_D ;'),
                 (55, '/*dD1/dt=*/ out[7] = ( +fC_DE*(C01+C11)+rB_D*(B1)-fB_D*(D1)-rC_DE*(D1) )/ pool_D ;'),
                 (57, '/*dE0/dt=*/ out[8] = ( +fA_E*(A0)+fC_DE*(C00+C01)-rA_E*(E0)-rC_DE*(E0) )/ pool_E ;'),
                 (59, '/*dE1/dt=*/ out[9] = ( +fA_E*(A1)+fC_DE*(C10+C11)-rA_E*(E1)-rC_DE*(E1) )/ pool_E ;'),
                 ]

    assert_same_lines(output, expected_output, fname(), sub_lines=sub_lines)

    return model, P, output, expected_output

def test_hessian_construction_A():
    model, P, output, expected_output = test_model_construction_A()

    output = model.system_hessian()

    expected_output = {'B0': {'B0': Calculus('-rA_B - fB_C'),
                              'B1': Calculus('0'),
                              'C0': Calculus('rB_C'),
                              'C1': Calculus('0')},
                       'B1': {'B0': Calculus('0'),
                              'B1': Calculus('-rA_B - fB_C'),
                              'C0': Calculus('0'),
                              'C1': Calculus('rB_C')},
                       'C0': {'B0': Calculus('fB_C'),
                              'B1': Calculus('0'),
                              'C0': Calculus('-fC_A - rB_C'),
                              'C1': Calculus('0')},
                       'C1': {'B0': Calculus('0'),
                              'B1': Calculus('fB_C'),
                              'C0': Calculus('0'),
                              'C1': Calculus('-fC_A - rB_C')}}


    assert output == expected_output, `output, expected_output`

def test_hessian_construction_AA():
    model, P, output, expected_output = test_model_construction_AA()
    output = model.system_hessian()

    expected_output = {'B0': {'B0': Calculus('-rA_B - fB_C'),
                              'B1': Calculus('0'),
                              'C0': Calculus('rB_C'),
                              'C1': Calculus('0')},
                       'B1': {'B0': Calculus('0'),
                              'B1': Calculus('-rA_B - fB_C'),
                              'C0': Calculus('0'),
                              'C1': Calculus('rB_C')},
                       'C0': {'B0': Calculus('fB_C'),
                              'B1': Calculus('0'),
                              'C0': Calculus('-fC_A - rB_C'),
                              'C1': Calculus('0')},
                       'C1': {'B0': Calculus('0'),
                              'B1': Calculus('fB_C'),
                              'C0': Calculus('0'),
                              'C1': Calculus('-fC_A - rB_C')}}


    assert output == expected_output, `output, expected_output`

    
def test_hessian_construction_B():
    model, P, output, expected_output = test_model_construction_B()

    output = model.system_hessian()

    expected_output = {'B0': {'B0': Calculus('-rB_D - fAB_C*(A1 + A0)'),
                              'B1': Calculus('0'),
                              'C00': Calculus('rAB_C'),
                              'C01': Calculus('0'),
                              'C10': Calculus('rAB_C'),
                              'C11': Calculus('0'),
                              'D0': Calculus('fB_D'),
                              'D1': Calculus('0'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'B1': {'B0': Calculus('0'),
                              'B1': Calculus('-rB_D - fAB_C*(A1 + A0)'),
                              'C00': Calculus('0'),
                              'C01': Calculus('rAB_C'),
                              'C10': Calculus('0'),
                              'C11': Calculus('rAB_C'),
                              'D0': Calculus('0'),
                              'D1': Calculus('fB_D'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'C00': {'B0': Calculus('A0*fAB_C'),
                               'B1': Calculus('0'),
                               'C00': Calculus('-rAB_C - fC_DE'),
                               'C01': Calculus('0'),
                               'C10': Calculus('0'),
                               'C11': Calculus('0'),
                               'D0': Calculus('rC_DE*E0'),
                               'D1': Calculus('0'),
                               'E0': Calculus('rC_DE*D0'),
                               'E1': Calculus('0')},
                       'C01': {'B0': Calculus('0'),
                               'B1': Calculus('A0*fAB_C'),
                               'C00': Calculus('0'),
                               'C01': Calculus('-fC_DE - rAB_C'),
                               'C10': Calculus('0'),
                               'C11': Calculus('0'),
                               'D0': Calculus('0'),
                               'D1': Calculus('rC_DE*E0'),
                               'E0': Calculus('rC_DE*D1'),
                               'E1': Calculus('0')},
                       'C10': {'B0': Calculus('A1*fAB_C'),
                               'B1': Calculus('0'),
                               'C00': Calculus('0'),
                               'C01': Calculus('0'),
                               'C10': Calculus('-fC_DE - rAB_C'),
                               'C11': Calculus('0'),
                               'D0': Calculus('rC_DE*E1'),
                               'D1': Calculus('0'),
                               'E0': Calculus('0'),
                               'E1': Calculus('rC_DE*D0')},
                       'C11': {'B0': Calculus('0'),
                               'B1': Calculus('A1*fAB_C'),
                               'C00': Calculus('0'),
                               'C01': Calculus('0'),
                               'C10': Calculus('0'),
                               'C11': Calculus('-fC_DE - rAB_C'),
                               'D0': Calculus('0'),
                               'D1': Calculus('rC_DE*E1'),
                               'E0': Calculus('0'),
                               'E1': Calculus('rC_DE*D1')},
                       'D0': {'B0': Calculus('rB_D'),
                              'B1': Calculus('0'),
                              'C00': Calculus('fC_DE'),
                              'C01': Calculus('0'),
                              'C10': Calculus('fC_DE'),
                              'C11': Calculus('0'),
                              'D0': Calculus('-rC_DE*(E1 + E0) - fB_D'),
                              'D1': Calculus('0'),
                              'E0': Calculus('-rC_DE*D0'),
                              'E1': Calculus('-rC_DE*D0')},
                       'D1': {'B0': Calculus('0'),
                              'B1': Calculus('rB_D'),
                              'C00': Calculus('0'),
                              'C01': Calculus('fC_DE'),
                              'C10': Calculus('0'),
                              'C11': Calculus('fC_DE'),
                              'D0': Calculus('0'),
                              'D1': Calculus('-rC_DE*(E1 + E0) - fB_D'),
                              'E0': Calculus('-rC_DE*D1'),
                              'E1': Calculus('-rC_DE*D1')},
                       'E0': {'B0': Calculus('0'),
                              'B1': Calculus('0'),
                              'C00': Calculus('fC_DE'),
                              'C01': Calculus('fC_DE'),
                              'C10': Calculus('0'),
                              'C11': Calculus('0'),
                              'D0': Calculus('-rC_DE*E0'),
                              'D1': Calculus('-rC_DE*E0'),
                              'E0': Calculus('-(D0 + D1)*rC_DE - rA_E'),
                              'E1': Calculus('0')},
                       'E1': {'B0': Calculus('0'),
                              'B1': Calculus('0'),
                              'C00': Calculus('0'),
                              'C01': Calculus('0'),
                              'C10': Calculus('fC_DE'),
                              'C11': Calculus('fC_DE'),
                              'D0': Calculus('-rC_DE*E1'),
                              'D1': Calculus('-rC_DE*E1'),
                              'E0': Calculus('0'),
                              'E1': Calculus('-(D0 + D1)*rC_DE - rA_E')}}

    #for k, vd in output.items():
    #    for ik, v in vd.items():
    #        assert expected_output[k][ik].data == v.data, `expected_output[k][ik].data, v.data, k, ik`

def test_hessian_construction_BB():
    model, P, output, expected_output = test_model_construction_BB()

    output = model.system_hessian()

    expected_output = {'B0': {'B0': Calculus('-rB_D - fAB_C'),
                              'B1': Calculus('0'),
                              'C00': Calculus('rAB_C'),
                              'C01': Calculus('0'),
                              'C10': Calculus('rAB_C'),
                              'C11': Calculus('0'),
                              'D0': Calculus('fB_D'),
                              'D1': Calculus('0'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'B1': {'B0': Calculus('0'),
                              'B1': Calculus('-rB_D - fAB_C'),
                              'C00': Calculus('0'),
                              'C01': Calculus('rAB_C'),
                              'C10': Calculus('0'),
                              'C11': Calculus('rAB_C'),
                              'D0': Calculus('0'),
                              'D1': Calculus('fB_D'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'C00': {'B0': Calculus('A0*fAB_C'),
                               'B1': Calculus('0'),
                               'C00': Calculus('-rAB_C - fC_DE'),
                               'C01': Calculus('0'),
                               'C10': Calculus('0'),
                               'C11': Calculus('0'),
                               'D0': Calculus('rC_DE*E0'),
                               'D1': Calculus('0'),
                               'E0': Calculus('rC_DE*D0'),
                               'E1': Calculus('0')},
                       'C01': {'B0': Calculus('0'),
                               'B1': Calculus('A0*fAB_C'),
                               'C00': Calculus('0'),
                               'C01': Calculus('-fC_DE - rAB_C'),
                               'C10': Calculus('0'),
                               'C11': Calculus('0'),
                               'D0': Calculus('0'),
                               'D1': Calculus('rC_DE*E0'),
                               'E0': Calculus('rC_DE*D1'),
                               'E1': Calculus('0')},
                       'C10': {'B0': Calculus('A1*fAB_C'),
                               'B1': Calculus('0'),
                               'C00': Calculus('0'),
                               'C01': Calculus('0'),
                               'C10': Calculus('-fC_DE - rAB_C'),
                               'C11': Calculus('0'),
                               'D0': Calculus('rC_DE*E1'),
                               'D1': Calculus('0'),
                               'E0': Calculus('0'),
                               'E1': Calculus('rC_DE*D0')},
                       'C11': {'B0': Calculus('0'),
                               'B1': Calculus('A1*fAB_C'),
                               'C00': Calculus('0'),
                               'C01': Calculus('0'),
                               'C10': Calculus('0'),
                               'C11': Calculus('-fC_DE - rAB_C'),
                               'D0': Calculus('0'),
                               'D1': Calculus('rC_DE*E1'),
                               'E0': Calculus('0'),
                               'E1': Calculus('rC_DE*D1')},
                       'D0': {'B0': Calculus('rB_D'),
                              'B1': Calculus('0'),
                              'C00': Calculus('fC_DE'),
                              'C01': Calculus('0'),
                              'C10': Calculus('fC_DE'),
                              'C11': Calculus('0'),
                              'D0': Calculus('-rC_DE - fB_D'),
                              'D1': Calculus('0'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'D1': {'B0': Calculus('0'),
                              'B1': Calculus('rB_D'),
                              'C00': Calculus('0'),
                              'C01': Calculus('fC_DE'),
                              'C10': Calculus('0'),
                              'C11': Calculus('fC_DE'),
                              'D0': Calculus('0'),
                              'D1': Calculus('-rC_DE - fB_D'),
                              'E0': Calculus('0'),
                              'E1': Calculus('0')},
                       'E0': {'B0': Calculus('0'),
                              'B1': Calculus('0'),
                              'C00': Calculus('fC_DE'),
                              'C01': Calculus('fC_DE'),
                              'C10': Calculus('0'),
                              'C11': Calculus('0'),
                              'D0': Calculus('0'),
                              'D1': Calculus('0'),
                              'E0': Calculus('-rC_DE - rA_E'),
                              'E1': Calculus('0')},
                       'E1': {'B0': Calculus('0'),
                              'B1': Calculus('0'),
                              'C00': Calculus('0'),
                              'C01': Calculus('0'),
                              'C10': Calculus('fC_DE'),
                              'C11': Calculus('fC_DE'),
                              'D0': Calculus('0'),
                              'D1': Calculus('0'),
                              'E0': Calculus('0'),
                              'E1': Calculus('-rC_DE - rA_E')}}

    dd = DictDiffer(output, expected_output)
    print dd.changed()
    print dd.unchanged()
    assert False

class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect 
    def removed(self):
        return self.set_past - self.intersect 
    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])
