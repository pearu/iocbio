#
# Author: David Schryer
# Created: March 2011
import os
import sys
import time
import numpy
import scipy
import textwrap
from sympycore import Symbol as sympycore_Symbol
import cPickle as pickle
from scipy import integrate as scipy_integrate
from collections import defaultdict
import argparse 

from tools import pp, pf, \
                  round_float, make_solution_name, decode_solution_name, \
                  get_ef_CK_from_frac_ATP, test_arguments, \
                  add_model_parameter_arguments, \
                  move_grid_job_files

def make_argument_parser():
    '''Returns options parser for this script.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''
                                     This script finds one or more solutions to an IsotopeModel definition
                                     ======================================================================
                                     It first sets the parameters for each of the solutions and then solves
                                     them.  Each set of parameters forms a unique solution_name under which
                                     the input parameters and output are written. The parameters are either
                                     F (float) or S (string).
                                     '''))

    fg = parser.add_argument_group('Flags')
    spg = parser.add_argument_group('Solving parameters')
    ifpg = parser.add_argument_group('Input file parameters')
    ofpg = parser.add_argument_group('Output file parameters')
    mpg = parser.add_argument_group('Model parameters')
    
    d = '[default:%(default)s]'
    c = '[choices:%(choices)s]'
    FL = 'F'
    ST = 'S'

    mpg = add_model_parameter_arguments(mpg)
    
    spg.add_argument("-X", "--x-max",
                     dest="x_max", default=610, metavar=FL, type=float,
                     help="Flag to specify the time to stop the solution. {0}".format(d))
    
    fg.add_argument("-v", "--verbose", action='store_true',
                     dest="verbose", default=True, 
                     help="Flag to specify if the solver should be verbose. {0}".format(d))
    
    fg.add_argument("-V", "--very-verbose", action='store_true',
                     dest="very_verbose", default=False, 
                     help="Flag to specify if the solver should be VERY verbose. {0}".format(d))
    
    fg.add_argument("-P", "--prep-grid-job", action='store_true',
                    dest="prepare_grid_job", default=False, 
                    help='Prepare the parameters for a grid job. {0}'.format(d))
    
    fg.add_argument("-R", "--run-grid-job", action='store_false',
                    dest="run_grid_job", default=True, 
                    help='Runs a job with the grid engine. Stores false. {0}'.format(d))
    
    fg.add_argument("-o", "--only-set-data", action='store_true', 
                    dest="only_set_data", default=False, 
                    help="Only set the data and write the input files.  No solve is attempted. {0}".format(d))
    
    fg.add_argument("-w", "--write-output", action='store_true',
                    dest="write_output_to_file", default=True, 
                    help="Flag to write the output to a file after solving. {0}".format(d))
    
    fg.add_argument("-W", "--write-input", action='store_true',
                    dest="write_input_to_file", default=True, 
                    help="Flag to write the input to a file. Required if one wants to solve the model. {0}".format(d))

    ofpg.add_argument("-D", "--save-dir",
                      dest="save_dir", default='solutions', metavar=ST,
                      help="Flag to specify the directory where solutions and input files are saved to. {0}".format(d))
    
    ifpg.add_argument("-d", "--import-dir",
                      dest="import_dir", default='local', metavar=ST,
                      help="Flag to specify the import_dir for the solution input file. {0}".format(d))

    ifpg.add_argument("-m", "--model-name",
                      dest="model_name", default='model_8_mass', metavar=ST,
                      help="Flag to specify the model_name of the solution input file. {0}".format(d))

    return parser

class IsotopomerSolver(object):

    def __init__(self):
        from oxygen_isotope_model import model
        
        import_string = 'import {0.import_dir}.{0.model_name} as c_package'.format(model)
        print import_string
        try:
            exec(import_string)
        except ImportError:
            model.compile_ccode(debug=False, stage=None)
            try:
                exec(import_string)
            except ImportError, msg:
                raise UserWarning(msg)

        import_string = 'from {0.import_dir}.{0.model_name}_c_variables import c_variables'.format(model)
        print import_string
        try:
            exec(import_string)
        except ImportError, msg:
            raise UserWarning(msg)

        if not c_variables or not c_package:
            msg = 'c_package or c_variables could not be imported.'
            raise UserWarning(msg)

        self.model = model
        self.function = c_package.c_equations
        self.c_variables = c_variables

    def set_data(self, args, independent_flux_dic=None, exchange_flux_dic=None, pool_dic=None, solution_name=None):

        if solution_name is None:
            solution_name = 1

        ifd = self.independent_flux_dic = independent_flux_dic
        efd = self.exchange_flux_dic = exchange_flux_dic
        pd = self.pool_dic = pool_dic

        nfd = self.net_flux_dic = self._compute_steady_fluxes(ifd, verbose=args.verbose)                 
        fd = self.flux_dic = self._make_flux_dic(nfd, efd)
        self._set_data(fd, pd)
        
        self.input_variables = iv = self.input_variables = dict(independent_flux_dic=independent_flux_dic,
                                                                exchange_flux_dic=exchange_flux_dic,
                                                                pool_dic=pool_dic,
                                                                net_flux_dic=nfd,
                                                                flux_dic=fd)
        if args.write_input_to_file:
            output_string = pf(iv)
            fn = '{0.import_dir}/{0.model_name}_solution_{1}_input.py'.format(args, solution_name)
            print 'Writing input file to', fn
            f = open(fn, 'w')
            f.write('input_variables = {0}\n'.format(output_string))
            f.close()

    def _set_data(self, fluxes, pools):
        flux_list = self.c_variables['flux_list']
        pool_list = self.c_variables['pool_list']
        input_list = self.c_variables['input_list']
        out = self.c_variables['out']

        data = {}
        data['pool_list'] = [pools[p] for p in pool_list]        
        data['flux_list'] = [fluxes[f] for f in flux_list]
        if self.model.use_sum:
            data['input_list'] = [0 for n in input_list]
        else:
            if self.model.use_mass:
                data['input_list'] = [int(not sum(map(int,n.split ('_')[1:]))) for n in input_list]
            else:
                def get_index (n):
                    i = 0
                    while not n[i].isdigit ():
                        i += 1
                    return n[i:]
                data['input_list'] = [int(sum(map(int, get_index(n).split ('_')))==0) for n in input_list]

        data['input_list_variables'] = input_list
        self.data = data

    def VODE_equations(self, solver_time, input_list):
        fl = self.data['flux_list']
        pl = self.data['pool_list']        

        out = self.function(pl, fl, solver_time, input_list)

        self.equation_call_counter += 1

        cpu_now = time.time()
        if self.first_solver_call:
            self.first_solver_call = False
            self.max_derivative = numpy.abs(out).max()
            self.slope_index = numpy.array(numpy.array(out) + 1).prod()
            self.cpu_time = cpu_now

        elif cpu_now - self.cpu_time > 0.1:
            self.max_derivative = numpy.abs(out).max()
            self.slope_index = numpy.array(numpy.array(out) + 1).prod()
            self.cpu_time = cpu_now
            
        #n = 0
        #for i in [2,2,2,4,4,4,3,3]:
        #    print out[n:n+i].sum(),
        #    n = n+i
        #print out.sum()
        return out

    def solve(self, args, solution_name=None, integrator_params=None, initial_time_step=0.1, end_time=None): 

        if end_time is None:
            end_time = args.x_max

        output_file_tmpl = '{0.model_name}_solution_{1}_output'.format(args, solution_name)
        output_file_name = '{0.import_dir}/{1}.py'.format(args, output_file_tmpl)

        i_str = 'solution_list, time_list'
        import_string = 'from {0.import_dir}.{1} import {2}'
        try:
            exec(import_string.format(args, output_file_tmpl, i_str))
            return solution_list, time_list
        except ImportError, msg:
            pass

        if args.very_verbose:
            ds = pf(self.data)
            print "Solving with the following input data:\n{0}\n".format(ds)
        
        sof = self.solver_output_file = open('{0.import_dir}/{0.model_name}_VODE_output'.format(args), 'a')
        iil = self.data['input_list']

        int_params = dict(method='bdf', 
                         rtol=1e-12, 
                         atol=1e-12, 
                         #order=3, 
                         nsteps=2000000, 
                         #max_step=0.1, 
                         #min_step=1e-8,
                         with_jacobian=False)

        if integrator_params:
            int_params.update(integrator_params) 

        # Create SciPy VODE integrator object.
        integrator_object = scipy_integrate.ode(self.VODE_equations, jac=None)

        # Choose the VODE integration routine with the appropriate solver.
        save_stdout = sys.stdout
        fsock = open('/dev/null', 'w')
        sys.stdout = fsock
        integrator_object.set_integrator('vode', **int_params)
        sys.stdout = save_stdout
        del fsock

        # Set initial value of integrator.
        integrator_object.set_initial_value(iil)

        solution_list = []
        solved_time_list = []

        solve_time = initial_time_step

        cpu_start = time.time()
        cpu_previous = cpu_start
        self.equation_call_counter = 0
        self.first_solver_call = True
        if args.verbose:
            format_string = '{0:<10} {1:<10} {2:<20} {3:<20}'
            titles = ('int_time', 'cpu_time', 'slope_index', 'max_derivative')
            print format_string.format(*titles)
            
        while True:
            if integrator_object.successful():  
                try:
                    sub_list = integrator_object.integrate(integrator_object.t + initial_time_step)
                except KeyboardInterrupt:
                    print 'Computation canceled by user with Ctrl-C'
                    has_solution = True
                    break
                solve_time = integrator_object.t

                if solve_time > 10:
                    initial_time_step = 0.2
                if solve_time > 20:
                    initial_time_step = 0.5
                if solve_time > 40:
                    initial_time_step = 1
                if solve_time > 80:
                    initial_time_step = 2
                if solve_time > 120:
                    initial_time_step = 4


                cpu_now = time.time()
                if cpu_now - cpu_previous > 10 and args.verbose:
                    cpu_previous = cpu_now
                    data = (solve_time, '{0:0.0f}'.format(cpu_now-cpu_start), self.slope_index, self.max_derivative)
                    status_string = format_string.format(*data)
                    print status_string

                if args.very_verbose:
                    print solve_time,
                    sys.stdout.flush()
                                
                solution_list.append(sub_list)
                solved_time_list.append(solve_time)

                slope_ok = numpy.round(self.slope_index, decimals=2) == 1.0

                # if self.max_derivative < 1e-7:
                #     msg = "Breaking with max_derivative {0} < 1e-7"
                #     print msg.format(self.max_derivative)
                #     has_solution = True
                #     break
                
                # elif slope_ok and self.max_derivative < 1e-6:
                #     msg = "Breaking with slope_ok {0} and max_derivative {1} < 1e-6"
                #     print msg.format(slope_ok, self.max_derivative)
                #     has_solution = True
                #     break

                if solve_time > int(end_time):
                    msg = "Lets stop now"
                    print msg, solve_time
                    has_solution = True
                    break
                
                elif max (sub_list) > 10 or min (sub_list)<-10:
                    print 'Going nuts..'
                    has_solution = True
                    break
                
            else:
                msg = "integrator_object.successful() == False"
                sof.write(msg)
                print msg
                has_solution = False
                break
            
        solution_list.insert(0, iil)
        solved_time_list.insert(0, 0.0)
        solution_list = numpy.array(solution_list)

        if args.write_output_to_file:
            print 'Writing output file to', output_file_name
            f = open(output_file_name, 'w')
            f.write('solution_list = {0}\ntime_list = {1}\n'.format(pf(solution_list.tolist()), pf(solved_time_list)))
            f.close()
            
        return solution_list, solved_time_list

    def _compute_steady_fluxes(self, indep_dic, verbose=False):
        model = self.model

        dep_candidates = []
        for r_key in model.flux_analyzer.reactions:
            if r_key not in indep_dic:
                dep_candidates.append(r_key)
                
        model.flux_analyzer.compute_kernel_GJE(dependent_candidates=dep_candidates)

        fluxes, indep_fluxes, kernel = model.flux_analyzer.get_kernel_GJE()

        print 'indep_dic keys', indep_dic.keys()
        print 'indep_fluxes  ', indep_fluxes
        
        dep_fluxes = fluxes[:model.flux_analyzer.rank]
        indep_symbols = map(sympycore_Symbol,indep_fluxes)
        print '\nSteady state relations:'
        for i in range(model.flux_analyzer.rank):
            print dep_fluxes[i],'=',[indep_symbols] * kernel[i].T

        if verbose:
            print '\nSteady state solution:'
            print model.flux_analyzer.label_matrix (kernel, fluxes, indep_fluxes)

        for flux_name in indep_fluxes:
            if flux_name not in indep_dic:
                raise ValueError ('must define the value of %r in independent flux dictionary' % (flux_name))        

        net_flux_dic = {}
        for i, flux_name in enumerate(fluxes):
            v = 0
            for j, indep_flux_name in enumerate(indep_fluxes):
                k = kernel[i,j]
                if k:
                    v += k * indep_dic[indep_flux_name]
            net_flux_dic[flux_name] = v

        return net_flux_dic

    def _make_flux_dic(self, net_flux_dic, exchange_flux_dic):
        model = self.model
        
        flux_dic = {}
        for rxn_key, info_dic in model.flux_analyzer.reactions_info.items():
            f_key = info_dic['forward']
            r_key = info_dic['reverse']
            nf = net_flux_dic[rxn_key]
            if not info_dic['reversible']:
                assert nf > 0
                flux_dic[f_key] = nf
                continue

            ef = exchange_flux_dic[rxn_key]
            assert ef >= 0
            if nf > 0:
                flux_dic[f_key] = nf + ef
                flux_dic[r_key] = ef
            elif nf < 0:
                flux_dic[r_key] = -nf + ef
                flux_dic[f_key] = ef
            elif nf == 0:
                flux_dic[r_key] = ef
                flux_dic[f_key] = ef

        return flux_dic
   

def make_input_dic(args):
    test_arguments(args)
    
    indep_flux_dic = dict(ASs=args.AS_flux,
                          ATPio=args.frac_ATP*args.AS_flux,
                          Cio=args.frac_CK*args.AS_flux,
                          )

    # This case specifies that we should use Marko's CK fluxes.
    if args.ef_CKi is None and args.ef_CKo is None:
        args = get_ef_CK_from_frac_ATP(args)
    #assert args.bidirectional_transport_case == 'biB',`args.bidirectional_transport_case`

    if args.bidirectional_transport_case == 'biA': # ignore
        ef_dic = dict(ATPio=0.3,
                      Cio=2.00,
                      ADPoi=0.02,
                      ATPmi=0.01,
                      Pom=0,
                      ADPim=0.01,
                      Weo=0,
                      Wos=0,
                      Peo=0,
                      Pms=0,
                      ADPeo=0,
                      ADPms=0,
                      )
    elif args.bidirectional_transport_case == 'biB':
        ef_dic = dict(ATPio=0.3,
                      Cio=2.00,
                      ADPoi=0.02,
                      ATPmi=0.01,
                      Pom=0,
                      ADPim=0.01,
                      Weo=5,
                      Wos=5,
                      Peo=5,
                      Pms=5,
                      ADPeo=5,
                      ADPms=5,
                      )
    elif args.bidirectional_transport_case == 'biC':
        ef_dic = dict(ATPio=0.3,
                      Cio=2.00,
                      ADPoi=0.02,
                      ATPmi=0.01,
                      Pom=0,
                      ADPim=0.01,
                      Weo=50,
                      Wos=50,
                      Peo=50,
                      Pms=50,
                      ADPeo=50,
                      ADPms=50,
                      )
        
    elif args.bidirectional_transport_case == 'uniA':
        ef_dic = dict(ATPio=0.01,
                      Cio=0.01,
                      ADPoi=0.01,
                      ATPmi=0.001,
                      Pom=0,
                      ADPim=0.001,
                      Weo=0,
                      Wos=0,
                      Peo=0,
                      Pms=0,
                      ADPeo=0,
                      ADPms=0,
                      )
    elif args.bidirectional_transport_case == 'uniB': # ignore
        ef_dic = dict(ATPio=0.01,
                      Cio=0.01,
                      ADPoi=0.01,
                      ATPmi=0.001,
                      Pom=0,
                      ADPim=0.001,
                      Weo=5,
                      Wos=5,
                      Peo=5,
                      Pms=5,
                      ADPeo=5,
                      ADPms=5,
                      )
    elif args.bidirectional_transport_case == 'uniC': # ignore
        ef_dic = dict(ATPio=0.01,
                      Cio=0.01,
                      ADPoi=0.01,
                      ATPmi=0.001,
                      Pom=0,
                      ADPim=0.001,
                      Weo=50,
                      Wos=50,
                      Peo=50,
                      Pms=50,
                      ADPeo=50,
                      ADPms=50,
                      )

    ef_dic['CKo'] = args.ef_CKo
    ef_dic['CKi'] = args.ef_CKi
    ef_dic['AKo'] = args.ef_AKo
    ef_dic['AKi'] = args.ef_AKi
    ef_dic['ASe'] = args.ef_ASe
    ef_dic['ASs'] = args.ef_ASs

    if args.pool_case == 'base':

        total_CP = 16.4        # +/- 2.44
        total_ATP = 7.55       # +/- 1.13
        total_P = 1.41         # +/- 0.78

        total_ADP = total_ATP*0.01  # OK

        frac_CP_in_i = 0.01 # OK
        frac_ATP_in_i = 0.01 # OK
        frac_ATP_in_e = 0.0005 # OK
        frac_ATP_in_s = 0.0005 # OK
        frac_ATP_in_mito = 0.125 # OK
        frac_ATP_in_cytosol = 1 - (frac_ATP_in_mito + frac_ATP_in_i + frac_ATP_in_e + frac_ATP_in_s) # OK
        frac_P_in_mito = 0.125 # fixed - OK
        frac_P_in_e = 0.0005 # OK
        frac_P_in_s = 0.0005 # OK
        frac_ADP_in_i = 0.01 # OK
        frac_ADP_in_e = 0.01 # OK  Minimum pool size for enzyme bound state
        frac_ADP_in_s = 0.01 # OK  Minimum pool size for enzyme bound state
        frac_ADP_in_mito = 0.125 # OK
        frac_ADP_in_cytosol = 1 - (frac_ADP_in_mito + frac_ADP_in_i + frac_ADP_in_e + frac_ADP_in_s) # OK

        pool_dic = dict(CPi=total_CP*frac_CP_in_i,
                        CPo=total_CP*(1 - frac_CP_in_i),
                        ADPi=total_ADP*frac_ADP_in_i,
                        ADPo=total_ADP*frac_ADP_in_cytosol,
                        ADPm=total_ADP*frac_ADP_in_mito, # fixed
                        ADPe=total_ADP*frac_ADP_in_e, # fixed
                        ADPs=total_ADP*frac_ADP_in_s, # fixed
                        ATPi=total_ATP*frac_ATP_in_i,
                        ATPe=total_ATP*frac_ATP_in_e,
                        ATPs=total_ATP*frac_ATP_in_s,
                        ATPo=total_ATP*frac_ATP_in_cytosol,
                        ATPm=total_ATP*frac_ATP_in_mito,
                        Po=total_P*(1 - frac_P_in_mito - frac_P_in_e - frac_P_in_s), # fixed
                        Pm=total_P*frac_P_in_mito,
                        Pe=total_P*frac_P_in_e,
                        Ps=total_P*frac_P_in_s,
                        We=0.0005,
                        Ws=0.0005,
                        ) # OK
        #pp(pool_dic)
    
    solution_name = make_solution_name(args)
    
    input_dic = dict(pool_dic=pool_dic,
                     exchange_flux_dic=ef_dic,
                     independent_flux_dic=indep_flux_dic,
                     solution_name=solution_name)

    return input_dic

def prepare_grid_job(args):
    ef_AK_values = scipy.linspace(0, 3.0, 31)    
    frac_ATP_values = scipy.linspace(0, 1.0, 21)
    frac_AK_values = scipy.linspace(0, 0.3, 11)

    ef_AK_values = scipy.linspace(0, 3.0, 3)    
    frac_ATP_values = scipy.linspace(0, 1.0, 2)
    frac_AK_values = scipy.linspace(0, 0.3, 1)
        
    task_list = []
    for ef_AKi in ef_AK_values:
        for ef_AKo in ef_AK_values:
            job_list = []
            for frac_ATP in frac_ATP_values:
                for frac_AK in frac_AK_values:
                    tol = 1e-5
                    if frac_AK + frac_ATP > 1.0 + tol:
                        continue

                    args.frac_ATP = frac_ATP
                    args.frac_AK = frac_AK
                    args.frac_CK = 1 - (frac_ATP + frac_AK) 
                    for CK_bi in [True, False]:
                        if CK_bi:
                            args = get_ef_CK_from_frac_ATP(args)
                        else:
                            args.ef_CKi = 0
                            args.ef_CKo = 0

                        sn = make_solution_name(args)
                        job_list.append(sn)                                       
                        
            task_list.append(job_list)
            
    fn = '{0.import_dir}/grid_parameters.py'.format(args)
    f = open(fn, 'w')
    f.write('grid_parameters = {0}\n'.format(pf(task_list)))
    f.close()
    print "qsub -V -cwd -t 1:{0} /usr/local/bin/python.sh solve_isotope_model.py".format(len(task_list))

def run_grid_job(args):
    
    taskid = int(os.environ['SGE_TASK_ID'])
    jobid = int(os.environ['JOB_ID'])
    f = open('output.txt', 'a')
    f.write('Running grid job with SGE_TASK_ID={0} and JOB_ID={1}\n'.format(taskid, jobid))
    
    
    import_string = 'from {0.import_dir}.grid_parameters import grid_parameters'.format(args)
    f.write('{0}\n'.format(import_string))
    f.close()
    
    try:
        exec(import_string)
    except ImportError, msg:
        raise UserWarning(msg)

    args.x_max = 610
    
    job_list = grid_parameters[taskid - 1]
        
    snl = []
    for sn in job_list:
        snl.append(sn)

    args.solution_names = snl
    
    return args
    
def process_arguments(args):

    if args.prepare_grid_job:
        prepare_grid_job(args)
        exit()

    if args.run_grid_job:
        args = run_grid_job(args)
            
    if args.solution_names is not None:
        for sn in args.solution_names:
            print sn
            args = decode_solution_name(args, sn)

            input_dic = make_input_dic(args)
            solver = IsotopomerSolver()
            solver.set_data(args, **input_dic)
            if not args.only_set_data:
                sl, stl = solver.solve(args, solution_name=input_dic['solution_name'])

        if args.run_grid_job:
            move_grid_job_files()
    else:
        input_dic = make_input_dic(args)    
        solver = IsotopomerSolver()
        solver.set_data(args, **input_dic)
        if not args.only_set_data:
            sl, stl = solver.solve(args, solution_name=input_dic['solution_name'])

if __name__ == '__main__':

    p = make_argument_parser()
    args = p.parse_args()
    process_arguments(args)
