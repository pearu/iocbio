#
# Author: David Schryer
# Created: March 2011

import sys
import time
import numpy
import textwrap
import argparse 

from mytools.tools import drop_to_ipython as dti

from scipy import integrate as scipy_integrate

from sympycore import Symbol as sympycore_Symbol

from builder import pf

class IsotopologueSolver(object):

    def __init__(self, model):
        
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
        self.slope_index = 1e10

    def set_data(self, independent_flux_dic=None, exchange_flux_dic=None, pool_dic=None, solution_name=None,
                 verbose=True, write_input_to_file=True):

        if solution_name is None:
            solution_name = 1
        

        ifd = self.independent_flux_dic = independent_flux_dic
        efd = self.exchange_flux_dic = exchange_flux_dic
        pd = self.pool_dic = pool_dic

        nfd = self.net_flux_dic = self._compute_steady_fluxes(ifd, verbose=verbose)                 
        fd = self.flux_dic = self._make_flux_dic(nfd, efd)
        self._set_data(fd, pd)
        
        self.input_variables = iv = self.input_variables = dict(independent_flux_dic=independent_flux_dic,
                                                                exchange_flux_dic=exchange_flux_dic,
                                                                pool_dic=pool_dic,
                                                                net_flux_dic=nfd,
                                                                flux_dic=fd)
        self.solution_name = solution_name
        if write_input_to_file:
            output_string = pf(iv)
            fn = '{0.import_dir}/{0.model_name}_solution_{1}_input.py'.format(self.model, solution_name)
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

    def solve(self, integrator_parameters=None, initial_time_step=10, end_time=100,
              verbose=True, very_verbose=False, write_output_to_file=True): 

        output_file_tmpl = '{0.model_name}_solution_{1}_output'.format(self.model, self.solution_name)
        output_file_name = '{0.import_dir}/{1}.py'.format(self.model, output_file_tmpl)

        i_str = 'solution_list, time_list'
        import_string = 'from {0.import_dir}.{1} import {2}'
        #try:
        #    exec(import_string.format(self.model, output_file_tmpl, i_str))
        #    return solution_list, time_list
        #except ImportError, msg:
        #    pass

        if very_verbose:
            ds = pf(self.data)
            print "Solving with the following input data:\n{0}\n".format(ds)
        
        sof = self.solver_output_file = open('{0.import_dir}/{0.model_name}_VODE_output'.format(self.model), 'a')
        iil = self.data['input_list']

        if 0:
            int_params = dict(method='bdf', 
                              rtol=1e-12, 
                              atol=1e-12, 
                              #order=3, 
                              nsteps=2000000, 
                              #max_step=0.1, 
                              #min_step=1e-8,
                              with_jacobian=False)
        else:
            int_params = dict(method='adams', 
                              rtol=1e-12, 
                              atol=1e-12, 
                              #order=3, 
                              nsteps=2000000, 
                              #max_step=None, 
                              #min_step=None,
                              with_jacobian=False)
            
        if integrator_parameters:
            int_params.update(integrator_parameters) 

        print 'Integrator parameters:', int_params

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
        if verbose:
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

                #if solve_time > 10:
                #    initial_time_step = 0.2
                #if solve_time > 20:
                #    initial_time_step = 0.5
                #if solve_time > 40:
                #    initial_time_step = 1
                #if solve_time > 80:
                #    initial_time_step = 2
                #if solve_time > 120:
                #    initial_time_step = 4


                cpu_now = time.time()
                if cpu_now - cpu_previous > 10 and verbose:
                    cpu_previous = cpu_now
                    data = (solve_time, '{0:0.0f}'.format(cpu_now-cpu_start), self.slope_index, self.max_derivative)
                    status_string = format_string.format(*data)
                    print status_string

                if very_verbose:
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

        if write_output_to_file:
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
   
