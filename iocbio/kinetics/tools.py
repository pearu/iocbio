#
# Author: David Schryer
# Created: May 2011

import pprint

def pp(item):
    print pprint.pformat(item)

def pf(item):
    return pprint.pformat(item)

import os
import time
import itertools
import numpy
import subprocess

from collections import defaultdict
from matplotlib import pyplot
from matplotlib import lines as mpl_lines
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Rectangle
from argparse import Namespace


sl = mpl_lines.Line2D.filled_markers
fl = mpl_lines.Line2D.fillStyles
cl = ['Indigo', 'CornflowerBlue', 'Gold', 'DarkGreen', 'LimeGreen', 
      'HotPink', 'OrangeRed', 'Crimson', 'Black']


def perform_latex_substitution(met):

    def mk(c):
        n = r'\Oba'
        f = ''
        for i in c:
            n += 'r'
            if i == '0':
                f += r'\circ'
            elif i == '1':
                f += r'\bullet'
            else:
                msg = "The isotopomer code must contain either a '0' or a '1'."
                raise UserWarning(msg, (i))
        return n + f

    m = str(met)
    if m.startswith('ATP') or m.startswith('ADP'):
        i = 4
    elif m.startswith('CP'):
        i = 3
    elif m.startswith('W') or m.startswith('P'):
        i = 2
    else:
        msg = "This metabolite has not yet been implimented."
        raise NotImplementedError(msg, (m))

    mn = r'\{0}name'.format(met[:i])
    c = met[i:]

    for i in c.split('_'):
        mn += mk(i)
    return mn

def make_terms(model):
    td = []
    for r in model._kinetic_terms:
        for (k, reactants), vl in r.items():
            rs = ''
            for reactant in reactants:
                rs += perform_latex_substitution(reactant) + '+'
            rs = rs[:-1]
            for products in vl:
                ps = ''
                for product in products:
                    ps += perform_latex_substitution(product) + '+'
                ps = ps[:-1]
                td.append((model.latex_name_map[k], rs, ps))
    return sorted(td)

def make_kinetic_terms(model):
    terms = make_terms(model)

    s = r'\section*{hello}' + '\n'
    s += r'\begin{align*}' + '\n' + r'\cee{'
    for index, t in enumerate(terms):
        wrap = index%4
        if wrap == 0:
            eq = r'&'
        if wrap == 1 or wrap == 2:
            eq += ' {0} ->C[{1}] {2} & \n'.format(t[1], t[0], t[2])
        if wrap == 3:
            if index == len(terms)-1:
                eq += ' {0} ->C[{1}] {2} }} \n'.format(t[1], t[0], t[2])
            else:
                eq += ' {0} ->C[{1}] {2} \\\\ \n'.format(t[1], t[0], t[2])

            s += eq
    s += r'\end{align*}' + '\n'
    return s


def get_index(time, time_list):
    tol = 0.1
    for t in time_list:
        if abs(t - time) < tol:
            return time_list.index(t)

    msg = "An index was not found with the given input."
    raise UserWarning(msg, (tol, time, time_list))

def calculate_errors(s_list, sp_list, time_list, pd):
    pool_dic = {}
    pool_dic['P'] = pd['Po'] + pd['Pm']
    pool_dic['CP'] = pd['CPo'] + pd['CPi']
    pool_dic['ATP'] = pd['ATPo'] + pd['ATPi'] + pd['ATPm']

    error_dic = {}
    for sp in sp_list:
        met_key = '_'.join(sp.split('_')[:-1])
        if met_key not in ['P', 'CP', 'ATP_g', 'ATP_b']:
            continue
        measured_dic = get_measured_data(met_key)
        predicted_dic = get_predicted_data(s_list, sp_list, time_list, measured_dic, met_key)       

        e_dic = {}
        for time, mv in measured_dic.items():
            e_dic[time] = predicted_dic[time] - mv
        error_dic[met_key] = e_dic

    return error_dic

def get_predicted_data(s_list, sp_list, time_list, measured_dic, met_key):
    data_dic = {}
    for sp in sp_list:
        if met_key != '_'.join(sp.split('_')[:-1]):
            continue
        sp_index = sp_list.index(sp)
        time_pt_dic = {}
        for t in measured_dic.keys():
            d_index = get_index(t, time_list)
            data = s_list[sp_index][d_index]
            time_pt_dic[t] = data
        data_dic[met_key] = time_pt_dic

    if data_dic == dict():
        msg = "This metabolite was not found in the species list. (The last _1 is removed)"
        raise UserWarning(msg, (met_key, sp_list))

    if not data_dic.has_key(met_key):
        msg = "The predicted_dic does not contain this met_key."
        raise UserWarning(msg, (met_key, data_dic))

    return data_dic[met_key]

def get_measured_data(met_name, return_plot_lists=False):

    #nm per Oxygen in each P per mg protein.
    labeling_dic = dict(P={5:11.4, 15:29.6, 30:51.7},
                        CP={5:6.5, 15:19.8, 30:39.0},
                        ATP_g={5:8.7, 15:25.0, 30:45.7},
                        ATP_b={5:1.9, 15:7.3, 30:14.7},
                        )

    # nm per mg protein.
    pool_dic = dict(P=28,
                    CP=37,
                    ATP=26,
                    )

    percentage_dic = {}
    for k, data_dic in labeling_dic.items():
        inner_dic = {}
        if len(k.split('_')) == 2:
            pool = pool_dic['ATP']
        else:
            pool = pool_dic[k]
            
        for time, v in data_dic.items():
            if k == 'P':
                nO = 4
            else:
                nO = 3
            inner_dic[time] = (v / float(nO)) / pool
        percentage_dic[k] = inner_dic

    if met_name not in percentage_dic.keys():
        msg = "This metabolite name does not have any data associated with it."
        raise UserWarning(msg, (met_name, percentage_dic))

    data_dic = percentage_dic[met_name]
    if not return_plot_lists:
        return data_dic
    
    x_list = []
    y_list = []
    for k, v in data_dic.items():
        x_list.append(k)
        y_list.append(v)
    return x_list, y_list

def make_plot_list(args, x_list, y_list, species_list):

    plot_list = []
    for index, species in enumerate(species_list):
        species_name = species[:-2]
        
        elements = species.split('_')
        met_name = elements[0]
        if len(elements) == 4:
            met_name += '_' + elements[1]
        if met_name not in ['P', 'ATP_b', 'ATP_g', 'CP']:
            continue
        
        series = int(elements[-1])
        nlabels = int(elements[-2])

        colour = cl[series - 1] 
        symbol = sl[nlabels]
        fill = fl[0]

        #for e in y_list:
        #    print len(e),

        solution = numpy.array(y_list)[index]
        if solution.max() < 1e-2:
            continue
        solution = solution.tolist()

        if args.plot_type == 'tr':
            plot_list += [(met_name, series, nlabels, colour, symbol, fill, x_list[index], solution)]
        else:
            plot_list += [(met_name, series, nlabels, colour, symbol, fill, x_list[index], solution)]

    return plot_list

def slice_soln(args, series, nlabels, x_values, solution):

    if args.plot_type == 'tr':
        tol = 1e-5
        if abs(args.x_min - 100) < tol:
            x_lists = [[130, 500],
                       [140, 510],
                       [150, 520],
                       [160, 530],
                       [170, 540],
                       [180, 550],
                       [190, 560],
                       [200, 570],
                       [210, 580],
                       [220, 590],
                       ]

        else:
            x_lists = [[5, 50],
                       [10, 55],
                       [15, 60],
                       [20, 65],
                       [25, 70],
                       [30, 75],
                       [35, 80],
                       [40, 85],
                       [45, 90],
                       ]
    elif args.plot_type in ['acer']:
        x_lists = [[0.05, 0.6],
                   [0.1, 0.65],
                   [0.15, 0.7],
                   [0.2, 0.75],
                   [0.25, 0.8],
                   [0.3, 0.85],
                   [0.35, 0.9],
                   ]                
    elif args.plot_type in ['ckn', 'cer']:
        x_lists = [[1.2, 4.0],
                   [1.4, 4.2],
                   [1.6, 4.4],
                   [1.8, 4.6],
                   [2.0, 4.8],
                   [2.2, 5.0],
                   [2.4, 5.2],
                   [2.6, 5.4],
                   ]
    elif args.plot_type in ['aer']:
        x_lists = [[0.01, 0.19],
                   [0.02, 0.195],
                   [0.03, 0.2],
                   [0.04, 0.205],
                   [0.05, 0.21],
                   [0.06, 0.215],
                   [0.07, 0.22],
                   [0.08, 0.225],
                   [0.09, 0.23],
                   ]       
    elif args.plot_type in ['cks']:
        x_lists = [[0.1, 1.4],
                   [0.3, 1.6],
                   [0.5, 1.8],
                   [0.7, 2.0],
                   [0.9, 2.2],
                   [1.1, 2.4],
                   [1.3, 2.6],
                   ]        
    elif args.plot_type in ['aks']:
        x_lists = [[0.02, 0.11],
                   [0.03, 0.12],
                   [0.04, 0.13],
                   [0.05, 0.14],
                   [0.06, 0.15],
                   [0.07, 0.16],
                   [0.08, 0.17],
                   [0.09, 0.18],
                   [0.10, 0.19],
                   ]        
    elif args.plot_type in ['fak']:
        x_lists = [[0.30, 1.2],
                   [0.34, 1.24],
                   [0.38, 1.28],
                   [0.42, 1.32],
                   [0.46, 1.36],
                   [0.5, 1.40],
                   [0.54, 1.44],
                   [0.58, 1.48],
                   [0.62, 1.52],
                   ]        
    else:
        msg = "This plot type has not been implimented yet."
        raise NotImplementedError(msg, ())
        
    x_list = x_lists[nlabels]

    sx = []
    sy = []
    for x in x_list:
        if x < x_values[0]:
            continue
        dx = x_values[-1]/60.0 * series 
        nx = x + dx
        
        for index, yv in enumerate(solution):
            xv = x_values[index]
            if xv > nx:
                sx.append(nx)
                sy.append(yv)
                break
        
    return sx, sy

def plot_output(y_list=None, x_list=None, species_list=None,
                plot_name='default',
                args=None,
                legend_columns=1,
                title=None,
                sub_title=None,
                marker_size=6,
                x_label=None,
                y_label=None,
                data_dic=dict(),
                verbose=False,
                xmin=0,
                xmax=1,
                save_plot_parameters=False,
                ):

    #print len(y_list), len(x_list), len(species_list)
    #pp(species_list)

    if sub_title is None and title is None:
        save_as_svg = True
        save_as_pdf = False
    else:
        save_as_svg = False
        save_as_pdf = True

    pyplot.figure(1, figsize=(9,8))
    if save_as_pdf:
        pyplot.title(sub_title, fontsize=10)
    
    cf = pyplot.gcf()
    cf.clear()
    if save_as_pdf:
        cf.suptitle(title, fontsize=12, fontweight='bold')
    
    subplot_dic = dict(CP=1, ATP_g=2, P=3, ATP_b=4)

    ax_dic = dict()
    for k, axis_number in subplot_dic.items():
        ax = cf.add_subplot(2, 2, axis_number)

        nf = NullFormatter()

        if k == 'ATP_g':
            ax.xaxis.set_major_formatter(nf)
            ax.yaxis.set_major_formatter(nf)
        if k == 'CP':
            ax.xaxis.set_major_formatter(nf)
            if save_as_svg:
                ax.yaxis.set_major_formatter(nf)
        if k == 'ATP_b':
            ax.yaxis.set_major_formatter(nf)
            if save_as_svg:
                ax.xaxis.set_major_formatter(nf)
        if k == 'P' and save_as_svg:
            ax.xaxis.set_major_formatter(nf)
            ax.yaxis.set_major_formatter(nf)
        
        ax_dic[k] = ax

    plot_list = make_plot_list(args, x_list, y_list, species_list)

    st_dic = defaultdict(list)
    for met_name, series, nlabels, colour, symbol, fill, x_list, solution in sorted(plot_list):
        #print met_name, colour, symbol, series, nlabels
        style = colour + '_' + symbol
        st_dic[style].append(dict(met_name=met_name, series=series, nlabels=nlabels))

    style_dic = dict()
    for style, t_list in st_dic.items():
        s = None
        n = None
        ml = []
        for t_dic in t_list:
            ml.append(t_dic['met_name'])
            if s is None:
                s = t_dic['series']
            if n is None:
                n = t_dic['nlabels']
            #assert s == t_dic['series']
            #assert n == t_dic['nlabels']
        style_dic[style] = dict(series=s, nlabels=n, met_names=ml)
    #pp(style_dic)
    
    for index, data in enumerate(sorted(plot_list)):
        met_name, series, nlabels, colour, symbol, fill, x_values, solution = data

        ax = ax_dic.get(met_name, False)
        axis_number = subplot_dic[met_name]
        pyplot.subplot(2, 2, axis_number)
        pyplot.ylim(0, 1)
        if args.plot_type == 'tr':
            pyplot.xlim(args.x_min, args.x_max)
        else:
            pyplot.xlim(xmin, xmax)
        ax.plot(x_values, solution, color=colour, linestyle='solid', linewidth=1)

    p_dic = dict()
    for index, data in enumerate(sorted(plot_list)):
        met_name, series, nlabels, colour, symbol, fill, x_values, solution = data

        ax = ax_dic.get(met_name, False)
        axis_number = subplot_dic[met_name]
        pyplot.subplot(2, 2, axis_number)
        pyplot.ylim(0, 1)
        if args.plot_type == 'tr':
            pyplot.xlim(args.x_min, args.x_max)
        else:
            pyplot.xlim(xmin, xmax)
        sx, sy = slice_soln(args, series, nlabels, x_values, solution)
        if sx != [] and sy != []:
            if not args.combine_species:
                p1 = ax.plot(sx, sy, markersize=marker_size, markeredgewidth=0.001, linestyle='None', color=colour, marker=symbol, fillstyle=fill)
            else:
                measured_x, measured_y = get_measured_data(met_name, return_plot_lists=True)
                p1 = ax.plot(measured_x, measured_y,
                             markersize=marker_size, linestyle='None', color='b', marker=symbol, fillstyle=fill)
            style = colour + '_' + symbol
            s = style_dic[style]['series']
            n = style_dic[style]['nlabels']
            label = 's{0} n{1}'.format(s, n)
            p_dic[style] = (p1, label)

    for k, axis_number in subplot_dic.items():
        ax = cf.add_subplot(2, 2, axis_number)
        pyplot.subplot(2, 2, axis_number)
        if args.plot_type == 'tr':
            pyplot.vlines(30, 0, 1, color='k', linestyles='solid', alpha=0.3)

        if args.plot_type == 'fak':
            rt = Rectangle((0,0), 0.112, 1, color='k', alpha=0.1, zorder=0)
            ax.add_patch(rt)
            rt = Rectangle((0.188,0), 0.3-0.188, 1, color='k', alpha=0.1, zorder=0)
            ax.add_patch(rt)
            rt = Rectangle((1.888,0), 2-1.888, 1, color='k', alpha=0.1, zorder=0)
            ax.add_patch(rt)
            pyplot.vlines(0.276, 0, 1, color='k', linestyles='solid', alpha=0.3)

        if args.plot_type == 'aer':
            pyplot.vlines(0.276, 0, 1, color='k', linestyles='solid', alpha=0.3)

        if args.plot_type == 'ckn':
            pyplot.vlines(2.02, 0, 1, color='k', linestyles='solid', alpha=0.3)

    a_list = []
    l_list = []
    for style, (artist, label) in sorted(p_dic.items()):
        a_list.append(artist)
        l_list.append(label)

    legend_props = dict(frameon=False, markerscale=1, ncol=legend_columns, prop=dict(size=6),
                        labelspacing=0.0, numpoints=1,
                        loc=(1.00, 0.0))
    if save_as_pdf:
        pyplot.legend(a_list, l_list, **legend_props)
        subplot_props = dict(left=0.08, top=0.90, bottom=0.08, right=0.88, wspace=0.06, hspace=0.06)
    if save_as_svg:
        subplot_props = dict(left=0.12, top=0.88, bottom=0.12, right=0.80, wspace=0.06, hspace=0.06)
        
    pyplot.subplots_adjust(**subplot_props)

    fnt = '{0.save_dir}/{1}.{2}'
    if save_as_pdf:        
        fn = fnt.format(args, plot_name, 'pdf')
    if save_as_svg:
        fn = fnt.format(args, plot_name, 'svg')
    print 'Saving {0}'.format(fn)
    pyplot.savefig(fn)

    if save_plot_parameters:
        fn = fnt.format(args, 'py')
        print 'Saving plot parameters to', fn
        f = open(fn, 'w')
        f.write('plot_parameters = {0}\n'.format(pf(args.__dict__)))
        f.close()


def move_grid_job_files():
    taskid = int(os.environ['SGE_TASK_ID'])
    jobid = int(os.environ['JOB_ID'])
    print 'Finished running grid job with SGE_TASK_ID={0} and JOB_ID={1}'.format(taskid, jobid)

    efn = 'python.sh.e{0}.{1}'.format(jobid, taskid)
    ofn = 'python.sh.o{0}.{1}'.format(jobid, taskid)

    f = open(efn, 'r')
    lines = f.readlines()
    f.close()

    if lines == []:
        os.system('rm {0}'.format(efn))
        os.system('rm {0}'.format(ofn))

def execute_cmd(cmd):
    f = open('output.txt', 'a')
    f.write('Executing:\n{0}\n'.format(cmd))
    f.close()
    proc = subprocess.Popen(cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=os.environ)
    out, err = proc.communicate()
    if out.strip() != str():
        f = open('output.txt', 'a')
        f.write('Output:\n{0}'.format(out))
        f.close()
    if err.strip() != str():
        f = open('output.txt', 'a')
        f.write('Error:\n{0}'.format(err))
        f.close()
    return out, err

def get_solution(args, solution_name, series_key, x_max=800, return_error_dic=False):    
    import_template = 'from {0.import_dir}.{0.model_name}_c_variables import c_variables'
    import_string = import_template.format(args)
    try:
        exec(import_string)
    except ImportError, msg:
        raise UserWarning(msg, args)
    species_list = c_variables['input_list']
    
    i_str = 'solution_list, time_list'
    import_string = 'from {0.import_dir}.{0.model_name}_solution_{1}_output import {2}'
    try:
        exec(import_string.format(args, solution_name, i_str))
    except ImportError, msg:
        f = open('output.txt', 'a')
        f.write('Solution not found.  Attempting to solve for it now.\n')
        f.close()
        cmd_tmpl = 'python solve_isotope_model.py -d {0.import_dir} -D {0.save_dir} -s {1} -R -X {2}'
        cmd = cmd_tmpl.format(args, solution_name, x_max)
        try:
            execute_cmd(cmd)
            exec(import_string.format(args, solution_name, i_str))
        except ImportError, msg:
            raise UserWarning(msg)

    import_string = 'from {0.import_dir}.{0.model_name}_solution_{1}_input import input_variables'
    try:
        exec(import_string.format(args, solution_name))
    except ImportError, msg:
        raise UserWarning(msg)
    pool_dic = input_variables['pool_dic']
    fd = input_variables['flux_dic']

    dn = None
    if args.add_diagrams:
        cmd_tmpl = 'python generate_diagram.py -s {1} -p {2} -d {0.import_dir} -D {0.import_dir}'
        dn = '{0.import_dir}/{0.model_name}_solution_{1}_flux_diagram_{2}.pdf'.format(args, solution_name, series_key)
        if not os.path.exists(dn):
            cmd = cmd_tmpl.format(args, solution_name, series_key)
            execute_cmd(cmd)

    if return_error_dic:
        args.combine_species = True
        args.split_ATP = True

    s_list, sp_list = combine_solution(numpy.array(solution_list).T, species_list, pool_dic, args=args)

    assert len(species_list) == len(solution_list[0]), `len(species_list) == len(solution_list[0])`

    if hasattr(s_list, 'tolist'):
        s_list = s_list.tolist()
    if hasattr(time_list, 'tolist'):
        time_list = time_list.tolist()

    if return_error_dic:
        return calculate_errors(s_list, sp_list, time_list, pool_dic)
    else:
        return s_list, sp_list, time_list, fd, dn


def add_model_parameter_arguments(p, string_arg='S', float_arg='F',
                                  default_template='[default:%(default)s]',
                                  choices_template='[choices:%(choices)s]'):

    ST = string_arg
    FL = float_arg
    d = default_template
    c = choices_template
                   
    p.add_argument("-s", "--solutions",
                   dest="solution_names", default=None, metavar=ST, nargs='+',
                   help='Isotopomer reaction solution names to solve. {0}'.format(d))

    p.add_argument("-S", "--AS-flux",
                   dest="AS_flux", default=2.24954, metavar=FL, type=float,
                   help="Value of AS_flux for the simulation. {0}".format(d))
    
    p.add_argument("-f", "--frac-ATP",
                   dest="frac_ATP", default=0.5, metavar=FL, type=float,
                   help="Value of frac_ATP, the fraction of energy usage in the form of ATP. {0}".format(d))
    
    p.add_argument("-K", "--frac-AK",
                   dest="frac_AK", default=0.0, metavar=FL, type=float,
                   help="Value of frac_AK, the fraction of energy usage in the form of ADP. {0}".format(d))
    
    p.add_argument("-F", "--frac-CK",
                   dest="frac_CK", default=0.5, metavar=FL, type=float,
                   help="Value of frac_CK, the fraction of energy usage in the form of CK. {0}".format(d))
    
    p.add_argument("-e", "--ef-ASs", type=float,
                   dest="ef_ASs", default=0, metavar=FL,
                   help="Value of the ATP synthase exchange flux. {0}".format(d))
    
    p.add_argument("-E", "--ef-ASe", type=float,
                   dest="ef_ASe", default=0, metavar=FL,
                   help="Value of the ATPase exchange flux. {0}".format(d))
    
    p.add_argument("-a", "--ef-AKi", type=float,
                   dest="ef_AKi", default=0, metavar=FL,
                   help="Value of the intermembrane AK exchange flux. {0}".format(d))
    
    p.add_argument("-A", "--ef-AKo", type=float,
                   dest="ef_AKo", default=0, metavar=FL,
                   help="Value of the cytosolic AK exchange flux. {0}".format(d))
    
    p.add_argument("-c", "--ef-CKi", type=float,
                   dest="ef_CKi", default=None, metavar=FL,
                   help="Value of the intermembrane CK exchange flux. {0}".format(d))
    
    p.add_argument("-C", "--ef-CKo", type=float, 
                   dest="ef_CKo", default=None, metavar=FL,
                   help="Value of the cytosolic CK exchange flux. {0}".format(d))
    
    p.add_argument("-b", "--ef-case", type=str, metavar=ST,
                   dest="bidirectional_transport_case", default='uniA', choices=['uniA', 'uniB', 'uniC', 'biA', 'biB', 'biC'],
                   help="Code for the bidirectional transport case used. {0} {1}".format(d, c))
    
    p.add_argument("-p", "--pool-case", type=str, 
                   dest="pool_case", default='base', metavar=ST, choices=['base'],
                   help="Code for the pool case used. {0} {1}".format(d, c))

    return p

def get_ef_CK_from_frac_ATP(args, verbose=False):
    #try:
    #    exec('from generated.exchange_flux_array import ef_array')
    #except ImportError, msg:
    #    raise UserWarning(msg)

    # This uses only one of the ten values provided by Marko.
    #for index, frac_ATP in enumerate(ef_array['frac_ATP']):
    #    tol = 1e-4
    #    if abs(frac_ATP - (args.frac_ATP+args.frac_AK)) < tol:
    #        args.ef_CKo = ef_array['ef_CKo'][index]
    #        args.ef_CKi = ef_array['ef_CKi'][index]
    #        break
    #assert args.ef_CKo is not None and args.ef_CKi is not None, `args.ef_CKo, args.ef_CKi`
    assert 0,'not using'

    a = 2.48
    A = 0.088
    
    b = 1.9
    B = 3.882
    
    c = -0.1545
    C = 4.9956

    f = args.frac_ATP + args.frac_AK

    if f <= 0.6:
        args.ef_CKo = f * b + B
        args.ef_CKi = 0
    if f > 0.6:
        args.ef_CKo = (f - 0.6) * c + C
        args.ef_CKi = (f - 0.6) * a + A

    if verbose:
        print 'get_ef_CK_from_frac_ATP in = {1}  ef_CKo = {0.ef_CKo}  ef_CKi = {0.ef_CKi}'.format(args, f) 
    
    return args

def make_solution_name(args, include_list=None):
    if include_list is None:
        include_list = range(12)

    test_arguments(args)
    s = 'X'
    nd = dict(A=s, B=s, C=s, D=s, E=s, F=s, G=s, H=s, J=s, K=s, L=s, M=s)
    for index in include_list:
        if index == 0: nd['A'] = round_float(args.AS_flux * 10000)
        if index == 1: nd['B'] = round_float(args.frac_ATP * 10000)
        if index == 2: nd['C'] = round_float(args.frac_AK * 10000)
        if index == 3: nd['D'] = round_float(args.frac_CK * 10000)
        if index == 4: nd['E'] = round_float(args.ef_AKi * 10000)
        if index == 5: nd['F'] = round_float(args.ef_AKo * 10000)
        if index == 6: nd['G'] = round_float(args.ef_CKi * 10000)
        if index == 7: nd['H'] = round_float(args.ef_CKo * 10000)
        if index == 8: nd['J'] = round_float(args.ef_ASs * 10000)
        if index == 9: nd['K'] = round_float(args.ef_ASe * 10000)
        if index == 10: nd['L'] = args.bidirectional_transport_case
        if index == 11: nd['M'] = args.pool_case

    ns = Namespace()
    ns.__dict__ = nd
    
    return '{0.A}_{0.B}_{0.C}_{0.D}_{0.E}_{0.F}_{0.G}_{0.H}_{0.J}_{0.K}_{0.L}_{0.M}'.format(ns)


def test_arguments(args):
    #bt_cases = ['uniA', 'uniB', 'uniC', 'biA', 'biB', 'biC']
    bt_cases = ['uniA', 'biB', 'biC']
    pool_cases = ['base']
    tol = 1e-3

    if args.frac_ATP > 1 or args.frac_CK > 1 or args.frac_AK > 1:
        msg = "A fraction of energy transfer must be smaller than 1."
        raise UserWarning(msg, (args.frac_ATP, args.frac_CK, args.frac_AK))
    elif args.frac_ATP < 0 or args.frac_CK < 0 or args.frac_AK < 0:
        msg = "A fraction of energy transfer must be greater than 0."
        raise UserWarning(msg, (args.frac_ATP, args.frac_CK, args.frac_AK))
    elif args.frac_ATP + args.frac_CK + args.frac_AK - 1 > tol:
        msg = "The sum of frac_AK + frac_ATP + frac_CK is greater than 1.0."
        raise UserWarning(msg, (args.frac_ATP, args.frac_CK, args.frac_AK))
    elif args.ef_AKi < 0 or args.ef_AKo < 0 or args.ef_ASe < 0 or args.ef_ASs < 0:
        msg = "An exchange flux is less than 0.0."
        raise UserWarning(msg, (args.ef_AKi, args.ef_AKo, args.ef_CKo, args.ef_CKi, args.ef_ASe, args.ef_ASs))
    elif args.bidirectional_transport_case not in bt_cases:
        msg = "A bidirectional transport case was specified that has not been implimemted."
        raise UserWarning(msg, (args.bidirectional_transport_case, bt_cases))
    elif args.pool_case not in pool_cases:
        msg = "A pool case was specified that has not been implimemted."
        raise UserWarning(msg, (args.pool_case, pool_cases))

def decode_solution_name(args, solution_name):
    e = solution_name.split('_')
    if len(e) != 12:
        msg = "A solution_name was passed that did not have twelve data elements separated by '_' characters."
        raise UserWarning(msg, (solution_name, e))
    
    args.AS_flux = float(e[0]) / 10000.0
    args.frac_ATP = float(e[1]) / 10000.0
    args.frac_AK = float(e[2]) / 10000.0
    args.frac_CK = float(e[3]) / 10000.0
    args.ef_AKi = float(e[4]) / 10000.0
    args.ef_AKo = float(e[5]) / 10000.0
    args.ef_CKi = float(e[6]) / 10000.0
    args.ef_CKo = float(e[7]) / 10000.0
    args.ef_ASs = float(e[8]) / 10000.0
    args.ef_ASe = float(e[9]) / 10000.0    
    args.bidirectional_transport_case = e[10]
    args.pool_case = e[11]
    
    return args


def round_float(float, precision=0, return_string=True):
    new = round(float, precision)
    if return_string:
        if precision < 0:
            p = '0'
        else:
            p = str(precision)
        exec('s = "%0.' + p + 'f" %new')
        return str(int(s)) 
    else:
        return new  


def combine_solution(solution_list, species_list, pool_dic,
                     args=None,
                     verbose=False):

    s_list = solution_list

    if args.combine_compartments or args.combine_species or args.split_ATP:
        if verbose:
            print 'Combining compartments.'
        new_species_list = []
        new_s_list = []
        total_ADP = pool_dic['ADPi'] + pool_dic['ADPm'] + pool_dic['ADPo'] + pool_dic['ADPe'] + pool_dic['ADPs']
        total_ATP = pool_dic['ATPi'] + pool_dic['ATPm'] + pool_dic['ATPo'] + pool_dic['ATPe'] + pool_dic['ATPs']
        total_CP = pool_dic['CPi'] + pool_dic['CPo']
        total_P = pool_dic['Pm'] + pool_dic['Po'] + pool_dic['Pe'] + pool_dic['Ps']
        for i in range(4):
            new_species_list.append('ADP_' + species_list[i].split('_')[1])
            new_s_list.append((s_list[i]*pool_dic['ADPe'] + \
                               s_list[i+4]*pool_dic['ADPi'] + \
                               s_list[i+8]*pool_dic['ADPm'] + \
                               s_list[i+12]*pool_dic['ADPo'] + \
                               s_list[i+16]*pool_dic['ADPs']) / total_ADP)
        for i in range(16):
            e = species_list[20+i].split('_')
            new_species_list.append('ATP_' + e[1] + '_' + e[2])
            new_s_list.append((s_list[20+i]*pool_dic['ATPe'] + \
                               s_list[20+i+16]*pool_dic['ATPi'] + \
                               s_list[20+i+32]*pool_dic['ATPm'] + \
                               s_list[20+i+48]*pool_dic['ATPo'] + \
                               s_list[20+i+64]*pool_dic['ATPs']) / total_ATP)
        for i in range(4):
            new_species_list.append('CP_' + species_list[100+i].split('_')[1])
            new_s_list.append((s_list[100+i]*pool_dic['CPi'] + s_list[100+i+4]*pool_dic['CPo']) / total_CP)
        for i in range(5):
            new_species_list.append('P_' + species_list[108+i].split('_')[1])
            new_s_list.append((s_list[108+i]*pool_dic['Pe'] + \
                               s_list[108+i+5]*pool_dic['Pm'] + \
                               s_list[108+i+10]*pool_dic['Po'] + \
                               s_list[108+i+15]*pool_dic['Ps']) / total_P)
        s_list = new_s_list
        species_list = new_species_list

    if args.split_ATP:
        if verbose:
            print 'Splitting ATP into beta and gamma.'

        new_species_list = []
        new_s_list = []

        new_species_list += ['ADP_0', 'ADP_1', 'ADP_2', 'ADP_3']
        new_s_list += [s_list[0], s_list[1], s_list[2], s_list[3]]

        new_species_list += ['ATP_g_0']
        new_s_list += [s_list[4] + s_list[8] + s_list[12] + s_list[16]]
        
        new_species_list += ['ATP_g_1']
        new_s_list += [s_list[5] + s_list[9] + s_list[13] + s_list[17]]

        new_species_list += ['ATP_g_2']
        new_s_list += [s_list[6] + s_list[10] + s_list[14] + s_list[18]]

        new_species_list += ['ATP_g_3']
        new_s_list += [s_list[7] + s_list[11] + s_list[15] + s_list[19]]

        new_species_list += ['ATP_b_0']
        new_s_list += [s_list[4] + s_list[5] + s_list[6] + s_list[7]]

        new_species_list += ['ATP_b_1']
        new_s_list += [s_list[8] + s_list[9] + s_list[10] + s_list[11]]

        new_species_list += ['ATP_b_2']
        new_s_list += [s_list[12] + s_list[13] + s_list[14] + s_list[15]]

        new_species_list += ['ATP_b_3']
        new_s_list += [s_list[16] + s_list[17] + s_list[18] + s_list[19]]

        new_species_list += ['CP_0', 'CP_1', 'CP_2', 'CP_3']
        new_s_list += [s_list[20], s_list[21], s_list[22], s_list[23]]

        new_species_list += ['P_0', 'P_1', 'P_2', 'P_3', 'P_4']
        new_s_list += [s_list[24], s_list[25], s_list[26], s_list[27], s_list[28]]

        assert len(s_list) == 29, `len(s_list)`

        s_list = new_s_list
        species_list = new_species_list
                
        if args.combine_species:
            if verbose:
                print 'Combining species.'
            if args.convert_to_fraction_of_total_labeling:
                f_ATP = 1/3.0
                f_ADP = 1/3.0
                f_P = 1/4.0
                f_CP = 1/3.0
                lwc = 1/0.3
            else:
                f_ADP = 1
                f_ATP = 1
                f_CP = 1
                f_P = 1
                lwc = 1

            new_species_list = []
            new_s_list = []

            new_species_list += ['ADP_1']
            new_s_list += [(s_list[1]*1 + s_list[2]*2 + s_list[3]*3) * f_ADP * lwc]

            new_species_list += ['ATP_g_1']
            new_s_list += [(s_list[5]*1 + s_list[6]*2 + s_list[7]*3) * f_ATP * lwc]

            new_species_list += ['ATP_b_1']
            new_s_list += [(s_list[9]*1 + s_list[10]*2 + s_list[11]*3) * f_ATP * lwc]

            new_species_list += ['CP_1']
            new_s_list += [(s_list[13]*1 + s_list[14]*2 + s_list[15]*3) * f_CP * lwc]

            new_species_list += ['P_1']
            new_s_list += [(s_list[17]*1 + s_list[18]*2 + s_list[19]*3 + s_list[20]*4) * f_P * lwc]

            assert len(s_list) == 21, `len(s_list)`

            s_list = new_s_list
            species_list = new_species_list

    if args.combine_compartments and args.combine_species and not args.split_ATP:
        if verbose:
            print 'Combining species.'
        if args.convert_to_fraction_of_total_labeling:
            f_ATP = 1/3.0
            f_ADP = 1/3.0
            f_P = 1/4.0
            f_CP = 1/3.0
            lwc = 1/0.3
        else:
            f_ADP = 1
            f_ATP = 1
            f_CP = 1
            f_P = 1
            lwc = 1

        new_species_list = []
        new_s_list = []

        new_species_list += ['ADP']
        new_s_list += [(s_list[1]*1 + s_list[2]*2 + s_list[3]*3) * f_ADP * lwc]

        new_species_list += ['ATP_g']
        new_s_list += [(s_list[5]*1 + s_list[6]*2 + s_list[7]*3 + \
                       s_list[9]*1 + s_list[10]*2 + s_list[11]*3 + \
                       s_list[13]*1 + s_list[14]*2 + s_list[15]*3 + \
                       s_list[17]*1 + s_list[18]*2 + s_list[19]*3) * f_ATP * lwc]

        new_species_list += ['ATP_b']
        new_s_list += [(s_list[8]*1 + \
                       s_list[9]*1 + s_list[10]*1 + s_list[11]*1 + s_list[12]*2 + \
                       s_list[13]*2 + s_list[14]*2 + s_list[15]*2 + s_list[16]*3 + \
                       s_list[17]*3 + s_list[18]*3 + s_list[19]*3) * f_ATP * lwc]

        new_species_list += ['CP']
        new_s_list += [(s_list[21]*1 + s_list[22]*2 + s_list[23]*3) * f_CP * lwc]

        new_species_list += ['P']
        new_s_list += [(s_list[25]*1 + s_list[26]*2 + s_list[27]*3 + s_list[28]*4) * f_P * lwc]

        assert len(s_list) == 29, `len(s_list)`

        s_list = new_s_list
        species_list = new_species_list

    return s_list, species_list


def drop_to_ipython(*z, **kwds):
    '''
    Drops to ipython at the point in the code where it is called to inspect the variables passed to it.

    Parameters
    ----------
    z: tuple
      All variables passed to this routine are wrapped into a tuple.
    kwds : dict
      If the keyword "local_variables" is passed (output of locals()),
      the call name is extracted from the calling class.
    '''
    from IPython.Shell import IPShellEmbed
    lvs = kwds.get('local_variables', False) 
    if not lvs:
        lvs = []
        
    try:
        call_name = local_variables['self'].__module__
    except Exception:
        call_name = "Module"

    b = 'Dropping into IPython'
    em = 'Leaving Interpreter, back to program.'
    msg = '***Called from %s. Hit Ctrl-D to exit interpreter and continue program.'
    ipshell = IPShellEmbed([], banner=b, exit_msg=em)
    ipshell(msg %(call_name))
