

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
