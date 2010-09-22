
from __future__ import division
import os
import time
import glob

from ..timeunit import Time, Seconds, Minutes, Hours

class Channel:
    """ Holds channel data.

    Attributes
    ----------
    model : Model
    index : int
      Channel index.
    protocol : str
      The name of channels protocol.
    time_data : list
      A list of time values.
    value_data : list
      A list of values.
    tasks : dict
      A dictionary of time values and task labels.
    params : dict
      A dictionary of parameter names and values.
    stream : file
      An open file where data will be saved.
    """
    
    def __init__ (self, model, index):
        """
        Parameters
        ----------
        model : Model
        index : int
          Specify channel index.
        """
        self.model = model
        self.index = index
        self.protocol = None
        self.time_data = None
        self.value_data = None
        self.stream = None
        self.params = {}
        self.parameters = None

    def set_protocol(self, protocol):
        self.protocol = protocol
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_parameters(self):
        if self.protocol is None:
            return []
        if self.parameters is None:
            self.parameters = [p.copy() for p in self.model.get_protocol_parameters (self.protocol)]
        return self.parameters

    def start(self):
        """ Notify the experiment start.
        """
        self.stop()
        filename = self.model.channel_data_template % self.index
        print 'Channel %s: starting the experiment, results go to %r' % (self.index, filename)

        self._last_data_index = None
        self._last_task = None
        self.time_data = []
        self.value_data = []
        self.tasks = {}
        self.params[' this file'] = filename
        self.params[' channel index'] = self.index
        self.params[' experiment title'] = self.model.title
        self.params[' start time'] = time.strftime('%y-%m-%d %H:%M', time.localtime(self.model.start_time))
        self.params[' protocol'] = self.protocol
        self.stream = open(filename, 'w')


    def stop(self):
        """ Notify the experiment end.
        """
        if self.stream is None:
            return
        self.save()
        self.stream.close()
        self.stream = None
        self._last_task = None

    def save(self):
        """ Save parameters and data.
        """
        if self.stream is None:
            print 'warning: no stream defined in channel'
            return
        if self._last_data_index is None:
            self._saved_params = []
            self._last_data_index = 0
            
        # update protocol parameters
        for p in self.get_parameters():
            self.params[self.protocol+'.'+p.name] = p.get_value()
        
        # update configuration parameters
        for p in self.model.get_configuration():
            self.params['Configuration.'+p.name] = p.get_value()

        if len(self._saved_params) != len(self.params):
            for key in sorted(self.params):
                item = key, self.params[key]
                if item not in self._saved_params:
                    self.stream.write('# %s : %s\n' % item)
                    self._saved_params.append(item)
        if self._last_data_index==0:
            label0 = self.model.get_axis_label(0)
            label1 = self.model.get_axis_label(1)
            label2 = self.model.get_axis_label(2)
            self.stream.write('# %16s %18s %18s %s\n' % (label0, label1, label2, '<event[<comment>]>'))
        start, end = self._last_data_index, len(self.time_data)
        print 'Saving channel %s data[%s:%s]' % (self.index, start, end)
        time = self.get_time()
        data = self.get_data()
        slope = self.get_data_slope()
        for i in range(start, end):
            t = time[i]
            v = data[i]
            r = slope[i]
            l = self.tasks.pop(t, '')
            self.stream.write('%18s %18s %18s %s\n' % (t, v, r,l))

        while self.tasks:
            t, l = self.tasks.popitem()
            i = time.index (t)
            v = data[i]
            r = slope[i]
            self.stream.write('# %18s %18s %18s %s\n' % (t, v,r, l))

        self.stream.flush()
        self._last_data_index = len(time)

    def add_data(self, t, value):
        """ Add value to data.

        Parameters
        ----------
        t : float
          Specify time moment in seconds.
        """
        self.time_data.append(t)
        self.value_data.append(value)
        self.data_slope.add(value)

    def add_task (self, tm, task):
        """ Add task to data.
        """
        dist = [(abs(t1-tm), i) for i,t1 in enumerate(self.time_data)]
        if dist:
            t = self.time_data[min(dist)[1]]
        if t in self.tasks:
            self.tasks[t] += ';' + task
        else:
            self.tasks[t] = task
        if task.endswith(']'):
            # remove comment part
            i = task.index('[')
            task = task[:i].rstrip()
        if task.lower ()=='comment' and self._last_task is not None:
            pass
        else:
            self._last_task = task
        return t

    _last_task = None
    def get_tasks(self):
        """ Return a list of tasks and a suggestion for the next task.
        """
        if self.protocol is None:
            return [], None
        tasks = self.model.protocols[self.protocol]
        tasks = [task for task in tasks if not task.startswith ('param:')]
        if not tasks:
            return [], None
        try:
            next = tasks.index(self._last_task) + 1
        except ValueError:
            next = 0
        if next >= len(tasks):
            next = len(tasks) - 1
        return tasks, tasks[next]

    def set_parameter(self, name, value):
        self.params[name] = value

    def get_time (self):
        unit = self.get_time_unit()
        return [Time (t, unit=unit) for t in self.time_data]
        #return self.time_data

    def get_data (self):
        return self.value_data

    def get_data_slope(self, s=None):
        if s is not None:
            self.data_slope.update(s)
        unit = self.get_time_unit()
        tfactor = dict(s=1, min=1/60, h=1/60/60)[unit]

        factor = self.get_volume_ml() / tfactor
        slope = self.data_slope.get_slope()
        if factor==1:
            return slope
        return [v*factor for v in slope]

    def init_slope(self, dt, s):
        self.data_slope = DataSlope(dt, s)

    def get_volume_ml(self):
        for p in self.get_parameters():
            if p.name=='volume_ml':
                return float(p.get_value() or 1)
            elif p.name=='volume_l':
                return float(p.get_value() or 1)*1e3
        return 1

    def get_time_unit (self):
        return self.model.get_axis_unit(0)

class DataSlope:
    """ Computes the slope of added data using last s regression points.
    """

    def __init__(self, dt, s):
        self.dt = dt
        self.init(s)

    def init (self, s):
        self.s = s
        self.y = []
        self.sum_y = [0]
        self.sum_iy = [0]
        self.n = 0
        self.slope = []

    def add(self, yn, negative_slope=True):
        self.n += 1
        s, n, dt = self.s, self.n, self.dt
        dt = float (dt) # seconds
        y = self.y
        sum_y = self.sum_y
        sum_iy = self.sum_iy
        slope = self.slope

        s = min(s, n)
        sum_iy.append(sum_iy[-1] + s*yn - (sum_y[-1] if n>=s+1 else 0))
        sum_y.append(sum_y[-1] + yn - (y[n-s-1] if n>=s+1 else 0))
        y.append(yn)

        if s == 1:
            slope.append(0)
        else:
            sv = ( sum_iy[-1] - (s+1)/2*sum_y[-1] ) / (dt/12*s*(s*s-1))
            if negative_slope:
                sv = -sv
            slope.append( sv )
        if s == 2:
            slope[0] = slope[1]

    def get_slope(self):
        return self.slope

    def update(self, s):
        if s==self.s:
            return
        y = self.y[:]
        self.init(s)
        map(self.add, y)


class Model:
    """ Holds protocols and channels.

    Attributes
    ----------
    main_dir : str
      A directory name where configuration files will be saved.
    title : str
      A one-line description of experiment.
    protocols : dict
      A dictionary of protocol names and tasks lists.
    channels : list
      A list of Channel instances.
    config : dict
      A dictonary of configuration names and values.
    """
    
    def __init__(self):
        self.main_dir = None
        self.title = None
        self.start_time = None

    def set_title(self, title):
        self.title = title

    def init(self, main_dir, nof_channels = 6):
        self.main_dir = main_dir
        if not os.path.isdir(main_dir):
            os.makedirs(main_dir)
        self.channels = []
        self.protocols = {} # {<protocol name>:<list of tasks>}
        for i in range(nof_channels):
            self.channels.append(Channel(self, i+1))
        self.load_protocols()

        # initialize path_to_results_dir configuration parameter:
        self.get_results_dir()

    def set_channel_protocol(self, channel_index, protocol):
        self.channels[channel_index - 1].set_protocol(protocol)

    def add_channel_data(self, channel_index, t, value):
        channel = self.channels[channel_index - 1]
        channel.add_data(t, value)

    def add_channel_task(self, channel_index, t, task):
        channel = self.channels[channel_index - 1]
        return channel.add_task(t, task)

    def get_channel_tasks(self, channel_index):
        channel = self.channels[channel_index - 1]
        return channel.get_tasks()

    def select_channel_protocol(self, channel_index, protocol):
        channel = self.channels[channel_index - 1]
        channel.set_protocol(protocol)

    def set_channel_parameter(self, channel_index, name, value):
        channel = self.channels[channel_index - 1]
        channel.set_parameter(name, value)

    def get_channel_protocol(self, channel_index):
        channel = self.channels[channel_index - 1]
        return channel.protocol

    def get_current_time(self):
        if self.start_time is None:
            return None
        channel = self.channels[0]
        if not channel.time_data:
            return 0
        return channel.time_data[-1]
        return channel.convert_time(channel.time_data[-1])

    def start(self):
        """ Notify the start of experiment.
        """
        self.start_time = start_time = time.time()
        today = time.strftime('%y-%m-%d', time.localtime(start_time))
        dirname = os.path.join(self.get_results_dir(), today)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        next = 1
        while 1:
            tmpl = os.path.join(dirname, '%s_%s_ch%%d.txt' % (today, next))
            tmpl_pdf = os.path.join(dirname, '%s_%s.pdf' % (today, next))
            if not os.path.isfile(tmpl % 1):
                break
            next += 1
        self.channel_data_template = tmpl
        self.figure_pdf = tmpl_pdf
        for channel in self.channels:
            channel.start()

    def init_slope(self, dt, s=2):
        for channel in self.channels:
            channel.init_slope(dt, s)
        
    def stop(self):
        """ Notify the end of experiment.
        """
        for channel in self.channels:
            channel.stop()
        self.start_time = None

    def save(self):
        """ Save channels data.
        """
        for channel in self.channels:
            channel.save()

    def load_protocols(self):
        """ Load protocols from protocols.txt file.
        """
        filename = os.path.join(self.main_dir, 'protocols.txt')
        print 'Loading protocols from "%s"' % (filename)
        self.protocols = {}

        self.protocols['Example protocol'] = ['First task, just try it.',
                                              'Second task, have a tea.',
                                              'Last task, be happy.',
                                              'param: text comments',
                                              'param: int cells = 1',
                                              'param: string [A, B] collagenases = B',
                                              'param: float temperature = 20.5 # Celsius',
                                              'Comment'
                                              ]
        # All channels will have `protocol` choice parameter and volume_ml parameter.

        if os.path.isfile(filename):
            stream = open(filename, 'r')
            for line in stream:
                line = line.rstrip()
                if line.startswith('#') or not line:
                    continue
                if line.startswith('\t'):
                    self.protocols[protocol].append(line.lstrip())
                else:
                    protocol = line.rstrip()
                    self.protocols[protocol] = []
            stream.close()

        if '_Configuration' not in self.protocols:
            # adding special protocol (their names start with underscore)
            self.protocols['_Configuration'] = []

        # Update configuration parameters
        params = []
        params.append('param: [auto, range, interval] time_axis = auto')
        params.append('param: time_axis_range = None..None # start..end, in time units' )
        params.append('param: time_axis_interval = None # width, in time units')
        params.append('param: [s, min, h] time_units = s')

        params.append('param: [auto, range] oxygen_axis = auto')
        params.append('param: oxygen_axis_range = None..None # start..end, in oxygen units')
        params.append('param: [ug/ml, mg/l, ul/ml, ml/l, umol/l, torr, kPa, %satn] oxygen_units = ug/ml')

        params.append('param: [auto, range] respiration_rate_axis = auto')
        params.append('param: respiration_rate_axis_range = None..None # start..end, in rate units')
        #params.append('param: [ug/min, ug/h, mg/min, mg/h] respiration_rate_units = ug/min')

        params.append('param: int rate_regression_points = 2')

        params.append(r'param: directory path_to_results_dir')
        #params.append(r'param: file strathkelvin_exe = C:\Strathk\Meter.exe')
        params.append(r'param: file path_to_strathkelvin_program = C:\Strathk\Meter.exe')
        config_params = self.protocols['_Configuration']

        for param_line in params:
            p = Parameter(param_line[6:].lstrip ())
            if not self.has_parameter(p.name):
                config_params.append(param_line)

        self.refresh()
        return

    def refresh (self):
        # initialize parameters cache, this will also add missing volume_ml parameters.
        map (self.get_protocol_parameters, self.protocols)        

    def save_protocols(self):
        """ Save protocols to protocols.txt file.
        """
        filename = os.path.join(self.main_dir, 'protocols.txt')
        if os.path.isfile(filename):
            fin = open(filename)
            fout = open(filename[:-4]+'_backup.txt', 'w')
            fout.write(fin.read())
            fout.close()
            fin.close()
        stream = open(filename, 'w')
        stream.write('# Creation time: %s\n' % (time.ctime()))
        for protocol, tasks in self.protocols.items():
            stream.write('%s\n' % (protocol))

            if protocol == '_Configuration':
                for p in self.get_configuration():
                    stream.write('\tparam: %s\n' % (p.to_line()))
            else:
                for task in tasks:
                    stream.write('\t%s\n' % (task))
        stream.close()
        return filename

    _parameters_cache = {}

    def get_protocol_parameters(self, protocol):
        """ Extract protocol parameters.

        Parameters
        ----------
        protocol : str
          Specify protocol name.

        Returns
        -------
        parameters : list
          A list of Parameter instances.

        See also
        --------
        parse_param
        """
        parameters = self._parameters_cache.get(protocol)
        if parameters is None:
            self._parameters_cache[protocol] = parameters = []
        old_parameters = dict([(p.param_line,p) for p in parameters])
        parameters[:] = []
        tasks = self.protocols.get(protocol, [])
        has_volume_ml = False
        existing_parameters = []

        for task in tasks:
            if task.startswith('param:'):
                param_line = task[6:].lstrip()
                p = old_parameters.get(param_line, None)
                if p is None:
                    p = Parameter(param_line)
                if p.name not in existing_parameters:
                    existing_parameters.append(p.name)
                    parameters.append(p)
                    has_volume_ml |= p.name == 'volume_ml'
                else:
                    print 'Warning: parameter with name %r already exists in protocol %r. Ignoring %r.' % (p.name, protocol, task)

        if protocol and not protocol.startswith('_'):
            if not has_volume_ml:
                param_line = 'float volume_ml = 1'
                tasks.insert(0, 'param: '+param_line)
                parameters.insert(0, Parameter(param_line))
        return parameters

    def get_parameters(self, obj):
        if isinstance (obj, str):
            return self.get_protocol_parameters(obj)
        elif isinstance(obj, Channel):
            return obj.get_parameters()
        else:
            raise NotImplementedError (`obj`)

    def get_configuration(self):
        return self.get_parameters('_Configuration')

    def get_parameter(self, name, protocol='_Configuration'):
        params = self.get_parameters(protocol)
        for p in params:
            if p.name==name:
                return p

    def get_parameter_value(self, name, protocol='_Configuration'):
        params = self.get_parameters(protocol)
        for p in params:
            if p.name==name:
                return p.get_value()

    def has_parameter (self, name, protocol='_Configuration'):
        params = self.get_parameters(protocol)
        for p in params:
            if p.name==name:
                return True
        return

    def _get_axis_config_label (self, axis):
        if axis==1:
            return 'oxygen'
        elif axis==2:
            return 'respiration_rate'
        elif axis==0:
            return 'time'
        else:
            raise NotImplementedError (`axis`)        

    def get_axis_factors(self, protocol='_Configuration'):
        unit0 = self.axis_unit (0, protocol=protocol)
        factor0 = dict(s=1, min=1/60, h=1/60/60).get(unit0,1)

        unit1 = self.axis_unit (1, protocol=protocol)
        factor1 = 1

        unit2 = self.axis_unit (2, protocol=protocol)
        factor2 = 1/factor0

        return factor0, factor1, factor2

    def get_axis_unit(self, axis=1, protocol='_Configuration'):
        unit = None
        if axis==2:
            unit0 = self.get_axis_unit (0, protocol=protocol)
            unit1 = self.get_axis_unit (1, protocol=protocol)
            # multiply unit1 by ml volume
            unit1 = {'ug/ml':'ug', 'mg/l':'ug', 'ul/ml':'ul', 'ml/l':'ul', 'umol/l':'nmol', 'torr':'torr*ml', 'kPa':'kPa*ml', '%satn':'%satn*ml'}.get(unit1)
            if unit1 and unit0:
                unit = '%s/%s' % (unit1, unit0)
        else:
            label = self._get_axis_config_label(axis)
            unit = self.get_parameter_value ('%s_units' % label)
        return unit

    def get_axis_label(self, axis=1, protocol='_Configuration'):        
        label = self._get_axis_config_label(axis)
        unit = self.get_axis_unit(axis, protocol=protocol)
        if unit is not None:
            label = '%s [%s]' % (label, unit)
        return label

    def get_axis_range(self, axis=1, protocol='_Configuration'):
        label = self._get_axis_config_label(axis)
        mode = self.get_parameter_value('%s_axis' % label, protocol=protocol)
        if mode == 'interval':
            assert axis == 0,`axis`
            t1 = self.get_current_time()
            interval = self.get_parameter_value('%s_axis_interval' % (label), protocol=protocol)
            unit = self.get_parameter_value('%s_units' % (label), protocol=protocol)
            dt = Time(interval, unit)
            t1 = Time(t1, unit)
            return float(t1 - dt), float(t1)
        if mode != 'range':
            return None, None
        range = self.get_parameter_value('%s_axis_range' % (label), protocol=protocol)
        if range is None:
            return None, None

        if '..' in range:
            min, max = range.split('..',1)
        elif '-' in range:
            min, max = range.rsplit('-',1)
        else:
            print 'Unknown range: %r' % (r)
            return None, None
        min = min.strip ()
        max = max.strip ()
        if label=='time':
            unit = self.get_parameter_value('%s_units' % (label), protocol=protocol)
            if min.lower() in ['', 'none']:
                min = None
            else:
                min = float(Time (min, unit))
            if max.lower() in ['', 'none']:
                max = None
            else:
                max = float(Time (max, unit))
        else:
            try:
                min = float (min)
            except ValueError:
                min = None
            try:
                max = float (max)
            except ValueError:
                max = None
        return min, max

    def get_axis_interval(self, axis=1, protocol='_Configuration'):
        label = self._get_axis_config_label (axis)
        mode = self.get_parameter_value('%s_axis' % label, protocol=protocol)
        if mode != 'interval':
            return
        interval = self.get_parameter_value ('%s_axis_interval'%label, protocol=protocol)

        unit = self.get_parameter_value('%s_units' % (label), protocol=protocol)
        interval = Time(interval, unit)
        return interval

    def get_slope_n(self, protocol='_Configuration'):
        n = self.get_parameter_value('rate_regression_points', protocol=protocol)
        if n is None:
            return 2
        try:
            return max(2,int(n))
        except ValueError:
            return 2

    def get_results_dir(self, protocol='_Configuration'):
        d = self.get_parameter_value('path_to_results_dir', protocol=protocol)
        if d:
            if not os.path.isdir(d):
                os.makedirs(d)
            return d
        p = self.get_parameter ('path_to_results_dir', protocol=protocol)
        if p is not None:
            p.set_value(self.main_dir)
        return self.main_dir

class Parameter:
    """ Holds parameter information.

    Attributes
    ----------
    type : {'string', 'int', 'float'}
    choices : {list, None}
    name : str
    default : {str, None}

    See also
    --------
    parse
    """

    def __init__(self, param_line):
        self.value = None
        self.param_line = param_line
        self.parse(param_line)

    def __repr__ (self):
        if self.value is None:
            return '%s(%r)' % (self.__class__.__name__, self.param_line)
        return '%s(%r).set_value(%r)' % (self.__class__.__name__, self.param_line, self.value)

    def parse(self, param):
        """ Parse parameter definition.

        Parameters
        ----------
        param : str
          Specify parameter definition as a string with the following
          format::

            <type> <empty | list of choices> <name> = <default value> # comment

          where ``<type>`` can be ``int``, ``float``, ``string``, ``text``,
          ``file``, ``directory``.
        """
        if '#' in param:
            param, comment = param.rsplit('#', 1)
            comment = comment.strip()
        else:
            comment = None
        self.comment = comment
        if '=' in param:
            param, default = param.rsplit('=',1)
            default = default.strip()
        else:
            default = None
        self.default = default

        param = param.strip()
        if ' ' in param:
            param, name = param.rsplit(' ', 1)
            param = param.strip()
        else:
            name = param
            param = ''

        self.name = name.strip()
        assert ' ' not in name, `self.name, param`

        if param.endswith(']'):
            i = param.index('[')
            choices = [v.strip() for v in param[i+1:-1].split(',')]
            param = param[:i].rstrip()
        else:
            choices = None

        self.choices = choices

        if param:
            type = param
            assert type in ['text', 'int', 'float', 'string', 'file', 'directory'],`type`
        else:
            type = 'string'

        self.type = type

    def copy(self):
        return self.__class__(self.param_line)

    def set_value(self, value):
        self.value = value
        return self
    
    def get_value(self):
        if self.value is None:
            self.value = self.default
        return self.value

    def to_line(self):
        line = self.type
        if self.choices:
            line = '%s [%s]' % (line, ', '.join (self.choices))
        line = '%s %s' % (line, self.name)
        v = self.get_value()
        if v is not None:
            line = '%s = %s' % (line, v)
        if self.comment:
            line = '%s # %s' % (line, self.comment) 
        return line
