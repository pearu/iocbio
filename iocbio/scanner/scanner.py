
import sys
import numpy
import nidaqmx
from cache import Cache
from nidaqmx.libnidaqmx import make_pattern
from sample_clock import SampleClock
from mirror import MirrorDriverCubic as MirrorDriver

class Scanner:

    def __init__ (self, options):
        self.options = options

    def title (self):
        options = self.options
        alpha = options.orientation_angle
        roi = (options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1)
        title = 'ROI%s_%s_%s_%s_ANGLE%s' % (roi + (int(alpha),))
        return title

    def setup(self, scan_speed, flyback_ratio):
        options = self.options
        alpha = options.orientation_angle
        roi = (options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1)
        if options.pixel_time_usec is None:
            pixel_size = options.pixel_size_um * 1e-6 #m
            pixel_time = pixel_size/scan_speed
        else:
            pixel_time = options.pixel_time_usec * 1e-6 # sec
            pixel_size = scan_speed * pixel_time # m

        image_width = MirrorDriver.compute_image_width(options.roi_x0, options.roi_x1, pixel_size)
        image_height = options.image_height

        sample_clock = SampleClock (pixel_time, pixel_size)
        self.sample_clock = sample_clock
        samples_per_pixel, clock_rate, min_tune_clock_rate = sample_clock.get_optimal_scanning_parameters()        
        output_input_clock_ratio = min(int(clock_rate / self.max_ai_clock_rate)+1, samples_per_pixel)
        while samples_per_pixel % output_input_clock_ratio:
            output_input_clock_ratio += 1

        self.ai_clock_rate = clock_rate // output_input_clock_ratio
        self.ao_clock_rate = clock_rate

        mirror = MirrorDriver(image_size = (image_width, image_height),
                              T=samples_per_pixel, 
                              flyback_ratio = flyback_ratio,
                              flystart_ratio = 0.1,
                              alpha = alpha,
                              roi = roi,
                              clock_rate = self.ao_clock_rate,
                              verbose=False)
        self.mirror = mirror


        self.ao_samples_per_channel = mirror.get_ticks()
        self.ai_samples_per_channel = (self.ao_samples_per_channel * self.ai_clock_rate) // self.ao_clock_rate 

        self.configure_tasks()

        self.vx_target = vx_target = self.mirror.vx_t_array()
        self.vy_target = vy_target = self.mirror.vy_t_array()

        self.xa_target = self.mirror.xa_v(vx_target, vy_target)
        self.ya_target = self.mirror.ya_v(vx_target, vy_target)

        self.i_target = self.mirror.i_v(vx_target, vy_target)
        self.j_target = self.mirror.j_v(vx_target, vy_target)

    def create_tasks(self):
        options = self.options
        self.delete_tasks()
        ao_channels = make_pattern([
                options.mirror_x_analog_output_channels,
                options.mirror_y_analog_output_channels,
                ])
        ai_channels = make_pattern([
                options.mirror_x_error_analog_input_channels,
                options.mirror_y_error_analog_input_channels,
                ])
        ao_task = nidaqmx.AnalogOutputTask()
        ao_task.create_voltage_channel(ao_channels, 
                                       min_val=-10,
                                       max_val=10,
                                       units='volts')
        ai_task = nidaqmx.AnalogInputTask()
        ai_task.create_voltage_channel(ai_channels, 
                                       terminal = 'nrse',
                                       min_val=-10,
                                       max_val=10,
                                       units='volts')
        self.max_ai_clock_rate = ai_task.get_sample_clock_max_rate ()
        do_task = nidaqmx.DigitalOutputTask()
        do_task.create_channel(options.start_trigger_digital_output_lines,
                               grouping='per_line')
        ao_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')
        ai_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')


        self.ao_task = ao_task
        self.ai_task = ai_task
        self.do_task = do_task

    def free_tasks (self):
        self.ao_task = None
        self.ai_task = None
        self.do_task = None        

    def delete_tasks (self):
        self.ai_clock_rate = None
        self.ao_clock_rate = None
        self.ai_samples_per_channel = None
        self.ao_samples_per_channel = None
        self.free_tasks()

    def configure_tasks (self):
        params = [self.ai_clock_rate, self.ao_clock_rate, self.ai_samples_per_channel, self.ao_samples_per_channel]
        assert None not in params, `params`
        self.ai_task.configure_timing_sample_clock(rate = self.ai_clock_rate,
                                                   active_edge = 'rising',
                                                   sample_mode = 'finite',
                                                   samples_per_channel = self.ai_samples_per_channel,
                                                   )
        self.ao_task.configure_timing_sample_clock(rate = self.ao_clock_rate,
                                                   active_edge = 'rising',
                                                   sample_mode = 'finite',
                                                   samples_per_channel = self.ao_samples_per_channel,
                                                   )

    def run_tasks(self, ao_data):
        ao_data = numpy.array(ao_data).T.ravel()

        print 'WRITING DATA(shape=%s)' % (ao_data.shape,),
        sys.stdout.flush()
        self.ao_task.write(ao_data, auto_start=False)

        print 'READY',
        sys.stdout.flush()
        self.do_task.start()
        self.do_task.write(0) 
        self.ao_task.start()
        self.ai_task.start()

        print 'START',
        sys.stdout.flush()
        self.do_task.write(1)

        print 'RUNNING',
        sys.stdout.flush()
        self.ao_task.wait_until_done()

        print 'FINISH',
        sys.stdout.flush()
        self.do_task.write(0)

        print 'READING',
        sys.stdout.flush()
        ai_data = self.ai_task.read().T
        print 'GOT DATA(shape=%s)' % (ai_data.shape,),

        print 'STOPPING',
        sys.stdout.flush()
        self.do_task.stop()
        self.ai_task.stop()
        self.ao_task.stop()
        print 'DONE.'
        sys.stdout.flush()

        return ai_data

    def scan(self, average=False, cache=Cache('Scanner_scan.pickle').load()):

        if not average:
            self.vx_input = vx_input = self.mirror.vx_t_array()
            self.vy_input = vy_input = self.mirror.vy_t_array()
            self.i_input = self.mirror.i_v(vx_input, vy_input)
            self.j_input = self.mirror.j_v(vx_input, vy_input)
            self.scan_average_counter = 1
        else:
            self.scan_average_counter += 1

        assert len(vx_input)==self.ao_samples_per_channel,`len(vx_input), self.ao_samples_per_channel`
        assert self.ai_samples_per_channel == self.ao_samples_per_channel,`self.ai_samples_per_channel, self.ao_samples_per_channel`

        self.configure_tasks()

        key = (self.mirror.get_key(), self.sample_clock.get_key())
        data = cache.data.get(key)
        if data is None:
            data = self.run_tasks([vx_input, vy_input])
            cache.data[key] = data
            cache.dump()

        vx_error = -data[0] * 2
        vy_error = -data[1] * 2

        self.vx_error = vx_error
        self.vy_error = vy_error

        if average:
            def apply_average (old, new, n=self.scan_average_counter):
                old[:] = (old * (n-1) + new)/n
                return old

            self.vx_response = vx_input - vx_error
            self.vy_response = vy_input - vy_error
            
            apply_average(self.xa_response, self.mirror.xa_v(self.vx_response, self.vy_response))
            apply_average(self.ya_response, self.mirror.ya_v(self.vx_response, self.vy_response))
            
            apply_average(self.i_response, self.mirror.i_v(self.vx_response, self.vy_response))
            apply_average(self.j_response, self.mirror.j_v(self.vx_response, self.vy_response))
            
            apply_average(self.xa_error, self.xa_target - self.xa_response)
            apply_average(self.ya_error, self.ya_target - self.ya_response)
            
            apply_average(self.i_error, self.i_target - self.i_response)
            apply_average(self.j_error, self.j_target - self.j_response)

        else:

            self.vx_response = vx_input - vx_error
            self.vy_response = vy_input - vy_error
            
            self.xa_response = self.mirror.xa_v(self.vx_response, self.vy_response)
            self.ya_response = self.mirror.ya_v(self.vx_response, self.vy_response)
            
            self.i_response = self.mirror.i_v(self.vx_response, self.vy_response)
            self.j_response = self.mirror.j_v(self.vx_response, self.vy_response)
            
            self.xa_error = self.xa_target - self.xa_response
            self.ya_error = self.ya_target - self.ya_response
            
            self.i_error = self.i_target - self.i_response
            self.j_error = self.j_target - self.j_response

        return

    def iterate(self, n=1):
        mirror = self.mirror

        self.scan()

        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = \
            mirror.estimate_offsets(self.i_target, self.j_target, self.i_response, self.j_response)

        mirror.set_params(t_offset=toffset + self.mirror.t_offset)

        self.scan()

        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = \
            mirror.estimate_offsets(self.i_target, self.j_target, self.i_response, self.j_response)

        mirror.set_params(i_offset=ioffset + mirror.i_offset,
                          j_offset=joffset + mirror.j_offset,
                          )

        self.scan()

        for i in range(n-1):
            self.scan(average=True)

    def show(self):
        
        plots = [511, 512, 513, 514, 515]
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,12))
        def on_keypressed(event, plt=plt, fig=fig):
            key = event.key
            if key=='q':
                plt.close(fig)
            if key=='x':
                plt.close(fig)
                sys.exit(0)

        fig.canvas.mpl_connect('key_press_event', on_keypressed)

        mirror = self.mirror
        vx_target, vy_target = self.vx_target, self.vy_target
        i_target, j_target = self.i_target, self.j_target
        vx_input, vy_input = self.vx_input, self.vy_input
        i_input, j_input = self.i_input, self.j_input
        vx_err, vy_err = self.vx_error, self.vy_error
        i_pos, j_pos = self.i_response, self.j_response
        vx_response, vy_response = self.vx_response, self.vy_response
        sample_clock = self.sample_clock

        output_input_clock_ratio = self.ai_clock_rate // self.ao_clock_rate

        i_target_av = average(i_target, mirror.T//output_input_clock_ratio)
        j_target_av = average(j_target, mirror.T//output_input_clock_ratio)
        
        i_input_av = average(i_input, mirror.T//output_input_clock_ratio)
        j_input_av = average(j_input, mirror.T//output_input_clock_ratio)
        
        i_response_av = average(i_pos, mirror.T//output_input_clock_ratio)
        j_response_av = average(j_pos, mirror.T//output_input_clock_ratio)
        
        i_err_av = i_target_av - i_response_av
        j_err_av = j_target_av - j_response_av

        t_start = mirror.T * mirror.K0 // output_input_clock_ratio
        t_end = mirror.T * mirror.K1 // output_input_clock_ratio + 1    
        plt.subplot(plots[0])
        plt.plot(vx_target[:t_start],vy_target[:t_start], '-',
                 vx_target[t_start:-t_end], vy_target[t_start:-t_end], '-',
                 vx_target[-t_end:], vy_target[-t_end:], '-')

        plt.xlabel ('Vx, volts')
        plt.ylabel ('Vy, volts')
        title = 'N=%(N)s, M=%(M)s, K0=%(K0)s, K=%(K)s, K1=%(K1)s, T=%(T)s' % (mirror.__dict__)
        title += ',\n scan/line time=%.3esec/%.3fmsec, scan speed=%sm/s, clock rate=%sHz' \
            % (vx_target.size * sample_clock.pixel_time * output_input_clock_ratio,
               mirror.N * sample_clock.pixel_time * output_input_clock_ratio * 1e3, 
               mirror.get_pixel_size_xa() / sample_clock.pixel_time,
               sample_clock.clock_rate)
        title += ',\n flyback ratio=%s, position std=%.3fum/%.3f' \
            % (mirror.K / mirror.N, 
               mirror.line_average(self.xa_error).std()*1e6, 
               mirror.line_average(self.i_error).std())
        plt.title(title)

        plt.plot (vx_response[:t_start],vy_response[:t_start], '-',
                  vx_response[t_start:-t_end], vy_response[t_start:-t_end], '-',
                  vx_response[-t_end:], vy_response[-t_end:], '-')
    
        plt.subplot(plots[1])
    
        plt.plot (vx_target, '-', vy_target, '-')
        plt.plot (vx_input[::output_input_clock_ratio], '-', vy_input[::output_input_clock_ratio], '-')
        plt.plot (vx_response, '-', vy_response, '-')
        if mirror.M <= 10:
            mk_state_lines(plt, mirror, T = mirror.T/output_input_clock_ratio)
        plt.legend(['vx_target', 'vy_target', 'vx_input', 'vy_input', 'vx_response', 'vy_response'])

    
        plt.subplot (plots[2])
        plt.plot (vx_err, '-', vy_err, '-')
        if mirror.M <= 10:
            mk_state_lines(plt, mirror, T = mirror.T/output_input_clock_ratio)
        plt.axhline(0)
        plt.legend(['vx_err', 'vy_err', 'vx_err2', 'vy_err2'][:2])

        plt.subplot (plots[3])
        plt.plot(i_target_av, '-', j_target_av, '-', 
                 i_input_av, '-', j_input_av,'-',
                 i_response_av,'-',j_response_av,'-',)
        if mirror.M <= 10:
            mk_state_lines(plt, mirror, T=1)
        plt.legend (['i_target', 'j_target', 'i_input', 'j_input', 'i_response', 'j_response'])

        plt.subplot (plots[4])

        t0 = 0
        t1 = t0 + mirror.K0
        for j in range(mirror.M):
            if j:
                t0 = t1
                t1 = t0 + mirror.K
            t0 = t1
            t1 = t0 + mirror.N
            plt.plot(range(t0, t1+1), i_err_av[t0:t1+1], '-b')
            #plt.plot(range(t0, t1+1), j_err_av[t0:t1+1], '-g')
        t0 = t1
        t1 = t0 + mirror.K1
        if mirror.M <= 10:
            mk_state_lines(plt, mirror, T=1)
        plt.axhline(0)
        plt.axvline(0)
        plt.legend (['i_err', 'j_err'][:1])
        
        plt.draw()

        return plt

def average(arr, T):
    if T==1:
        return arr
    r = numpy.zeros(arr.size // T, dtype=float)
    S = r.size
    for i in range(T):
        r += arr[i:S*T:T]
    return r / T

def mk_state_lines (plt, mirror, T=None):
    if T is None:
        T = mirror.T
    t = 0
    t += mirror.K0 * T
    plt.axvline(t)
    for j in range(mirror.M):
        if j:
            t += mirror.K * T
            plt.axvline(t)
        t += mirror.N * T
        plt.axvline(t)
    t += mirror.K1 * T
    plt.axvline(t)
