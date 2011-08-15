
from __future__ import division

import os
import sys
import time
import numpy
from optparse import OptionGroup
from iocbio.optparse_gui import OptionParser
from iocbio.scanner.configuration import camera_area_width, camera_area_height, pixel_flyback_rate
from iocbio.scanner.mirror import MirrorDriverCubic as MirrorDriver
from iocbio.scanner.mirror import xya_pq
from iocbio.scanner.sample_clock import SampleClock
from iocbio.scanner.script_options import set_run_mirrors_options

from iocbio.scanner.cache import Cache

def get_range(line):
    l = line.split(':')
    if len(l)==1:
        return map (eval, l)
    elif len(l)==2:
        start = eval(l[0].strip ())
        end = eval(l[1].strip ())
        step = 1
    elif len(l)==3:
        start = eval(l[0].strip ())
        end = eval(l[1].strip ())
        step = eval(l[2].strip ())
    else:
        raise NotImplementedError(`line`)
    return numpy.arange(start, end, step, dtype=float)
         

def runner (parser, options, args):

    if options.task=='scan':
        from scanner import Scanner
        scan = Scanner(options)

        if options.flyback is None:
            flyback_range = get_range(options.flyback_range)
        else:
            flyback_range = [options.flyback]

        scan_speed_range = [options.scan_speed]

        title = scan.title ()
        filename = 'pos_dict_%s.pickle' % (title)
        pos_dict_cache = Cache(filename).load()

        pos_dict = pos_dict_cache.data
        if 0:
            # scan
            for scan_speed in scan_speed_range:
                if scan_speed not in pos_dict:
                    pos_dict[scan_speed] = set([])
                elif isinstance (pos_dict[scan_speed], list):
                    pos_dict[scan_speed] = set(pos_dict[scan_speed])
                stop = False
                for flyback in flyback_range:
                    scan.create_tasks()
                    scan.setup (scan_speed, flyback)
                    try:
                        scan.iterate()
                    except RuntimeError, msg:
                        print 'Scanning failed with runtime error: %s' % (msg)
                        stop = True
                    scan.free_tasks()
                    if stop:
                        break
                    data = (flyback, scan.mirror.line_average(scan.xa_error).std()*1e6)
                    if data not in pos_dict[scan_speed]:
                        pos_dict[scan_speed].add(data)
                        pos_dict_cache.dump()
            #plt = scan.show()
            #plt.draw()
            #plt.show()
            pos_dict_cache.dump()


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

        legends = []
        scan_speed_list = sorted(pos_dict.keys ())
        for scan_speed in scan_speed_list:
            flyback_data = pos_dict[scan_speed]
            legends.append ('%.3f m/s' % (scan_speed))
            flyback, data = zip (*flyback_data)
            plt.semilogy(flyback,data, 'x')
        plt.title('Scan speed - flyback ratio curves: %s' % (title))
        plt.xlabel('flyback ratio')
        plt.ylabel('position resolution, um')
        plt.legend(legends)
        plt.show()

        return

    pixel_size = options.pixel_size_um * 1e-6 # m
    if options.image_width is None:
        image_width = MirrorDriver.compute_image_width(options.roi_x0, options.roi_x1, pixel_size)
    else:
        image_width = options.image_width

    pixel_size = MirrorDriver.compute_pixel_size_x(options.roi_x0, options.roi_x1, image_width) # m

    scan_speed = options.scan_speed # m/sec

    if options.pixel_time_usec is None:
        pixel_time = pixel_size / scan_speed # sec
    else:
        pixel_time =  options.pixel_time_usec*1e-6 # sec
    
    scan_speed = pixel_size / pixel_time

    if options.flyback is None:
        flyback = min(2, pixel_size / pixel_time / pixel_flyback_rate)
    else:
        flyback = options.flyback

    assert flyback>=0, `flyback`



    mirror_params = {}
    for k,v in options.__dict__.items():
        if k.startswith ('param_'):
            mirror_params[k[6:]] = v

    if options.task!='scan':
        sample_clock = SampleClock(pixel_time = pixel_time,
                                   pixel_size = pixel_size)
        samples_per_pixel, clock_rate, min_clock_rate = sample_clock.get_optimal_scanning_parameters()
        mirror = MirrorDriver(image_size = (image_width, options.image_height),
                              T=samples_per_pixel, 
                              flyback_ratio = flyback,
                              flystart_ratio = 0.05,
                              alpha = options.orientation_angle,
                              roi = (options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1),
                              clock_rate = clock_rate,
                              #params = mirror_params
                              )



    if options.task=='initialize':
        return
    if options.task=='plot':
        mirror.set_params(**mirror_params)

        print mirror.flyback_alpha

        start_time = time.time ()
        vx = mirror.vx_t_array()
        vy = mirror.vy_t_array()
        print 'Computing X and Y mirror input took %.3f sec' % (time.time () - start_time)


        t_start = mirror.T * mirror.K0
        t_end = mirror.T * mirror.K1 + 1

        import matplotlib.pyplot as plt
        fig = plt.figure(1, figsize=(12,12))
        def on_keypressed(event):
            key = event.key
            if key=='q':
                sys.exit(0)
        fig.canvas.mpl_connect('key_press_event', on_keypressed)
        
        print vx[:10]

        plt.subplot (311)
        plt.plot (vx[:t_start],vy[:t_start], '.',
                  vx[t_start:-t_end], vy[t_start:-t_end], '.',
                  vx[-t_end:], vy[-t_end:], '.')
        plt.xlabel ('Vx, volts')
        plt.ylabel ('Vy, volts')
        title = 'N=%(N)s, M=%(M)s, K0=%(K0)s, K=%(K)s, K1=%(K1)s, T=%(T)s' % (mirror.__dict__)
        title += ', scan time=%ssec, clock rate=%sHz' % (vx.size * pixel_time, clock_rate)
        plt.title(title)

        plt.subplot (312)
        plt.plot (vx, '-', vy, '-')
        mk_state_lines(plt, mirror)
        plt.legend(['vx', 'vy'])

        plt.subplot (313)
        plt.plot (numpy.diff(vx), '-', numpy.diff(vy), '-')
        mk_state_lines(plt, mirror)
        plt.legend(['Dvx', 'Dvy'])

        plt.draw()
        plt.show()
        return
    if options.task=='measure':
        import nidaqmx
        from nidaqmx.libnidaqmx import make_pattern

        vx_target = mirror.vx_t_array()
        vy_target = mirror.vy_t_array()
        i_target = mirror.i_t_array()
        j_target = mirror.j_t_array()

        mirror.set_params(**mirror_params)
        
        ao_channels = make_pattern([
                options.mirror_x_analog_output_channels,
                options.mirror_y_analog_output_channels,
                ])
        ai_channels = make_pattern([
                options.mirror_x_error_analog_input_channels,
                options.mirror_y_error_analog_input_channels,
                ])

        ao_task = nidaqmx.AnalogOutputTask()
        print 'Creating AO voltage channel:', ao_channels
        ao_task.create_voltage_channel(ao_channels, 
                                       min_val=-10,
                                       max_val=10,
                                       units='volts')
        ao_task.configure_timing_sample_clock(rate = clock_rate,
                                              active_edge = 'rising',
                                              sample_mode = 'finite',
                                              samples_per_channel = vx_target.size
                                              )

        ai_task = nidaqmx.AnalogInputTask()
        print 'Creating AI voltage channel:', ai_channels
        ai_task.create_voltage_channel(ai_channels, 
                                       terminal = 'nrse',
                                       min_val=-10,
                                       max_val=10,
                                       units='volts')

        max_ai_clock_rate = ai_task.get_sample_clock_max_rate ()

        output_input_clock_ratio = min(int(clock_rate / max_ai_clock_rate)+1, mirror.T)
        while mirror.T % output_input_clock_ratio: # or clock_rate // output_input_clock_ratio > max_ai_clock_rate:
            output_input_clock_ratio += 1
        print 'Analog output - analog input clock ratio =', output_input_clock_ratio

        ai_clock_rate = clock_rate // output_input_clock_ratio
        print 'Analog input clock rate: %s <= %s' % (ai_clock_rate, max_ai_clock_rate)
        
        if output_input_clock_ratio != 1:
            vx_target = vx_target[::output_input_clock_ratio]
            vy_target = vy_target[::output_input_clock_ratio]
            i_target = i_target[::output_input_clock_ratio]
            j_target = j_target[::output_input_clock_ratio]

        ai_task.configure_timing_sample_clock(rate = ai_clock_rate,
                                              active_edge = 'rising',
                                              sample_mode = 'finite',
                                              samples_per_channel = vx_target.size
                                              )


        do_task = nidaqmx.DigitalOutputTask()
        do_task.create_channel(options.start_trigger_digital_output_lines,
                               grouping='per_line')


        ao_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')
        ai_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')


        start_time = time.time()
        vx_input = mirror.vx_t_array()
        vy_input = mirror.vy_t_array()
        print 'Computing mirror input took %.3f sec' % (time.time () - start_time)
        
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                     vx_input, vy_input, output_input_clock_ratio)

        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)
        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)
        mirror.set_params(t_offset=toffset + mirror.t_offset)
        vx_input = mirror.vx_t_array()
        vy_input = mirror.vy_t_array()
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                     vx_input, vy_input, output_input_clock_ratio)

        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)
        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)
        mirror.set_params(i_offset=ioffset + mirror.i_offset,
                          j_offset=joffset + mirror.j_offset,
                          )
        vx_input = mirror.vx_t_array()
        vy_input = mirror.vy_t_array()
        
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                     vx_input, vy_input, output_input_clock_ratio)
        
        n = 2
        for i in range (n-1):
            vx_pos1, vy_pos1, vx_err1, vy_err1 = run_mirrors(ao_task, ai_task, do_task, 
                                                             vx_input, vy_input, output_input_clock_ratio)
            vx_pos += vx_pos1
            vy_pos += vy_pos1
            vx_err += vx_err1
            vy_err += vy_err1
        vx_pos /= n
        vy_pos /= n
        vx_err /= n
        vy_err /= n

        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)

        i_input = mirror.i_v(vx_input, vy_input)
        j_input = mirror.j_v(vx_input, vy_input)

        plt = mk_figure(vx_target, vy_target, 
                        i_target, j_target, 
                        vx_input, vy_input, i_input, j_input,
                        vx_pos, vy_pos, i_pos, j_pos,
                        vx_err, vy_err, mirror, 
                        sample_clock, output_input_clock_ratio)

        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)

        plt.xlabel('ioffset=%s(%.3f,%.3f), joffset=%s(%.3f,%.3f), toffset=%s' % (ioffset, ioffset_std, ioffset_max, joffset, joffset_std, joffset_max, toffset))
        plt.draw()

        del ao_task, ai_task, do_task
        plt.show ()

        return
    if options.task=='scan':
        import nidaqmx
        from nidaqmx.libnidaqmx import make_pattern
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
        max_ai_clock_rate = ai_task.get_sample_clock_max_rate ()
        do_task = nidaqmx.DigitalOutputTask()
        do_task.create_channel(options.start_trigger_digital_output_lines,
                               grouping='per_line')
        ao_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')
        ai_task.configure_trigger_digital_edge_start (options.start_trigger_terminal,
                                                      edge='rising')


        min_pixel_size = 1 # um
        max_pixel_size = 10 # um
        min_pixel_time = 10 # us
        max_pixel_time = 30 # us
        min_scan_speed = min_pixel_size / max_pixel_time
        max_scan_speed = max_pixel_size / min_pixel_time
        nof_scan_speeds = 10
        nof_pixel_times = 2
        scan_speed_range = min_scan_speed + numpy.arange(nof_scan_speeds, dtype=float)/(nof_scan_speeds-1) * (max_scan_speed - min_scan_speed)
        scan_speed_log_range = numpy.exp(numpy.log(min_scan_speed) + numpy.arange(nof_scan_speeds, dtype=float)/(nof_scan_speeds-1) * (numpy.log(max_scan_speed/min_scan_speed)))
        pixel_time_range = min_pixel_time + numpy.arange(nof_pixel_times, dtype=float)/(nof_pixel_times-1) * (max_pixel_time - min_pixel_time)
        alpha = options.orientation_angle
        roi = (options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1)
        ll = xya_pq(roi[0], roi[1])
        ur = xya_pq(roi[2], roi[3])
        scan_area_width = (ur[0] - ll[0]) * 1e6 # um
        scan_area_height = (ur[1] - ll[1]) * 1e6 # um
        M = options.image_height

        data_scan_speed = []
        data_flyback_ratio = []

        cache = Cache()
        cache.load()

        flystart_ratio = 0.5

        target_ioffset_std = 1.0
        error_margin = 0.1
        for pixel_time in pixel_time_range:
            for scan_speed in scan_speed_log_range:
                pixel_size_x = scan_speed * pixel_time # um
                N = max(1, int(scan_area_width / pixel_size_x))

                K = N
                
                error_must_increase = None
                last_error = None
                last_K = None
                for i in range(5):
                    S = int(K * flystart_ratio)
                    ioffset_std = do_scan(S, K, max_ai_clock_rate, ai_task, ao_task, do_task, N, M, pixel_time, scan_speed, pixel_size_x, alpha, roi, cache=cache)

                    error = target_ioffset_std - ioffset_std
                    if abs (error) <= error_margin:
                        print 'Iteration converged:',i, error, last_error
                        break
                    if K == last_K:
                        print i, (error,K), (last_error, last_K)
                        print 'K did not change, stopping'
                        break
                    if error_must_increase is not None:
                        if error_must_increase:
                            if error <= last_error:
                                print i, (error,K), (last_error, last_K)
                                print 'Expected error increase, got decrease. Breaking!'
                                error, K = last_error, last_K
                                break
                        else:
                            if error >= last_error:
                                print i, (error,K), (last_error, last_K)
                                print 'Expected error decrease, got increase. Breaking!'
                                error, K = last_error, last_K
                                break
                    last_error = error
                    last_K = K
                    if error > error_margin:
                        K = max(1,min(K-1,int(K / 1.3)))
                        error_must_increase = False
                    elif error < -error_margin:
                        K = max(K+1, int(1.3 * K))
                        error_must_increase = True
                    else:
                        print error
                        raise UNEXPECTED_CODE

                data_scan_speed.append(scan_speed)
                data_flyback_ratio.append(K/N)
                
        del ao_task, ai_task, do_task

        print 'Scan speed:', data_scan_speed
        print 'Flyback ratio:',data_flyback_ratio

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,12))
        def on_keypressed(event, plt=plt, fig=fig):
            key = event.key
            if key=='q':
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_keypressed)
        plt.semilogx(data_scan_speed, data_flyback_ratio, 'x')
        plt.xlabel('Scan speed [m/s]')
        plt.ylabel('Flyback ratio')
        plt.draw()
        plt.show()

        return
    raise NotImplementedError (`options.task`)

def do_scan(S, K, max_ai_clock_rate, ai_task, ao_task, do_task, N, M, pixel_time, scan_speed, pixel_size_x, alpha, roi,
            cache = None):

    key = (S, K, N, M, pixel_time, scan_speed, alpha, roi)

    

    flyback_ratio = K/N
    flystart_ratio = S/K
    sample_clock = SampleClock(pixel_time = pixel_time * 1e-6,
                               pixel_size = pixel_size_x * 1e-6)
    T, clock_rate, min_clock_rate = sample_clock.get_optimal_scanning_parameters(verbose=False)
    mirror = MirrorDriver(image_size = (N,M),
                          T=T, flyback_ratio = flyback_ratio,
                          flystart_ratio = flystart_ratio,
                          alpha = alpha,
                          roi = roi,
                          clock_rate = clock_rate,
                          verbose=False)

    output_input_clock_ratio = min(int(clock_rate / max_ai_clock_rate)+1, T)
    while mirror.T % output_input_clock_ratio:
        output_input_clock_ratio += 1

    ai_clock_rate = clock_rate // output_input_clock_ratio
    
    result = None
    if cache is not None:
        result = cache.data.get(key)
    if result is None:
        vx_target = mirror.vx_t_array()
        vy_target = mirror.vy_t_array()
        i_target = mirror.i_t_array()
        j_target = mirror.j_t_array()

        ai_task.configure_timing_sample_clock(rate = ai_clock_rate,
                                              active_edge = 'rising',
                                              sample_mode = 'finite',
                                              samples_per_channel = vx_target.size
                                              )
        ao_task.configure_timing_sample_clock(rate = clock_rate,
                                              active_edge = 'rising',
                                              sample_mode = 'finite',
                                              samples_per_channel = vx_target.size
                                              )

        # first iteration
        vx_input = vx_target
        vy_input = vy_target
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                     vx_input, vy_input, output_input_clock_ratio)
        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)
        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)
        
        
        # second iteration
        mirror.set_params(t_offset=toffset + mirror.t_offset)
        vx_input = mirror.vx_t_array()
        vy_input = mirror.vy_t_array()
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                 vx_input, vy_input, output_input_clock_ratio)                
        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)
        toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)
        
        # third iteration
        mirror.set_params(i_offset=ioffset + mirror.i_offset,
                          j_offset=joffset + mirror.j_offset,
                          )
        vx_input = mirror.vx_t_array()
        vy_input = mirror.vy_t_array()
        vx_pos, vy_pos, vx_err, vy_err = run_mirrors(ao_task, ai_task, do_task, 
                                                     vx_input, vy_input, output_input_clock_ratio)
        i_pos = mirror.i_v(vx_pos, vy_pos)
        j_pos = mirror.j_v(vx_pos, vy_pos)
        i_input = mirror.i_v(vx_input, vy_input)
        j_input = mirror.j_v(vx_input, vy_input)

        result = vx_target, vy_target, i_target, j_target, vx_input, vy_input, i_input, j_input, vx_pos, vy_pos, i_pos, j_pos, vx_err, vy_err
        if cache is not None:
            cache.data[key] = result
            cache.dump()

    else:
        print 'Using cache for key=', key

    vx_target, vy_target, i_target, j_target, vx_input, vy_input, i_input, j_input, vx_pos, vy_pos, i_pos, j_pos, vx_err, vy_err = result

    toffset, ioffset, joffset, ioffset_std, joffset_std, ioffset_max, joffset_max = mirror.estimate_offsets(i_target, j_target, i_pos, j_pos)

    print 'XXX:', pixel_time, scan_speed, N,M,K,T,'->',toffset, ioffset, joffset, ioffset_std, joffset_std

    from ioc.microscope.regress import regress
    regress_scale = min(1, 20/N)
    i_pos, i_pos_grad = regress(i_pos, (regress_scale,), method='average')
    j_pos, j_pos_grad = regress(j_pos, (regress_scale,), method='average')
    vx_pos, vx_pos_grad = regress(vx_pos, (regress_scale,), method='average')
    vy_pos, vy_pos_grad = regress(vy_pos, (regress_scale,), method='average')

    plt = mk_figure(vx_target, vy_target, i_target, j_target, vx_input, vy_input,
                    i_input, j_input,
                    vx_pos, vy_pos,  i_pos, j_pos, vx_err, vy_err,
                    mirror, sample_clock, output_input_clock_ratio
                    )
    plt.xlabel('ioffset=%s(%.3f,%.3f), joffset=%s(%.3f,%.3f), toffset=%s' % (ioffset, ioffset_std, ioffset_max, joffset, joffset_std, joffset_max, toffset))
    plt.draw()
    plt.show()

    return ioffset_max

def mk_figure(vx_target, vy_target, 
              i_target, j_target, 
              vx_input, vy_input, i_input, j_input,
              vx_pos, vy_pos, i_pos, j_pos,
              vx_err, vy_err, 
              mirror, sample_clock, output_input_clock_ratio):

    K0, K, N, M, K1, T = [getattr(mirror, name) for name in ['K0', 'K', 'N', 'M', 'K1', 'T']]


    i_target_av = average(i_target, T//output_input_clock_ratio)
    j_target_av = average(j_target, T//output_input_clock_ratio)

    i_input_av = average(i_input, T//output_input_clock_ratio)
    j_input_av = average(j_input, T//output_input_clock_ratio)

    i_pos_av = average(i_pos, T//output_input_clock_ratio)
    j_pos_av = average(j_pos, T//output_input_clock_ratio)

    i_err_av = i_target_av - i_pos_av
    j_err_av = j_target_av - j_pos_av

    #plots = [321, 323, 324, 325, 326]
    plots = [511, 512, 513, 514, 515]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,12))
    def on_keypressed(event, plt=plt, fig=fig):
        key = event.key
        if key=='q':
            plt.close(fig)
            #sys.exit(0)
    fig.canvas.mpl_connect('key_press_event', on_keypressed)
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
    title += ',\n flyback ratio=%s' % (mirror.K / mirror.N)
    plt.title(title)

    plt.plot (vx_pos[:t_start],vy_pos[:t_start], '-',
              vx_pos[t_start:-t_end], vy_pos[t_start:-t_end], '-',
              vx_pos[-t_end:], vy_pos[-t_end:], '-')
    
    plt.subplot(plots[1])
    
    plt.plot (vx_target, '-', vy_target, '-')
    plt.plot (vx_input[::output_input_clock_ratio], '-', vy_input[::output_input_clock_ratio], '-')
    plt.plot (vx_pos, '-', vy_pos, '-')
    if M <= 10:
        mk_state_lines(plt, mirror, T = T/output_input_clock_ratio)
    plt.legend(['vx_target', 'vy_target', 'vx_input', 'vy_input', 'vx_pos', 'vy_pos'])

    
    plt.subplot (plots[2])
    plt.plot (vx_err, '-', vy_err, '-')
    #plt.plot (vx_err2, '-', vy_err2, '-')
    if M <= 10:
        mk_state_lines(plt, mirror, T = T/output_input_clock_ratio)
    plt.axhline(0)
    plt.legend(['vx_err', 'vy_err', 'vx_err2', 'vy_err2'][:2])


    plt.subplot (plots[3])
    plt.plot(i_target_av, '-', j_target_av, '-', 
             i_input_av, '-', j_input_av,'-',
             i_pos_av,'-',j_pos_av,'-',)
    if M <= 10:
        mk_state_lines(plt, mirror, T=1)
    plt.legend (['i_target', 'j_target', 'i_input', 'j_input', 'i_pos', 'j_pos'])

    plt.subplot (plots[4])

    if 1:
        t0 = 0
        t1 = t0 + K0
        for j in range(M):
            if j:
                t0 = t1
                t1 = t0 + K
            t0 = t1
            t1 = t0 + N
            plt.plot(range(t0, t1+1), i_err_av[t0:t1+1], '-b')
            #plt.plot(range(t0, t1+1), j_err_av[t0:t1+1], '-g')
        t0 = t1
        t1 = t0 + K1
        if M <= 10:
            mk_state_lines(plt, mirror, T=1)
        plt.axhline(0)
        plt.axvline(0)
        plt.legend (['i_err', 'j_err'][:1])

    plt.draw()

    return plt
    #plt.show()        

def run_mirrors(ao_task, ai_task, do_task, vx, vy, output_input_clock_ratio):
    arr = numpy.array([vx, vy]).T.ravel()
    #print 'MIRROR RUNNER:'

    print 'WRITING DATA(shape=%s)' % (arr.shape,),
    sys.stdout.flush()
    ao_task.write(arr, auto_start=False)

    print 'READY',
    sys.stdout.flush()
    do_task.start()
    do_task.write(0) 
    ao_task.start()
    ai_task.start()

    print 'START',
    sys.stdout.flush()
    do_task.write(1)

    print 'RUNNING',
    sys.stdout.flush()
    ao_task.wait_until_done()

    print 'FINISH',
    sys.stdout.flush()
    do_task.write(0)

    print 'READING',
    sys.stdout.flush()
    data = ai_task.read().T
    print 'GOT DATA(shape=%s)' % (data.shape,),

    print 'STOPPING',
    sys.stdout.flush()
    do_task.stop()
    ai_task.stop()
    ao_task.stop()
    print 'DONE.'
    sys.stdout.flush()

    vx_err = -data[0] * 2
    vy_err = -data[1] * 2

    vx_out = numpy.zeros(vx_err.shape, float)
    vy_out = numpy.zeros(vy_err.shape, float)

    if output_input_clock_ratio==1:
        if 1:
            vx_out[:] = vx - vx_err
            vy_out[:] = vy - vy_err
        else:
            #print 'Discarding first error measurments:',vx_err[0], vy_err[0]
            vx_out[:-1] = vx[:-1] - vx_err[1:]
            vy_out[:-1] = vy[:-1] - vy_err[1:]
            vx_out[-1] = vx[-1] - vx_err[-1]
            vy_out[-1] = vy[-1] - vy_err[-1]
        #print 'Max diff:', numpy.diff(abs(vx_out)).max(), numpy.diff(abs(vy_out)).max(),\
        #    numpy.diff(abs(vx)).max(), numpy.diff(abs(vy)).max()
    else:
        vx_out[:] = vx[::output_input_clock_ratio] - vx_err
        vy_out[:] = vy[::output_input_clock_ratio] - vy_err

    return vx_out, vy_out, vx_err, vy_err

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

def average(arr, T):
    if T==1:
        return arr
    r = numpy.zeros(arr.size // T, dtype=float)
    S = r.size
    for i in range(T):
        r += arr[i:S*T:T]
    return r / T


def main ():
    parser = OptionParser()
    set_run_mirrors_options(parser)
    if hasattr(parser, 'runner'):
        parser.runner = runner
    options, args = parser.parse_args()
    runner(parser, options, args)
    return

if __name__=="__main__":
    main()
