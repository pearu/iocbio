
from __future__ import division

from configuration import sample_clock_max_rate,  sample_clock_max_user_rate, \
    sample_clock_min_user_rate, pixel_min_size, pixel_rate_factor,\
    pixel_min_tune_rate, pixel_flyback_rate

class SampleClock:

    def __init__ (self, pixel_time, pixel_size):
        self.pixel_time = pixel_time #sec
        self.pixel_size = pixel_size #m
        #print self.get_optimal_scanning_parameters()

    def get_key (self):
        return (self.__class__.__name__, self.pixel_time, self.pixel_size)

    def __hash__ (self):
        return hash(self.get_key ())

    def get_optimal_scanning_parameters(self, verbose=True):
        """
        Returns a 3-tuple:
          (samples_per_pixel, clock_rate, min_tune_clock_rate)
        """
        ticks_per_pixel_max = int(self.pixel_time * sample_clock_max_rate)
        assert ticks_per_pixel_max>0,`ticks_per_pixel_max, pixel_time, sample_clock_max_rate`
        samples_per_pixel_min = int(self.pixel_time * sample_clock_min_user_rate) or 1
        samples_per_pixel_max =  min(int(self.pixel_size / pixel_min_size) or 1, ticks_per_pixel_max)

        pixel_rate = self.pixel_size / self.pixel_time

        if verbose:
            print 'Pixel rate: %.3e m/s' % (pixel_rate)
            
            print 'Flyback ratio: %s' % (pixel_flyback_rate / pixel_rate)
            
            print 'Samples per pixel range: [%s (to avoid stepping), %s (to avoid stopping)] <= %s (defined by hw clock max rate)' \
            % (samples_per_pixel_min, samples_per_pixel_max, ticks_per_pixel_max)

        if samples_per_pixel_min*pixel_rate_factor > samples_per_pixel_max:
            if verbose:
                print 'Switching to one sample per one pixel mode'
            samples_per_pixel = 1
            ticks_per_sample = ticks_per_pixel_max // samples_per_pixel
            pixel_rate = sample_clock_max_rate // (samples_per_pixel * ticks_per_sample)
            self.clock_rate = pixel_rate
            return samples_per_pixel, pixel_rate, pixel_rate
        else:
            if samples_per_pixel_min > samples_per_pixel_max:
                if verbose:
                    print 'Forcing continuous mirror movement at the expense of extra nof samples.'
                samples_per_pixel_max = samples_per_pixel_min
            pixel_min_tune_rate_per_pixel = int(pixel_min_tune_rate / self.pixel_size)
            if verbose:
                print 'Minimal user tuning pixel rate: %s Hz' % (pixel_min_tune_rate_per_pixel)
            for samples_per_pixel in range (samples_per_pixel_min, samples_per_pixel_max+1):
                ticks_per_sample = ticks_per_pixel_max // samples_per_pixel
                next_ticks_per_sample = ticks_per_pixel_max // (samples_per_pixel+1)
                if ticks_per_sample == next_ticks_per_sample:
                    continue
                clock_rate = sample_clock_max_rate // (samples_per_pixel * ticks_per_sample)
                #print sample_clock_max_rate / clock_rate, clock_rate
                # minimal pixel rate that avoids stepping
                min_clock_rate = sample_clock_min_user_rate // samples_per_pixel
                if verbose:
                    print 'Samples per pixel: %s, ticks per sample: %s, actual pixel time: %s s, ticks per second: %s, min ticks per second: %s' \
                        % (samples_per_pixel, ticks_per_sample, 1/clock_rate, clock_rate,
                           min_clock_rate)
                if pixel_min_tune_rate_per_pixel > min_clock_rate:
                    break
            if verbose:
                print 'Flyback ratio: %s (how many times more pixels are needed for flyback compared to line scan)' % (samples_per_pixel_max / samples_per_pixel)


            self.clock_rate = clock_rate
            return samples_per_pixel, clock_rate, max(pixel_min_tune_rate_per_pixel, min_clock_rate)

    def get_flyback (self, samples_per_pixel):
        return 

if __name__ == '__main__':
    c = SampleClock(pixel_time = 1e-5, pixel_size = 10e-6)
