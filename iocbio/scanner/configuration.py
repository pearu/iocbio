
import numpy

# Sample clock of NI card (that applies to analog output):
sample_clock_max_rate = 10000000 #Hz

# Mirrors cannot move faster than 2kHz x 2kHz, so we define maximal
# user sample clock:
sample_clock_max_user_rate = 4000000 #Hz

# Mirror small angle response time is 100us, ie 10kHz.  To achive
# continuous movement of mirrors (that is, not giving time mirrors to
# stabilize), mirror positions should be updated faster than
# 100us. Say, 10 times faster:
sample_clock_min_user_rate = int(10*(1/100e-6)) #Hz

# Image area is maximal area that can be scanned for an image.
# It depends on the objective.
image_area_width =  623e-6 # m
image_area_height = 623e-6 # m

# Camera area is normalized image area that is used for
# selecting regions of interest.
camera_area_width = 1000  # au
camera_area_height = 1000 # au

# Scanner mirror maximal angle corresponding to borders of image
# area. Numeric value of mirror angle corresponds to the voltage value
# given as the input to mirror driver.
mirror_max_user_angle = 4.7 # deg, volts

# Maximal allowed voltage to mirror driver.
mirror_max_angle = 10  # deg, volts

# X and Y mirrors offsets are used to center image area to scanning
# area center.
mirror_x_offset = 0.0 # volts
mirror_y_offset = 0.0 # volts

# Mirror (position detector) short term repeatability is 8urad = 0.45
# mdeg. With total scan range of 10 degrees that corresponds to scan
# area 0.623mm width, the resolution is 10/0.45e-3 = 22222 < 2**16 (NI
# cards resolution).  The minimal pixel size corresponding to highest
# resolution of the mirror movement, is 0.028um.  So, the maximum
# number of samples per 1um pixel would be 1um / 0.028um = 36, for
# example.
mirror_min_step = 8e-6 * 180/numpy.pi # deg, volts

# Mirror settling time (step 0.1deg, 99% accuracy).
mirror_settling_time = 100e-6 # sec

# Mirror large angle settling time, assuming a step 1deg for Y mirror.
mirror_large_angle_settling_time = 200e-6

def get_mirror_settling_time(angle):
    return mirror_large_angle_settling_time + (mirror_large_angle_settling_time - mirror_settling_time)*(abs(angle) - 1)/(1-0.1)

# Mirror maximim overshoot angle during flyback
mirror_max_overshoot = 1.0 # deg, volts

# Minimal pixel size defined by the resolution of scanning mirrors.
pixel_min_size = min(image_area_width, image_area_height) * mirror_min_step / mirror_max_user_angle

# For long pixel times, it is not possible to avoid stepping when
# forcing mirrors minimal steps (which defines memory bounds for
# driver signal). In such cases it is better to have one sample
# per one pixel. Minimal pixel rate is defined as

pixel_min_rate = pixel_min_size * sample_clock_min_user_rate

# and if pixel rate (defined as pixel size/pixel time) is
# pixel_rate_factor smaller than minimal pixel rate, then
# the algorithm switches to one sample per one pixel mode.
pixel_rate_factor = 0.1

# Maximal controlled pixel speed.
pixel_max_rate = 2000 * image_area_width * mirror_max_angle / mirror_max_user_angle

# For high pixel speeds, mirror driver input needs to be
# tuned to achive desired mirror movement. Minimal tuning pixel rate
# specifies minimal rate that requires tuning. Smaller rates
# are assumed to provide mirror driver input that guarantees
# specified mirror trajectory. The parameter is used for
# estimating smallest clock rate at which tuning algorithm
# starts to iterate.
pixel_min_tune_rate = pixel_max_rate / 10

# The mirror driver input tuning algorithm uses given number of steps
# between minimal tuning pixel rate and given pixel rate.
mirror_tuning_steps = 4

# Flyback pixel rate. Flyback should be as fast as possible
# but with reasonable errors. 
pixel_flyback_rate = pixel_max_rate / 30




print '''
Configuration parameters:

Hardware sample clock rate:      %(sample_clock_max_rate)s Hz
Maximal user sample clock rate:  %(sample_clock_max_user_rate)s Hz
Minimal user sample clock rate:  %(sample_clock_min_user_rate)s Hz
Minimal pixel size (estimated):  %(pixel_min_size).3e m
Minimal pixel rate:              %(pixel_min_rate).3e m/s
Maximal pixel rate:              %(pixel_max_rate).3e m/s
Minimal tuning pixel rate:       %(pixel_min_tune_rate).3e m/s

''' % (locals())
