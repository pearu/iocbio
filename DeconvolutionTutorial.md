# Introduction #

This tutorial introduces iocbio.deconvolve program for deconvolving microscope image with measured point spread function (PSF).

Note that in the following all iocbio programs are run with ```--no-gui``` options. Discarding this option from a command line will create a graphical user interface for specifying various options. The old tutorial, http://sysbio.ioc.ee/download/software/ioc.microscope/tutorial/, shows how the GUI of various iocbio programs looks like.

Also note that all commands used for creating the data and figures for this document is available in http://iocbio.googlecode.com/svn/trunk/iocbio/microscope/tutorial/mk_figures.sh

# Microscope image #

As an example, let us consider the following microscope image of a rat cardiomyocyte with stained mitochondria that is acquired with a confocal microscope with photon counting detector ([compressed TIF image (23MB)](http://sysbio.ioc.ee/download/software/iocbio.microscope/cell2_mitogreen_px25.tif.gz)):

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/cell_original.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/cell_original.png' />
</a>

This view of a microscope image is created with the following command:
```
iocbio.show -i cell2_mitogreen_px25.tif --view-3d=c,c,c --no-gui
```

# Point spread function #

The point spread function was estimated from a cluster of microspheres (d=0.175um) measurements ([compressed TIF image (3.5MB)](http://sysbio/download/software/iocbio.microscope/psf4_505_pxt50.tif.gz)):

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/microspheres.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/microspheres.png' />
</a>

The estimated PSF ([compressed TIF image (64KB)](http://sysbio/download/software/iocbio.microscope/psf_airy_478_water.tif.gz)) is obtained by running:
```
iocbio.estimate_psf -i psf4_505_pxt50.tif --save-intermediate-results -o psf_airy_478_water.tif --no-gui
```
The result is shown below:

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/psf.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/psf.png' />
</a>


# Deconvolving microscope image #

To deconvolve microscope image with given PSF image, run
```
iocbio.deconvolve -i cell2_mitogreen_px25.tif -k psf_airy_478_water.tif -o result.tif \
  --rltv-estimate-lambda --rltv-lambda-lsq-coeff=0 --max-nof-iterations=20 \
  --no-degrade-input --save-intermediate-results --float-type=single --no-gui
```
that will show the following output:
```
Fixing cell2_mitogreen_px25.tif to /home/pearu/svn/iocbio/iocbio/microscope/tutorial/cell2_mitogreen_px25.tif
Fixing psf_airy_478_water.tif to /home/pearu/svn/iocbio/iocbio/microscope/tutorial/psf_airy_478_water.tif
PSF was zoomed by (0.40740740740740744, 0.17837891324424673, 0.17837891324424673)
Input image has signal-to-noise ratio 7.80550612875
Suggested RLTV regularization parameter: 5.50893168114[blocky]..7.68688141555[honeycomb]
Initial photon count: 184532944.000
Initial minimum: 0.000
Initial maximum: 82.000
lambda-opt= 6.40573451295
lambda-lsq-0= 0.840393
lambda-lsq-coeff= 7.62231091018
[=>                1/20                ] E/S/U/N=13/6793900/191323945/0, KLIC=0.584, LAM/MX=10.4/10.4, LEAK=-7.36%, MSE=9.68E-2, TAU1/2=0.188/0.280, TIME=72.1s
```
Notice that the signal-to-noise ratio (SNR) is estimated to 7.8 which is rather small. This can also be observed from the microscope image above: noise is well visible.

The deconvolution process can be characterized by running
```
iocbio.rowfile_plot -i result.tif.iocbio.deconvolve/deconvolve_data.txt --x-keys=count --y-keys=lambda_lsq,mse,tau1 --no-gui
```
that will show the evolution of estimated lambda\_lsq, mse (mean square error of convolve(estimate, psf) and input image), tau1 (a measure of convergence):

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result_params.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result_params.png' />
</a>

We suggest stopping the iteration after 5 steps of lambda\_lsq obtaining its maximum,
that is at 7th iteration in this particular case. The result of 7th iteration
is

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result_7.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result_7.png' />
</a>

that is created with:
```
iocbio.show -i result.tif.iocbio.deconvolve/result_7.tif --view-3d=c,c,c --no-gui
```
Notice that the noise is considerably reduced.

In the above we let iocbio.deconvolve to estimate regularization parameter. Since we have estimated the noise-to-signal ratio of microscope image, we can use the following formula
```
  lambda = 50/SNR
```
to estimate the value for regularization parameter lambda, that is, 50/7.8=6.4.
Next we re-run deconvolution with fixed lambda value as follows:
```
iocbio.deconvolve -i cell2_mitogreen_px25.tif -k psf_airy_478_water.tif -o result2.tif \
  --no-rltv-estimate-lambda --rltv-lambda-lsq-coeff=0 --max-nof-iterations=20 \
  --no-degrade-input --save-intermediate-results --float-type=single --rltv-lambda=6.4 \
  --rltv-compute-lambda-lsq --no-gui
```
The 8th iteration result is

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result2_8.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result2_8.png' />
</a>

that was suggested by the max lambda\_lsq + 5 stopping criteria, see:

<a href='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result2_params.png'>
<img width='400px,' src='http://sysbio.ioc.ee/download/software/iocbio.microscope/deconvolve_result2_params.png' />
</a>