#!/bin/bash

iocbio.show -i cell2_mitogreen_px25.tif --view-3d=c,c,c --no-gui -o cell_original.png 
convert -trim cell_original.png cell_original.png

iocbio.show -i psf4_505_pxt50.tif --view-3d=25,480,550 --no-gui -o microspheres.png
convert -trim microspheres.png microspheres.png

#iocbio.estimate_psf -i psf4_505_pxt50.tif --save-intermediate-results -o psf_airy_478_water.tif --no-gui

iocbio.show -i psf_airy_478_water.tif --view-3d=14,c,c --no-gui -o psf.png 
convert -trim psf.png psf.png

#iocbio.deconvolve -i cell2_mitogreen_px25.tif -k psf_airy_478_water.tif -o result.tif \
#  --rltv-estimate-lambda --rltv-lambda-lsq-coeff=0 --max-nof-iterations=20 \
#  --no-degrade-input --save-intermediate-results --float-type=single --no-gui

iocbio.rowfile_plot -i result.tif.iocbio.deconvolve/deconvolve_data.txt --x-keys=count --y-keys=lambda_lsq,mse,tau1 --no-gui -o deconvolve_result_params.png 
convert -trim deconvolve_result_params.png deconvolve_result_params.png

iocbio.show -i result.tif.iocbio.deconvolve/result_7.tif --view-3d=c,c,c --no-gui -o deconvolve_result_7.png 
convert -trim deconvolve_result_7.png deconvolve_result_7.png 

#iocbio.deconvolve -i cell2_mitogreen_px25.tif -k psf_airy_478_water.tif -o result2.tif \
#  --no-rltv-estimate-lambda --rltv-lambda-lsq-coeff=0 --max-nof-iterations=20 \
#  --no-degrade-input --save-intermediate-results --float-type=single --rltv-lambda=6.4 \
#  --rltv-compute-lambda-lsq --no-gui

iocbio.show -i result2.tif.iocbio.deconvolve/result_8.tif --view-3d=c,c,c --no-gui -o deconvolve_result2_8.png
convert -trim deconvolve_result2_8.png deconvolve_result2_8.png

iocbio.rowfile_plot -i result2.tif.iocbio.deconvolve/deconvolve_data.txt --x-keys=count --y-keys=lambda_lsq,mse,tau1 --no-gui -o deconvolve_result2_params.png 
convert -trim deconvolve_result2_params.png deconvolve_result2_params.png