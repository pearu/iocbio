/*
  Header file for imageinterp.c. See the source for documentation.

  Author: Pearu Peterson
  Created: May 2011
 */

#ifndef IMAGEINTERP_H
#define IMAGEINTERP_H

#ifdef __cplusplus
extern "C" {
#endif

extern void imageinterp_get_roi(int image_width, int image_height, double *image,
				double di_size, double dj_size,
				double i0, double j0, double i1, double j1,
				int roi_width, int roi_height, double *roi,
				double *roi_di_size, double *roi_dj_size,
				int interpolation
				);

#ifdef __cplusplus
}
#endif

#endif
