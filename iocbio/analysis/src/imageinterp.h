/*
  Header file for imageinterp.c.

  Author: Pearu Peterson
  Created: May 2011
 */

#ifndef IMAGEINTERP_H
#define IMAGEINTERP_H

#ifdef __cplusplus
extern "C" {
#endif

extern void imageinterp_get_roi(int image_width, int image_height, double *image,
				int i0, int j0, int i1, int j1, double width,
				int roi_width, int roi_height, double *roi
				);
#ifdef __cplusplus
}
#endif

#endif
