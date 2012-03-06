/*
  Provides C functions: imageinterp_get_roi, imageinterp_get_roi_corners.
  For Python extension module, compile with -DPYTHON_EXTENSION.
  Author: Pearu Peterson
  Created: June 2011
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
/*
  void imageinterp_get_roi(image_width, image_height, image,
                           di_size, dj_size, i0, j0, i1, j1,
                           roi_width, roi_height, roi, roi_di_size, roi_dj_size,
			   interpolation )

  imageinterp_get_roi function returns a subimage (roi) of an image
  using interpolation.
  
  Input parameters
  ----------

  image_width, image_height : int
    Specify the number of columns and rows of the image.
  image : double*
    Specify pointer to image array that uses row-storage order.
  image_di_size, image_dj_size : double
    Specify image pixel sizes in image units (e.g. um).
  i0, j0, i1, j1 : double
    Specify roi center line in pixel units.
  roi_width, roi_height : int
    Specify the number of columns and rows of the roi image.
  interpolation : int
    Specify interpolation method: 0=nearest-neighbor interpolation,
    1=bilinear, 2=bicubic.

  Output parameters
  -----------------

  roi : double*
    Specify pointer to roi array (row-storage order) that will be
    filled with image data.
  roi_di_size, roi_dj_size : double*
    Roi pixel sizes::

      rdi = hypot((i1-i0)*di, (j1-j0)*dj) / hypot (i1-i0,j1-j0)
      rdj = hypot((i1-i0)*di, (j1-j0)*dj) / hypot ((i1-i0)*dj/di,(j1-j0)*di/dj)
*/

/* http://en.wikipedia.org/wiki/Cubic_interpolation */
inline
double CINT(double x, double pm1, double p0, double p1, double p2)
{
  return 0.5 * (x*((2-x)*x-1)*(pm1) + (x*x*(3*x-5)+2)*(p0) + (x*((4-3*x)*x+1))*p1 + ((x-1)*x*x)*p2);
}

void imageinterp_get_roi(int image_width, int image_height, double *image,
			 double di_size, double dj_size,
			 double i0, double j0, double i1, double j1,
			 int roi_width, int roi_height, double *roi,
			 double* roi_di_size, double* roi_dj_size,
			 int interpolation
			 )
{
#define GET_IMAGE_PTR(I, J) (image + (J)*image_width + (I))
#define GET_IMAGE(I, J) (((I)>=0 && (I)<image_width && (J)>=0 && (J)<image_height) ? (*(image + (J)*image_width + (I))) : 0.0)
#define GET_ROI(I, J) (*(roi + (J)*roi_width + (I)))
  
  int ri, rj;
  int ii, ji, ii1, ji1;
  double i, j;
  double v;
  double rmi, rmj;
  double bm1, b0, b1, b2;
  double r = dj_size / di_size;
  //printf("imageinterp_get_roi: i0,j0,i1,j1,rw,rh=%f,%f,%f,%f,%d,%d\n",i0,j0,i1,j1,roi_width, roi_height);
 
  if (j1==j0 && i1>i0 && i1-i0+1==roi_width)
    {
      *roi_di_size = di_size;
      *roi_dj_size = dj_size;
      for (rj=0; rj<roi_height; ++rj)
	memcpy(&GET_ROI(0, rj), GET_IMAGE_PTR((int)i0, (int)j0 + rj - (roi_height/2)), sizeof(double)*roi_width);
    }
  else
    {

      double dti = (i1-i0)/hypot ((i1-i0),(j1-j0));
      double dtj = (j1-j0)/hypot ((i1-i0),(j1-j0));
      double dsi = -(j1-j0)*r*r/hypot ((j1-j0)*r*r,(i1-i0));
      double dsj = (i1-i0)/hypot ((j1-j0)*r*r,(i1-i0));
      double l = hypot((i1-i0)*di_size, (j1-j0)*dj_size);
      *roi_di_size = l/hypot ((i1-i0),(j1-j0));
      *roi_dj_size = l*r/hypot ((j1-j0)*r*r,(i1-i0));

      for (rj=0; rj<roi_height; ++rj)
	{
	  for (ri=0; ri<roi_width; ++ri)
	    {
	      i = i0 + ri*dti + (rj-(roi_height/2))*dsi;
	      j = j0 + ri*dtj + (rj-(roi_height/2))*dsj;
	      ii = floor(i);
	      ji = floor(j);
	      rmi = i-ii;
	      rmj = j-ji;
	      ii1 = ii+1;
	      ji1 = ji+1;
	      switch (interpolation)
		{
		case 1: /* bilinear interpolation */
		  v = (GET_IMAGE(ii,ji)*(1.0-rmi)+GET_IMAGE(ii1,ji)*rmi)*(1.0-rmj)
		    +(GET_IMAGE(ii,ji1)*(1.0-rmi)+GET_IMAGE(ii1,ji1)*rmi)*rmj;
		  break;
		case 2: /* bicubic interpolation */
		  bm1 = CINT(rmi, GET_IMAGE(ii-1, ji-1), GET_IMAGE(ii, ji-1), GET_IMAGE(ii+1, ji-1), GET_IMAGE(ii+2, ji-1));
		  b0 = CINT(rmi, GET_IMAGE(ii-1, ji), GET_IMAGE(ii, ji), GET_IMAGE(ii+1, ji), GET_IMAGE(ii+2, ji));
		  b1 = CINT(rmi, GET_IMAGE(ii-1, ji+1), GET_IMAGE(ii, ji+1), GET_IMAGE(ii+1, ji+1), GET_IMAGE(ii+2, ji+1));
		  b2 = CINT(rmi, GET_IMAGE(ii-1, ji+2), GET_IMAGE(ii, ji+2), GET_IMAGE(ii+1, ji+2), GET_IMAGE(ii+2, ji+2));
		  v = CINT(rmj, bm1, b0, b1, b2);
		  break;
		case 0:
		default: /* nearest-neighbor interpolation */
		  v = GET_IMAGE(ii, ji);
		}	      
	      GET_ROI(ri, rj) = v;      
	    }
	}
    }
}

void imageinterp_get_roi_corners(int image_width, int image_height,
				 double di_size, double dj_size,
				 int i0, int j0, int i1, int j1, double height,
				 double *lli, double *llj,
				 double *lri, double *lrj,
				 double *uri, double *urj,
				 double *uli, double *ulj)
{

  double r = dj_size / di_size;
  double dti = (i1-i0)/hypot ((i1-i0),(j1-j0));
  double dtj = (j1-j0)/hypot ((i1-i0),(j1-j0));
  double dsi = -(j1-j0)*r*r/hypot ((j1-j0)*r*r,(i1-i0));
  double dsj = (i1-i0)/hypot ((j1-j0)*r*r,(i1-i0));
  int roi_width = (int)(hypot((i1-i0),(j1-j0))) + 1;
  int roi_height = (int)height + 1;

  *lli = i0 - (roi_height/2)*dsi;
  *llj = j0 - (roi_height/2)*dsj;
  *lri = i0 + roi_width*dti - (roi_height/2)*dsi;
  *lrj = j0 + roi_width*dtj - (roi_height/2)*dsj;
  *uri = i0 + roi_width*dti + (roi_height-(roi_height/2))*dsi;
  *urj = j0 + roi_width*dtj + (roi_height-(roi_height/2))*dsj;
  *uli = i0 + (roi_height-(roi_height/2))*dsi;
  *ulj = j0 + (roi_height-(roi_height/2))*dsj;

}


#ifdef PYTHON_EXTENSION

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

static char py_imageinterp_get_roi_doc[] = "\
  get_roi(image, (di, dj), (i0, i1), (j0, j1), w, interpolation) -> roi, (rdi,rdj)\n\
  imageinterp_get_roi function returns a subimage (roi) of an image\n\
  using interpolation.\n\
  \n\
  Parameters\n\
  ----------\n\
  image : numpy.ndarray\n\
    Specify image array that uses row-storage order.\n\
  image_di_size, image_dj_size : float\n\
    Specify image pixel sizes in image units (e.g. um).\n\
  i0, j0, i1, j1 : float\n\
    Specify roi center line in pixel units.\n\
  line_width : int\n\
    Specify the width of roi center line.\n\
  interpolation : int\n\
    Specify interpolation method: 0=nearest-neighbor interpolation,\n\
    1=bilinear, 2=bicubic\n\
\n\
  Returns\n\
  -------\n\
\n\
  roi : numpy.ndarray\n\
    Roi array (row-storage ordered) with image data. The size of roi\n\
    is (line width, int(hypot((i1-i0),(j1-j0))) + 1).\n\
  roi_di_size, roi_dj_size : float\n\
    Roi pixel sizes::\n\
\n\
      rdi = hypot((i1-i0)*di, (j1-j0)*dj) / hypot (i1-i0,j1-j0)\n\
      rdj = hypot((i1-i0)*di, (j1-j0)*dj) / hypot ((i1-i0)*dj/di,(j1-j0)*di/dj)\
";

static PyObject *py_imageinterp_get_roi(PyObject *self, PyObject *args)
{
  PyObject* image = NULL;
  PyObject* roi = NULL;
  double i0, i1, j0, j1;
  double di_size, dj_size;
  double roi_di_size, roi_dj_size;
  int roi_width = 0, roi_height = 0;
  int interpolation;
  int line_width;
  npy_intp roi_dims[] = {0, 0};
  if (!PyArg_ParseTuple(args, "O(dd)(dd)(dd)ii", &image, &di_size, &dj_size, &i0, &i1, &j0, &j1, &line_width, &interpolation))
    return NULL;
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_DOUBLE && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"1st argument must be rank-2 double array object");
      return NULL;
    }
  roi_height = line_width + 1;
  if (j1==j0)
    roi_width = (i1>i0?i1-i0:i0-i1) + 1;
  else
    roi_width = (int)(hypot((i1-i0),(j1-j0))) + 1;

  roi_dims[0] = roi_height;
  roi_dims[1] = roi_width;
  roi = PyArray_SimpleNew(2, roi_dims, PyArray_DOUBLE);
  imageinterp_get_roi(PyArray_DIMS(image)[1], PyArray_DIMS(image)[0], PyArray_DATA(image),
		      di_size, dj_size,
		      i0, j0, i1, j1,
		      roi_width, roi_height, PyArray_DATA(roi), &roi_di_size, &roi_dj_size,
		      interpolation
		      );
  return Py_BuildValue("N(dd)", roi, roi_di_size, roi_dj_size);
}

static PyObject *py_imageinterp_get_roi_corners(PyObject *self, PyObject *args)
{
  int i0, i1, j0, j1, image_width, image_height;
  double w;
  double di_size, dj_size;
  double lli, llj, lri, lrj, uri, urj, uli, ulj;

  if (!PyArg_ParseTuple(args, "(ii)(dd)(ii)(ii)d", &image_width, &image_height, &di_size, &dj_size, &i0, &i1, &j0, &j1, &w))
    return NULL;
  imageinterp_get_roi_corners(image_width, image_height,
			      di_size, dj_size,
			      i0, j0, i1, j1, w,
			      &lli, &llj, &lri, &lrj, &uri, &urj, &uli, &ulj
			      );
  return Py_BuildValue("(dd)(dd)(dd)(dd)",lli, llj, lri, lrj, uri, urj, uli, ulj);
}

static PyMethodDef module_methods[] = {
  {"get_roi", py_imageinterp_get_roi, METH_VARARGS, py_imageinterp_get_roi_doc},
  {"get_roi_corners", py_imageinterp_get_roi_corners, METH_VARARGS, "get_roi_corners((w,h), (di,dj), (i0, i1), (j0, j1), w) -> (ll,lr,ur,ul)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initimageinterp(void) 
{
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module imageinterp (failed to import numpy)"); return;}
  Py_InitModule3("imageinterp", module_methods, "Provides get_roi, get_roi_corners functions.");
}
#endif
