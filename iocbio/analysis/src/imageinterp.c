/*
  Implements imageinterp_get_roi function.
  Author: Pearu Peterson
  Created: June 2011
 */

#include <math.h>

void imageinterp_get_roi(int image_width, int image_height, double *image,
			 int i0, int j0, int i1, int j1, double width,
			 int roi_width, int roi_height, double *roi
			 )
{
#define GET_IMAGE(I, J) (*(image + (J)*image_width + (I)))
#define GET_ROI(I, J) (*(roi + (J)*roi_width + (I)))
  int ri, rj;
  int ii, ji, ii1, ji1;
  double i, j;
  double l = hypot(i1-i0, j1-j0);
  double dt = 1.0 / (roi_width-1);
  double ds = 1.0 / (roi_height-1);

  double di = (double)(j1-j0)/l*width*0.5;
  double dj = -(double)(i1-i0)/l*width*0.5;
  double t;
  double s;
  double v;
  double is, js;

  for (rj=0; rj<roi_height; ++rj)
    {
      s = rj * ds;
      is = (1.0-s)*(i0 + di) + s*(i0 - di);
      js = (1.0-s)*(j0 - dj) + s*(j0 + dj);
      for (ri=0; ri<roi_width; ++ri)
	{
	  t = ri * dt;
	  i = (1.0-t)*is + t*(is+i1-i0);
	  j = (1.0-t)*js + t*(js+j1-j0);
	  ii = (int)i;
	  ji = (int)j;
	  ii1 = ii+1;
	  ji1 = ji+1;
	  if (ii>=0 && ii1<image_width
	      && ji>=0 && ji1<image_height)
	    v = (GET_IMAGE(ii,ji)*(1.0-(i-ii))+GET_IMAGE(ii1,ji)*(i-ii))*(1.0-(j-ji))
	      +(GET_IMAGE(ii,ji1)*(1.0-(i-ii))+GET_IMAGE(ii1,ji1)*(i-ii))*(j-ji);
	  else
	    v = 0.0;
	  GET_ROI(ri, rj) = v;
	}
    }
}

#ifdef PYTHON_EXTENSION

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif


static PyObject *py_imageinterp_get_roi(PyObject *self, PyObject *args)
{
  PyObject* image = NULL;
  PyObject* roi = NULL;
  int i0, i1, j0, j1, roi_width = 0, roi_height = 0;
  double w;
  npy_intp roi_dims[] = {0, 0};
  if (!PyArg_ParseTuple(args, "O(ii)(ii)d", &image, &i0, &j0, &i1, &j1, &w))
    return NULL;
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_DOUBLE && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"1st argument must be rank-2 double array object");
      return NULL;
    }
  roi_width = (int)(hypot(i1-i0, j1-j0)+1);
  roi_height = w + 1;
  roi_dims[0] = roi_height;
  roi_dims[1] = roi_width;
  roi = PyArray_SimpleNew(2, roi_dims, PyArray_DOUBLE);
  imageinterp_get_roi(PyArray_DIMS(image)[1], PyArray_DIMS(image)[0], PyArray_DATA(image),
		      i0, j0, i1, j1, w,
		      roi_width, roi_height, PyArray_DATA(roi)
		      );
  return roi;
}

static PyMethodDef module_methods[] = {
  {"get_roi", py_imageinterp_get_roi, METH_VARARGS, "get_roi(image, (i0, j0), (i1, j1), w) -> roi"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initimageinterp(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module imageinterp (failed to import numpy)"); return;}
  m = Py_InitModule3("imageinterp", module_methods, "Provides get_roi function.");
}
#endif
