/*
  Provides C functions: imageinterp_get_roi, imageinterp_get_roi_corners.
  For Python extension module, compile with -DPYTHON_EXTENSION.
  Author: Pearu Peterson
  Created: June 2011
 */

#include <math.h>
#include <string.h>

/*
  imageinterp_get_roi function returns a subimage (roi) of an image
  using interpolation.
  
  Input parameters
  ----------

  image_width, image_height : int
    Specify the number of columns and rows of the image.
  image : double*
    Specify pointer to image array that uses row-storage order.
  image_di_size, image_dj_size : double
    Specify image pixel sizes in image units.
  i0, j0, i1, j1 : double
    Specify roi center line.
  width : double
    Specify roi width in image units.
  roi_width, roi_height : int
    Specify the number of columns and rows of the roi image.
  interpolation : int
    Specify interpolation method: 0=nearest-neighbor interpolation,
    1=bilinear, 2=bicubic

  Output parameters
  -----------------

  roi : double *
    Specify pointer to roi array (row-storage order)

  Notes
  -----

  If the image pixel sizes are dx and dy then the roi pixel
  sizes will be
    DX = L / roi_width
    DY = width / roi_height
  where L = hypot((i1-i0)*dx, (j1-j0)*dy)
 */

/* http://en.wikipedia.org/wiki/Cubic_interpolation */
inline
double CINT(double x, double pm1, double p0, double p1, double p2)
{
  return 0.5 * (x*((2-x)*x-1)*(pm1) + (x*x*(3*x-5)+2)*(p0) + (x*((4-3*x)*x+1))*p1 + ((x-1)*x*x)*p2);
}

void imageinterp_get_roi(int image_width, int image_height, double *image,
			 double di_size, double dj_size,
			 double i0, double j0, double i1, double j1, double width,
			 int roi_width, int roi_height, double *roi,
			 int interpolation
			 )
{
#define GET_IMAGE_PTR(I, J) (image + (J)*image_width + (I))
#define GET_IMAGE(I, J) (((I)>=0 && (I)<image_width && (J)>=0 && (J)<image_height) ? (*(image + (J)*image_width + (I))) : 0.0)
#define GET_ROI(I, J) (*(roi + (J)*roi_width + (I)))
  int ri, rj;
  int ii, ji, ii1, ji1;
  double i, j;
  double l = hypot((i1-i0)*di_size, (j1-j0)*dj_size);
  double dt = (roi_width>1?1.0 / (roi_width-1):0.0);
  double ds = (roi_height>1?1.0 / (roi_height-1):0.0);

  double di = -(j1-j0)/l*width*0.5*dj_size/di_size;
  double dj = +(i1-i0)/l*width*0.5*di_size/dj_size;
  double t;
  double s;
  double v;
  double is, js;
  double rmi, rmj;
  double bm1, b0, b1, b2;

  //printf("imageinterp_get_roi: i0,j0,i1,j1,rw,rh=%f,%f,%f,%f,%d,%d\n",i0,j0,i1,j1,roi_width, roi_height);

  if (j0==j1 && roi_width == i1-i0+1)
    {
      v = width*0.5/dj_size;
      if (fabs(v-round(v))<1e-12 && round(2*v+1)==roi_height)
	{
	  ji = round(j0 - v);
	  for (rj=0; rj<roi_height; ++rj, ++ji)
	    {
	      if (ji<0 || ji>=image_height)
		memset(&GET_ROI(0, rj), 0, sizeof(double)*roi_width);
	      else if (i1<image_width)
		memcpy(&GET_ROI(0, rj), GET_IMAGE_PTR((int)i0, ji), sizeof(double)*roi_width);
	      else
		for (ii=i0; ii<=i1; ++ii)
		  GET_ROI(ii-(int)i0, rj) = GET_IMAGE(ii, ji);
	    }
	}
      else
	{
	  for (rj=0; rj<roi_height; ++rj)
	    {
	      s = (double)rj * ds;
	      j = (1.0-s)*(j0 - dj) + s*(j0 + dj);
	      //j = j0 + (2*rj/(roi_height-1)-1)*width*0.5/dj_size;
	      ji = floor(j);
	      rmj = j-ji;
	      ji1 = ji+1;
	      for (ri=0; ri<roi_width; ++ri)
		{
		  t = (double)ri * dt;
		  i = i0 + t*(i1-i0);
		  ii = floor(i);
		  rmi = i-ii;
		  ii1 = ii+1;
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
  else if (j0==j1)
    {
      for (rj=0; rj<roi_height; ++rj)
	{
	  s = (double)rj * ds;
	  j = (1.0-s)*(j0 - dj) + s*(j0 + dj);
	  ji = floor(j);
	  rmj = j-ji;
	  ji1 = ji+1;
	  if (rmj==0.0)
	    {
	      for (ri=0; ri<roi_width; ++ri)
		{
		  t = (double)ri * dt;
		  i = i0 + t*(i1-i0);
		  ii = floor(i);
		  rmi = i-ii;
		  ii1 = ii+1;
		  if (rmi==0.0)
		    {
		      v = GET_IMAGE(ii, ji);
		    }
		  else
		    {
		      switch (interpolation)
			{
			case 1: /* bilinear interpolation */
			  v = GET_IMAGE(ii,ji)*(1.0-rmi)+GET_IMAGE(ii1,ji)*rmi;
			  break;
			case 2: /* bicubic interpolation */
			  v = CINT(rmi, GET_IMAGE(ii-1, ji), GET_IMAGE(ii, ji), GET_IMAGE(ii+1, ji), GET_IMAGE(ii+2, ji));
			  break;
			case 0:
			default: /* nearest-neighbor interpolation */
			  v = GET_IMAGE(ii, ji);
			}
		    }
		  GET_ROI(ri, rj) = v;
		}
	    }
	  else
	    {
	      for (ri=0; ri<roi_width; ++ri)
		{
		  t = (double)ri * dt;
		  i = i0 + t*(i1-i0);
		  ii = floor(i);
		  rmi = i-ii;
		  ii1 = ii+1;
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
  else
    {
      for (rj=0; rj<roi_height; ++rj)
	{
	  s = (double)rj * ds;
	  is = (1.0-s)*(i0 - di) + s*(i0 + di);
	  js = (1.0-s)*(j0 - dj) + s*(j0 + dj);
	  for (ri=0; ri<roi_width; ++ri)
	    {
	      t = (double)ri * dt;
	      i = (1.0-t)*is + t*(is+i1-i0);
	      j = (1.0-t)*js + t*(js+j1-j0);
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
				 int i0, int j0, int i1, int j1, double width,
				 double *lli, double *llj,
				 double *lri, double *lrj,
				 double *uri, double *urj,
				 double *uli, double *ulj)
{
  double l = hypot((i1-i0)*di_size, (j1-j0)*dj_size);
  double di = -(double)(j1-j0)/l*width*0.5*dj_size/di_size;
  double dj = +(double)(i1-i0)/l*width*0.5*di_size/dj_size;
  *lli = i0 - di;
  *llj = j0 - dj;
  *lri = i1 - di;
  *lrj = j1 - dj;
  *uri = i1 + di;
  *urj = j1 + dj;
  *uli = i0 + di;
  *ulj = j0 + dj;
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
  double i0, i1, j0, j1;
  double di_size, dj_size;
  double roi_di_size, roi_dj_size;
  int roi_width = 0, roi_height = 0;
  int interpolation;
  double w, l;
  npy_intp roi_dims[] = {0, 0};
  if (!PyArg_ParseTuple(args, "O(dd)(dd)(dd)di", &image, &di_size, &dj_size, &i0, &i1, &j0, &j1, &w, &interpolation))
    return NULL;
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_DOUBLE && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"1st argument must be rank-2 double array object");
      return NULL;
    }
  l = hypot((i1-i0)*di_size, (j1-j0)*dj_size);
  if (j1==j0 && 1)
    {
      roi_width = i1-i0+1;
    }
  else
    {
      roi_width = (int)(hypot(i1-i0, j1-j0)+1);
    }
  roi_height = ceil((double)roi_width * w / (l+1.0)) + 1;
  roi_di_size = l / roi_width;
  roi_dj_size = w / roi_height;
  roi_dims[0] = roi_height;
  roi_dims[1] = roi_width;
  roi = PyArray_SimpleNew(2, roi_dims, PyArray_DOUBLE);
  imageinterp_get_roi(PyArray_DIMS(image)[1], PyArray_DIMS(image)[0], PyArray_DATA(image),
		      di_size, dj_size,
		      i0, j0, i1, j1, w,
		      roi_width, roi_height, PyArray_DATA(roi),
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
  {"get_roi", py_imageinterp_get_roi, METH_VARARGS, "get_roi(image, (di, dj), (i0, i1), (j0, j1), w, interpolation) -> roi, (rdi,rdj)"},
  {"get_roi_corners", py_imageinterp_get_roi_corners, METH_VARARGS, "get_roi_corners((w,h), (di,dj), (i0, i1), (j0, j1), w) -> (ll,lr,ur,ul)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initimageinterp(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module imageinterp (failed to import numpy)"); return;}
  m = Py_InitModule3("imageinterp", module_methods, "Provides get_roi, get_roi_corners functions.");
}
#endif
