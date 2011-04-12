/*
  Implements interpolate_bilinear function.
  Author: Pearu Peterson
  Created: September 2009
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

// http://en.wikipedia.org/wiki/Cubic_interpolation
inline
double CINT(double x, double pm1, double p0, double p1, double p2)
{
  return 0.5 * (x*((2-x)*x-1)*(pm1) + (x*x*(3*x-5)+2)*(p0) + (x*((4-3*x)*x+1))*p1 + ((x-1)*x*x)*p2);
}

void interpolate_bicubic(int N, // nof line points
			 int M, // nof image colums 
			 double *line, 
			 double* image,
			 double i0, double j0, // line start 
			 double di, double dj  // line direction
			 )
{
  int n, i, j;
  double ir, jr;
  double pm1, p0, p1, p2;
  double bm1, b0, b1, b2;
  for (n = 0; n < N; ++n)
    {
      ir = i0 + n * di;
      jr = j0 + n * dj;
      i = ir;
      j = jr;

      pm1 = (image+(i-1)*M)[j-1];
      p0 = (image+(i)*M)[j-1];
      p1 = (image+(i+1)*M)[j-1];
      p2 = (image+(i+2)*M)[j-1];
      bm1 = CINT(ir-i, pm1, p0, p1, p2);

      pm1 = (image+(i-1)*M)[j];
      p0 = (image+(i)*M)[j];
      p1 = (image+(i+1)*M)[j];
      p2 = (image+(i+2)*M)[j];
      b0 = CINT(ir-i, pm1, p0, p1, p2);

      pm1 = (image+(i-1)*M)[j+1];
      p0 = (image+(i)*M)[j+1];
      p1 = (image+(i+1)*M)[j+1];
      p2 = (image+(i+2)*M)[j+1];
      b1 = CINT(ir-i, pm1, p0, p1, p2);

      pm1 = (image+(i-1)*M)[j+2];
      p0 = (image+(i)*M)[j+2];
      p1 = (image+(i+1)*M)[j+2];
      p2 = (image+(i+2)*M)[j+2];
      b2 = CINT(ir-i, pm1, p0, p1, p2);

      line[n] = CINT(jr-j, bm1, b0, b1, b2);
    }
}

inline
double interpolate_bilinear_at_point(int M, double* image, double ir, double jr)
{
  int i = ir, j=jr;
  double *p1 = image+i*M;
  double *p2 = p1 + M;
  return (ir-(i+1))*(p1[j]*(jr-(j+1)) - p1[j+1]*(jr-j)) + (-p2[j]*(jr-(j+1)) + p2[j+1]*(jr-j)) * (ir-i);
}

void interpolate_bilinear(int N, // nof line points
			  int M, // nof image colums 
			  double *line, 
			  double* image,
			  double i0, double j0, // line start 
			  double di, double dj  // line direction
			  )
{
  int n;
  for (n = 0; n < N; ++n)
    line[n] = interpolate_bilinear_at_point(M, image, i0 + n*di, j0+n*dj);
}

void interpolate_bilinear_at(int N, // nof line points
			     int M, // nof image colums 
			     double *line, 
			     double* image,
			     double* icoords, // point coordinates
			     double* jcoords
			  )
{
  int n;
  for (n = 0; n < N; ++n)
    line[n] =  interpolate_bilinear_at_point(M, image, icoords[n], jcoords[n]);
}


static PyObject *py_interpolate_bilinear(PyObject *self, PyObject *args)
{
  double i0, j0, di, dj;
  PyObject* line = NULL;
  PyObject* image = NULL;
  int N;
  int M;
  if (!PyArg_ParseTuple(args, "OO(dd)(dd)", &line, &image, &i0, &j0, &di, &dj))
    return NULL;
  if (!(PyArray_Check(line) && PyArray_TYPE(line) == PyArray_FLOAT64 && PyArray_NDIM(line)==1))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be rank-1 double array object");
      return NULL;
    }
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_FLOAT64 && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"second argument must be rank-2 double array object");
      return NULL;
    }
  N = PyArray_DIMS(line)[0];
  M = PyArray_DIMS(image)[1];
  interpolate_bilinear(N,M,PyArray_DATA(line),PyArray_DATA(image),i0,j0,di,dj);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *py_interpolate_bilinear_at(PyObject *self, PyObject *args)
{

  PyObject* line = NULL;
  PyObject* image = NULL;
  PyObject* icoords = NULL;
  PyObject* jcoords = NULL;
  int N;
  int M;
  if (!PyArg_ParseTuple(args, "OOOO", &line, &image, &icoords, &jcoords))
    return NULL;
  if (!(PyArray_Check(line) && PyArray_TYPE(line) == PyArray_FLOAT64 && PyArray_NDIM(line)==1))
    {
      PyErr_SetString(PyExc_TypeError,"1st argument must be rank-1 double array object");
      return NULL;
    }
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_FLOAT64 && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"2nd argument must be rank-2 double array object");
      return NULL;
    }
  if (!(PyArray_Check(icoords) && PyArray_TYPE(icoords) == PyArray_FLOAT64 && PyArray_NDIM(icoords)==1))
    {
      PyErr_SetString(PyExc_TypeError,"3rd argument must be rank-1 double array object");
      return NULL;
    }
  if (!(PyArray_Check(jcoords) && PyArray_TYPE(jcoords) == PyArray_FLOAT64 && PyArray_NDIM(jcoords)==1))
    {
      PyErr_SetString(PyExc_TypeError,"4th argument must be rank-1 double array object");
      return NULL;
    }
  N = PyArray_DIMS(line)[0];
  M = PyArray_DIMS(image)[1];
  interpolate_bilinear_at(N,M,PyArray_DATA(line),PyArray_DATA(image),PyArray_DATA(icoords), PyArray_DATA(jcoords));
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *py_interpolate_bilinear_at_point(PyObject *self, PyObject *args)
{
  PyObject* image = NULL;
  double i, j, r;
  int M;
  if (!PyArg_ParseTuple(args, "Odd", &image, &i, &j))
    return NULL;
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_FLOAT64 && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"1st argument must be rank-2 double array object");
      return NULL;
    }
  M = PyArray_DIMS(image)[1];
  r = interpolate_bilinear_at_point(M,PyArray_DATA(image), i, j);
  return Py_BuildValue("d", r);
}

static PyObject *py_interpolate_bicubic(PyObject *self, PyObject *args)
{
  double i0, j0, di, dj;
  PyObject* line = NULL;
  PyObject* image = NULL;
  int N;
  int M;
  if (!PyArg_ParseTuple(args, "OO(dd)(dd)", &line, &image, &i0, &j0, &di, &dj))
    return NULL;
  if (!(PyArray_Check(line) && PyArray_TYPE(line) == PyArray_FLOAT64 && PyArray_NDIM(line)==1))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be rank-1 double array object");
      return NULL;
    }
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_FLOAT64 && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"second argument must be rank-2 double array object");
      return NULL;
    }
  N = PyArray_DIMS(line)[0];
  M = PyArray_DIMS(image)[1];
  interpolate_bicubic(N,M,PyArray_DATA(line),PyArray_DATA(image),i0,j0,di,dj);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *py_acf(PyObject *self, PyObject *args)
{
  double k, acf = 0, vi, v0, v1, vk, ik;
  PyObject* line = NULL;
  int N;
  int i, j;
  if (!PyArg_ParseTuple(args, "Od", &line, &k))
    return NULL;
  if (!(PyArray_Check(line) && PyArray_TYPE(line) == PyArray_FLOAT64 && PyArray_NDIM(line)==1))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be rank-1 double array object");
      return NULL;
    }
  
  N = PyArray_DIMS(line)[0];
  for (i=0; i<N-k; ++i)
    {
      vi = ((double*)PyArray_DATA(line))[i];
      ik = k + i;
      j = ik;
      v0 = ((double*)PyArray_DATA(line))[j];
      v1 = ((double*)PyArray_DATA(line))[j+1];
      vk = v0 + (v1 - v0) * (ik - j);
      acf += vi * vk;
    }

  return Py_BuildValue("d", acf/N);
}

static PyObject *py_acf2(PyObject *self, PyObject *args)
{
  double p, i0, j0, di, dj, dp1, dp2, r;
  double v1,v2,w1,w2;
  PyObject* image = NULL;
  int M, N;
  int i, floor_p, ceil_p;
  if (!PyArg_ParseTuple(args, "diO(dd)(dd)", &p, &N, &image, &i0, &j0, &di, &dj))
    return NULL;
  if (!(PyArray_Check(image) && PyArray_TYPE(image) == PyArray_FLOAT64 && PyArray_NDIM(image)==2))
    {
      PyErr_SetString(PyExc_TypeError,"3rd argument must be rank-2 double array object");
      return NULL;
    }

  M = PyArray_DIMS(image)[1];
  floor_p = p;
  ceil_p = floor_p+1;
  dp1 = (double)ceil_p - p;
  dp2 = p - (double)floor_p;
  r = 0;
  v1 = w1 = 0;
  for (i=0; i+ceil_p<N; ++i)
    {
      v2 = interpolate_bilinear_at_point(M, PyArray_DATA(image), i0+i*di, j0+i*dj);
      w2 = interpolate_bilinear_at_point(M, PyArray_DATA(image), i0+(i+p)*di, j0+(i+p)*dj);
      if (dp2 != 0.0)
	{
	  if (i)
	    r += dp2*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6;
	  v1 = v2;
	  w1 = w2;
	  v2 = interpolate_bilinear_at_point(M, PyArray_DATA(image), i0+(i+dp1)*di, j0+(i+dp1)*dj);
	  w2 = interpolate_bilinear_at_point(M, PyArray_DATA(image), i0+(i+ceil_p)*di, j0+(i+ceil_p)*dj);
	  r += dp1*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6;
	}
      else
	{
	  if (i)
	    r += dp1*(2*(v1*w1+v2*w2)+v1*w2+v2*w1)/6;
	}
      v1 = v2;
      w1 = w2;
    } 

  return Py_BuildValue("d", r);
}


static PyMethodDef module_methods[] = {
  {"interpolate_bilinear", py_interpolate_bilinear, METH_VARARGS, "interpolate_bilinear(line, image, (i0,j0), (di,dj))"},
  {"interpolate_bilinear_at", py_interpolate_bilinear_at, METH_VARARGS, "interpolate_bilinear_at(line, image, icoords, jcoords)"},
  {"interpolate_bilinear_at_point", py_interpolate_bilinear_at_point, METH_VARARGS, "interpolate_bilinear_at(image, i, j)"},
  {"interpolate_bicubic", py_interpolate_bicubic, METH_VARARGS, "interpolate_bicubic(line, image, (i0,j0), (di,dj))"},
  {"acf", py_acf, METH_VARARGS, "acf(line, k)"},
  {"acf2", py_acf2, METH_VARARGS, "acf2(p, N, image, (i0,j0), (di, dj))"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initlineinterp(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module apply_window_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("lineinterp", module_methods, "Provides interpolate_bilinear function.");
}
