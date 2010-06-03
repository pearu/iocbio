/*
  Implements apply_window_inplace function.
  Author: Pearu Peterson
  Created: September 2009
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

static npy_float64 g(npy_float64 x)
{
  npy_float64 r = sin(x*0.5*M_PI);
  return r * r;
}

static npy_float64 f(npy_float64 x, int n)
{
  /*
    f(x) is a function with the following properties:
      f(x) = 0 if x>=1
      0 < f(x) < 1 if 0<x<1
      f(x) = 1 if x<=0
      f(x) is continuously oo-differentiable every where except at x=0 and x=1
      f(x) is continuously (2*n+1 [or more])-differentiable at x=0 and x=1
   */
  int i;
  npy_float64 r=x;
  if (r<=0) return 1.0;
  if (r>=1) return 0.0;
  for (i=0;i<n;++i)
    r = g(r);
  return 0.5*(1+cos(r * M_PI));
}

static PyObject *apply_window_inplace(PyObject *self, PyObject *args)
{
  int n=0;
  PyObject* a = NULL;
  PyObject* scales_obj = NULL;
  npy_intp sz = 0, i, rank=0;
  npy_intp *dims = NULL;
  npy_float64* scales = NULL;
  npy_float64 background = 0.0;
  if (!PyArg_ParseTuple(args, "OOid", &a, &scales_obj, &n, &background))
    return NULL;
  if (!PyArray_Check(a))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be array object");
      return NULL;
    }
  if (!PyTuple_Check(scales_obj))
    {
      PyErr_SetString(PyExc_TypeError,"second argument must be tuple object");
      return NULL;
    }
  sz = PyArray_SIZE(a);
  dims = PyArray_DIMS(a);
  rank = PyArray_NDIM(a);
  if (rank > 4)
    {
      PyErr_SetString(PyExc_NotImplementedError,"only rank <=4 arrays are supported");
      return NULL;
    }
  if (PyTuple_Size(scales_obj) != rank)
    {
      PyErr_SetString(PyExc_TypeError,"second argument must have size equal to the rank of the first argument");
      return NULL;
    }
  scales = (npy_float64*)malloc(sizeof(npy_float64)*rank);
  for (i=0; i<rank; ++i)
    scales[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(scales_obj, i));

#define APPLY_WINDOW_IMUL_OP_T(TYPE) *(TYPE*)ptr = (*(TYPE*)ptr) * r + background*(1.0-r)

#define APPLY_WINDOW_IMUL_OP \
  if (PyArray_TYPE(a) == PyArray_FLOAT32)			\
    APPLY_WINDOW_IMUL_OP_T(npy_float32);			\
  else if (PyArray_TYPE(a) == PyArray_FLOAT64)			\
    APPLY_WINDOW_IMUL_OP_T(npy_float64);			\
  else if (PyArray_TYPE(a) == PyArray_INT8)			\
    APPLY_WINDOW_IMUL_OP_T(npy_int8);				\
  else if (PyArray_TYPE(a) == PyArray_INT16)			\
    APPLY_WINDOW_IMUL_OP_T(npy_int16);				\
  else if (PyArray_TYPE(a) == PyArray_INT32)			\
    APPLY_WINDOW_IMUL_OP_T(npy_int32);				\
  else if (PyArray_TYPE(a) == PyArray_INT64)			\
    APPLY_WINDOW_IMUL_OP_T(npy_int64);				\
  else if (PyArray_TYPE(a) == PyArray_UINT8)			\
    APPLY_WINDOW_IMUL_OP_T(npy_uint8);				\
  else if (PyArray_TYPE(a) == PyArray_UINT16)			\
    APPLY_WINDOW_IMUL_OP_T(npy_uint16);				\
  else if (PyArray_TYPE(a) == PyArray_UINT32)			\
    APPLY_WINDOW_IMUL_OP_T(npy_uint32);				\
  else if (PyArray_TYPE(a) == PyArray_UINT64)			\
    APPLY_WINDOW_IMUL_OP_T(npy_uint64);				\
  else								\
    {								\
      PyErr_SetString(PyExc_TypeError,"unsupported array dtype");	\
      return NULL;							\
    }

#define APPLY_WINDOW_EVAL_F(index) \
  f(scales[index] * (2*i##index<=dims[index]?i##index:dims[index]-i##index-1), n)

  switch (rank)
    {
    case 1:
      {
	npy_intp i0;
	npy_float64 r;
	for (i0=0;i0<dims[0];++i0)
	  {
	    r = 1.0 - APPLY_WINDOW_EVAL_F(0);
	    if (r != 1.0)
	      {
		void* ptr = PyArray_GETPTR1(a, i0);
		APPLY_WINDOW_IMUL_OP;
	      }
	  }
      }
      break;
    case 2:
      {
	npy_intp i0,i1;
	npy_float64 d0,d1,r;
	for (i0=0;i0<dims[0];++i0)
	  {
	    d0 = 1.0 - APPLY_WINDOW_EVAL_F(0);
	    for (i1=0;i1<dims[1];++i1)
	      {
		d1 = 1.0 - APPLY_WINDOW_EVAL_F(1);
		r = d0 * d1;
		if (r != 1.0)
		  {
		    void* ptr = PyArray_GETPTR2(a, i0, i1);
		    APPLY_WINDOW_IMUL_OP;
		  }
	      } 
	  }
      }
      break;
    case 3:
      {
	npy_intp i0,i1,i2;
	npy_float64 d0,d1,d2,r;
	for (i0=0;i0<dims[0];++i0)
	  {
	    d0 = 1.0 - APPLY_WINDOW_EVAL_F(0);
	    for (i1=0;i1<dims[1];++i1)
	      {
		d1 = 1.0 - APPLY_WINDOW_EVAL_F(1);
		for (i2=0;i2<dims[2];++i2)
		  {
		    d2 = 1.0 - APPLY_WINDOW_EVAL_F(2);
		    r = d0 * d1 * d2;
		    if (r != 1.0)
		      {
			void* ptr = PyArray_GETPTR3(a, i0, i1, i2);
			APPLY_WINDOW_IMUL_OP;
		      }
		  } 
	      }
	  }
      }
      break;
    case 4:
      {
	npy_intp i0,i1,i2,i3;
	npy_float64 d0,d1,d2,d3,r;
	for (i0=0;i0<dims[0];++i0)
	  {
	    d0 = 1.0 - APPLY_WINDOW_EVAL_F(0);
	    for (i1=0;i1<dims[1];++i1)
	      {
		d1 = 1.0 - APPLY_WINDOW_EVAL_F(1);
		for (i2=0;i2<dims[2];++i2)
		  {
		    d2 = 1.0 - APPLY_WINDOW_EVAL_F(2);
		    for (i3=0;i3<dims[3];++i3)
		      {
			d3 = 1.0 - APPLY_WINDOW_EVAL_F(3);
			r = d0 * d1 * d2 * d3;
			if (r != 1.0)
			  {
			    void* ptr = PyArray_GETPTR4(a, i0, i1, i2, i3);
			    APPLY_WINDOW_IMUL_OP;
			  }
		      } 
		  }
	      }
	  }
	break;
      }
    default:
      return NULL;
    }
  free(scales);
  return Py_BuildValue("");
}

static PyMethodDef module_methods[] = {
  {"apply_window_inplace", apply_window_inplace, METH_VARARGS, "apply_window_inplace(a,scales,smoothness,background)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initapply_window_ext(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module apply_window_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("apply_window_ext", module_methods, "Provides apply_window_inplace function.");
}
