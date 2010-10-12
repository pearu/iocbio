/*
  Implements wrapper functions to discrete_gauss.c code.
  Author: Pearu Peterson
  Created: October 2010
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

#include "discrete_gauss.h"

static PyObject *py_dg_convolve(PyObject *self, PyObject *args)
{
  int n, rows;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  double t;
  double *f = NULL;
  if (!PyArg_ParseTuple(args, "Od", &f1_py, &t))
    return NULL;
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 2);
  if (f_py==NULL)
    return NULL;
  if (PyArray_NDIM(f_py)==2)
    {
      n = PyArray_DIMS(f_py)[0];
      rows = PyArray_DIMS(f_py)[1];
      if (rows<=0)
	{
	  PyErr_SetString(PyExc_TypeError,"first argument is empty");
	  return NULL;	  
	}
    }
  else
    {
      n = PyArray_DIMS(f_py)[0];
      rows = 1;
    }
  assert(rows==1);
  f = (double*)PyArray_DATA(f_py);
  dg_convolve(f, n, t);
  if (f1_py == f_py)
    {
      Py_INCREF(f_py);
    }
  return f_py;
}


static PyObject *py_dg_high_pass_filter(PyObject *self, PyObject *args)
{
  int n, rows;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  double t;
  double *f = NULL;
  if (!PyArg_ParseTuple(args, "Od", &f1_py, &t))
    return NULL;
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 2);
  if (f_py==NULL)
    return NULL;
  if (PyArray_NDIM(f_py)==2)
    {
      n = PyArray_DIMS(f_py)[1];
      rows = PyArray_DIMS(f_py)[0];
      if (rows<=0)
	{
	  PyErr_SetString(PyExc_TypeError,"first argument is empty");
	  return NULL;	  
	}
    }
  else
    {
      n = PyArray_DIMS(f_py)[0];
      rows = 1;
    }
  printf("n,rows=%d,%d\n", n, rows);
  f = (double*)PyArray_DATA(f_py);
  dg_high_pass_filter(f, n, rows, t);
  if (f1_py == f_py)
    {
      Py_INCREF(f_py);
    }
  return f_py;
}


static PyMethodDef module_methods[] = {
  {"convolve", py_dg_convolve, METH_VARARGS, "convolve(f, t)"},
  {"high_pass_filter", py_dg_high_pass_filter, METH_VARARGS, "high_pass_filter(f, t)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initdiscrete_gauss_ext(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module discrete_gauss_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("discrete_gauss_ext", module_methods, "Provides wrappers to discrete_gauss.c functions.");
}
