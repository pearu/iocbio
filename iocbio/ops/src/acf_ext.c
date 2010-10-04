/*
  Implements wrapper functions to acf.c code.
  Author: Pearu Peterson
  Created: September 2010
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

#include "acf.h"

static PyObject *py_acf_evaluate(PyObject *self, PyObject *args)
{
  int i, n;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  PyObject* y_py = NULL;
  PyObject* y1_py = NULL;
  PyObject* r_py = NULL;
  ACFInterpolationMethod mth = ACFUnspecified;
  double y, r;
  double *f = NULL;
  if (!PyArg_ParseTuple(args, "OOi", &f1_py, &y1_py, &mth))
    return NULL;
  switch (mth)
    {
    case ACFInterpolationConstant: ;
    case ACFInterpolationLinear: ;
    case ACFInterpolationCatmullRom: 
    case ACFInterpolationConstantWithSizeReduction: ;
    case ACFInterpolationLinearWithSizeReduction: ;
    case ACFInterpolationCatmullRomWithSizeReduction: ;
      break;
    default:
      PyErr_SetString(PyExc_TypeError,"third argument must be 0, 1, or 2");
      return NULL;
    }
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 1);
  if (f_py==NULL)
    return NULL;

  n = PyArray_SIZE(f_py);
  f = (double*)PyArray_DATA(f_py);

  if (PyFloat_Check(y1_py))
    {
      y = PyFloat_AsDouble(y1_py);
      r = acf_evaluate(f, n, y, mth);
      if (f1_py != f_py)
	{
	  Py_DECREF(f_py);
	}
      return Py_BuildValue("d",r);
    }

  y_py = PyArray_ContiguousFromAny(y1_py, PyArray_DOUBLE, 1, 1);
  if (y_py==NULL)
    return NULL;
  r_py =  PyArray_SimpleNew(PyArray_NDIM(y_py), 
			    PyArray_DIMS(y_py),
			    PyArray_DOUBLE);
  for (i=0; i<PyArray_SIZE(y_py); ++i)
    {
      y = *((double*)PyArray_GETPTR1(y_py, i));
      r = acf_evaluate(f, n, y, mth);
      *(double*)PyArray_GETPTR1(r_py, i) = r;
    }
  if (y1_py != y_py)
    {
      Py_DECREF(y_py);
    }
  if (f1_py != f_py)
    {
      Py_DECREF(f_py);
    }
  return r_py;
}

static PyObject *py_acf_maximum_point(PyObject *self, PyObject *args)
{
  int n;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  ACFInterpolationMethod mth = ACFUnspecified;
  double r;
  double *f = NULL;
  int start_j = 0;
  if (!PyArg_ParseTuple(args, "Oii", &f1_py, &start_j, &mth))
    return NULL;
  switch (mth)
    {
    case ACFInterpolationConstant: ;
    case ACFInterpolationLinear: ;
    case ACFInterpolationCatmullRom: 
    case ACFInterpolationConstantWithSizeReduction: ;
    case ACFInterpolationLinearWithSizeReduction: ;
    case ACFInterpolationCatmullRomWithSizeReduction: ;
      break;
    default:
      PyErr_SetString(PyExc_TypeError,"third argument must be 0, 1, or 2");
      return NULL;
    }
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 1);
  if (f_py==NULL)
    return NULL;
  n = PyArray_SIZE(f_py);
  f = (double*)PyArray_DATA(f_py);
  r = acf_maximum_point(f, n, start_j, mth);
  if (f1_py != f_py)
    {
      Py_DECREF(f_py);
    }
  return Py_BuildValue("d",r);
}

static PyObject *py_acf_sine_fit(PyObject *self, PyObject *args)
{
  int n;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  ACFInterpolationMethod mth = ACFUnspecified;
  double r;
  double *f = NULL;
  int start_j = 0;
  if (!PyArg_ParseTuple(args, "Oii", &f1_py, &start_j, &mth))
    return NULL;
  switch (mth)
    {
    case ACFInterpolationConstant: ;
    case ACFInterpolationLinear: ;
    case ACFInterpolationCatmullRom: ;
    case ACFInterpolationConstantWithSizeReduction: ;
    case ACFInterpolationLinearWithSizeReduction: ;
    case ACFInterpolationCatmullRomWithSizeReduction: ;
      break;
    default:
      PyErr_SetString(PyExc_TypeError,"third argument must be 0, 1, or 2");
      return NULL;
    }
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 1);
  if (f_py==NULL)
    return NULL;
  n = PyArray_SIZE(f_py);
  f = (double*)PyArray_DATA(f_py);
  r = acf_sine_fit(f, n, start_j, mth);
  if (f1_py != f_py)
    {
      Py_DECREF(f_py);
    }
  return Py_BuildValue("d",r);
}

static PyObject *py_acf_sine_power_spectrum(PyObject *self, PyObject *args)
{
  int i, n;
  PyObject* f_py = NULL;
  PyObject* f1_py = NULL;
  PyObject* y_py = NULL;
  PyObject* y1_py = NULL;
  PyObject* r_py = NULL;
  ACFInterpolationMethod mth = ACFUnspecified;
  double y, r;
  double *f = NULL;
  if (!PyArg_ParseTuple(args, "OOi", &f1_py, &y1_py, &mth))
    return NULL;
  switch (mth)
    {
    case ACFInterpolationConstant: ;
    case ACFInterpolationLinear: ;
    case ACFInterpolationCatmullRom: ;
    case ACFInterpolationConstantWithSizeReduction: ;
    case ACFInterpolationLinearWithSizeReduction: ;
    case ACFInterpolationCatmullRomWithSizeReduction: ;
      break;
    default:
      PyErr_SetString(PyExc_TypeError,"third argument must be 0, 1, or 2");
      return NULL;
    }
  f_py = PyArray_ContiguousFromAny(f1_py, PyArray_DOUBLE, 1, 1);
  if (f_py==NULL)
    return NULL;

  n = PyArray_SIZE(f_py);
  f = (double*)PyArray_DATA(f_py);

  if (PyFloat_Check(y1_py))
    {
      y = PyFloat_AsDouble(y1_py);
      r = acf_sine_power_spectrum(f, n, y, mth);
      if (f1_py != f_py)
	{
	  Py_DECREF(f_py);
	}
      return Py_BuildValue("d",r);
    }

  y_py = PyArray_ContiguousFromAny(y1_py, PyArray_DOUBLE, 1, 1);
  if (y_py==NULL)
    return NULL;
  r_py =  PyArray_SimpleNew(PyArray_NDIM(y_py), 
			    PyArray_DIMS(y_py),
			    PyArray_DOUBLE);
  for (i=0; i<PyArray_SIZE(y_py); ++i)
    {
      y = *((double*)PyArray_GETPTR1(y_py, i));
      r = acf_sine_power_spectrum(f, n, y, mth);
      *(double*)PyArray_GETPTR1(r_py, i) = r;
    }
  if (y1_py != y_py)
    {
      Py_DECREF(y_py);
    }
  if (f1_py != f_py)
    {
      Py_DECREF(f_py);
    }
  return r_py;
}


static PyMethodDef module_methods[] = {
  {"acf", py_acf_evaluate, METH_VARARGS, "acf(f, y, mth)"},
  {"acf_argmax", py_acf_maximum_point, METH_VARARGS, "acf_argmax(f, start_j, mth)"},
  {"acf_sinefit", py_acf_sine_fit, METH_VARARGS, "acf_sinefit(f, start_j, mth)"},
  {"acf_sine_power_spectrum", py_acf_sine_power_spectrum, METH_VARARGS, "acf_sine_power_spectrum(f, omega, mth)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initacf_ext(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module acf_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("acf_ext", module_methods, "Provides wrappers to acf.c functions.");
}
