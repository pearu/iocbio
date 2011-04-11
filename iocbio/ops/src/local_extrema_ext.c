/*
  Implements local extrema functions.
  Author: Pearu Peterson
  Created: December 2010
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#define LOOP(index)				\
  for (i##index=0;i##index<dims[index];++i##index)	\
    {
#define LOOP_END }

#define INNER_LOOP(index)				\
  for (j##index=-1;j##index<2;++j##index)	\
    { k##index = i##index + j##index;					\
    switch (boundary)							\
      {									\
 case BC_CONSTANT: k##index = (k##index>=dims[index]?dims[index]-1:(k##index<0?0:k##index)); break; \
 case BC_FINITE: break;							\
 case BC_PERIODIC: k##index = (k##index>=dims[index]?k##index - dims[index]:(k##index<0?k##index + dims[index]:k##index)); break; \
 case BC_REFLECTIVE: k##index = (k##index>=dims[index]?2*dims[index]-k##index-2:(k##index<0?-k##index:k##index)); break; \
 default:								\
 goto fail;								\
      }

#define INNER_LOOP_END }

#define GET_VALUE(ARR, PTR, value)				\
  switch (PyArray_TYPE(ARR))						\
    {									\
    case PyArray_FLOAT64: value = *(npy_float64*)(PTR); break;		\
    case PyArray_FLOAT32: value = *(npy_float32*)(PTR); break;		\
    case PyArray_INT64: value = *(npy_int64*)(PTR); break;		\
    case PyArray_INT32: value = *(npy_int32*)(PTR); break;		\
    case PyArray_INT16: value = *(npy_int16*)(PTR); break;		\
    case PyArray_INT8: value = *(npy_int8*)(PTR); break;		\
    case PyArray_UINT64: value = *(npy_uint64*)(PTR); break;		\
    case PyArray_UINT32: value = *(npy_uint32*)(PTR); break;		\
    case PyArray_UINT16: value = *(npy_uint16*)(PTR); break;		\
    case PyArray_UINT8: value = *(npy_uint8*)(PTR); break;		\
    default:								\
      PyErr_SetString(PyExc_TypeError,"local_extrema_ext:GET_VALUE: unsupported array dtype"); \
      goto fail;							\
    }

#define FIX_INDEX(INDEX, DIM) ((INDEX)<0?(INDEX)+dims[(DIM)]:((INDEX)>=dims[(DIM)]?(INDEX)-dims[(DIM)]:(INDEX)))
#define FIX_INDEX_CONSTANT(INDEX, DIM) ((INDEX)<0?0:((INDEX)>=dims[(DIM)]?dims[(DIM)]-1:(INDEX)))
#define FIX_INDEX_FINITE(INDEX, DIM) ((INDEX)<0?1:((INDEX)>=dims[(DIM)]?dims[(DIM)]-2:(INDEX)))
#define FIX_INDEX_PERIODIC(INDEX, DIM) ((INDEX)<0?(INDEX)+dims[(DIM)]:((INDEX)>=dims[(DIM)]?(INDEX)-dims[(DIM)]:(INDEX)))
#define FIX_INDEX_REFLECTIVE(INDEX, DIM) ((INDEX)<0?-(INDEX):((INDEX)>=dims[(DIM)]?2*dims[(DIM)]-(INDEX)-2:(INDEX)))


#define CALL_WRITE(ARGS) \
  if (verbose && PyObject_CallFunctionObjArgs ARGS == NULL) goto fail;

typedef enum {BC_CONSTANT=0, BC_FINITE=1, BC_PERIODIC=2, BC_REFLECTIVE=3} BoundaryCondition;

static PyObject *local_maxima(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  npy_intp sz = 0, rank=0;
  npy_intp *dims = NULL;
  int i0, i1, i2, i3;
  int j0, j1, j2, j3;
  int k0, k1, k2, k3;
  double v, value, level;
  PyObject *result = NULL;
  int is_extremum;
  PyObject* write_func = NULL;
  int verbose = 0;
  int count;
  BoundaryCondition boundary;
  if (!PyArg_ParseTuple(args, "OdiO", &a, &level, &boundary, &write_func))
    return NULL;
  if (!PyArray_Check(a))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be array object");
      return NULL;
    }
  if (boundary<0 || boundary>3)
    {
      PyErr_SetString(PyExc_ValueError,"third argument must be 0, 1, 2, or 3");
      return NULL;
    }
  if (write_func == Py_None)
    verbose = 0;
  else if (PyCallable_Check(write_func))
    verbose = 1;
  else
    {
      PyErr_SetString(PyExc_TypeError,"fourth argument must be None or callable object");
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
  result = PyList_New(0);
  count = 0;
  switch (rank)
    {
    case 1:
      {
	LOOP(0);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR1(a, i0), value);
	if (value <= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0);
	if (j0)
	  {
	    GET_VALUE(a, PyArray_GETPTR1(a, k0), v);
	    if (v>value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(di)", value, i0));
	LOOP_END;
	break;
      }
    case 2:
      {
	LOOP(0); LOOP(1);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR2(a, i0, i1), value);
	if (value <= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1);
	if (j0 || j1)
	  {
	    GET_VALUE(a, PyArray_GETPTR2(a, k0, k1), v);
	    if (v>value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(dii)", value, i0, i1));
	LOOP_END; LOOP_END;
	break;
      }
    case 3:
      {
	LOOP(0); LOOP(1); LOOP(2);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1 ||
		i2<=0 || i2>=dims[2]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR3(a, i0, i1, i2), value);
	if (value <= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1); INNER_LOOP(2);
	if (j0 || j1 || j2)
	  {
	    GET_VALUE(a, PyArray_GETPTR3(a, k0, k1, k2), v);
	    if (v>value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(diii)", value, i0, i1, i2));
	count ++;
	LOOP_END; 
	if (verbose)
	  {
	    CALL_WRITE((write_func, PyFloat_FromDouble(((double)count)/sz), result, NULL));
	  }
	LOOP_END; LOOP_END;
	break;
      }
    case 4:
      {
	LOOP(0); LOOP(1); LOOP(2); LOOP(3);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1 ||
		i2<=0 || i2>=dims[2]-1 ||
		i3<=0 || i3>=dims[3]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR4(a, i0, i1, i2, i3), value);
	if (value <= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1); INNER_LOOP(2); INNER_LOOP(3);
	if (j0 || j1 || j2 || j3)
	  {
	    GET_VALUE(a, PyArray_GETPTR4(a, k0,k1,k2,k3), v);
	    if (v>value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(diiii)", value, i0, i1, i2, i3));
	LOOP_END; LOOP_END; LOOP_END; LOOP_END;
	break;
      }
    default:
      PyErr_SetString(PyExc_ValueError,"local_extrema: unsupported array rank");
      goto fail;
    }
  
  return result;
fail:
  if (result != NULL)
    Py_DECREF(result);
  return NULL;
}

static PyObject *local_minima(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  npy_intp sz = 0, rank=0;
  npy_intp *dims = NULL;
  int i0, i1, i2, i3;
  int j0, j1, j2, j3;
  int k0, k1, k2, k3;
  double v, value, level;
  PyObject *result = NULL;
  int is_extremum;
  PyObject* write_func = NULL;
  int verbose = 0;
  int count;
  BoundaryCondition boundary;
  if (!PyArg_ParseTuple(args, "OdiO", &a, &level, &boundary, &write_func))
    return NULL;
  if (!PyArray_Check(a))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be array object");
      return NULL;
    }
  if (boundary<0 || boundary>3)
    {
      PyErr_SetString(PyExc_ValueError,"third argument must be 0, 1, 2, or 3");
      return NULL;
    }
  if (write_func == Py_None)
    verbose = 0;
  else if (PyCallable_Check(write_func))
    verbose = 1;
  else
    {
      PyErr_SetString(PyExc_TypeError,"fourth argument must be None or callable object");
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
  result = PyList_New(0);
  count = 0;
  switch (rank)
    {
    case 1:
      {
	LOOP(0);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR1(a, i0), value);
	if (value >= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0);
	if (j0)
	  {
	    GET_VALUE(a, PyArray_GETPTR1(a, k0), v);
	    if (v<value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(di)", value, i0));
	LOOP_END;
	break;
      }
    case 2:
      {
	LOOP(0); LOOP(1);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR2(a, i0, i1), value);
	if (value >= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1);
	if (j0 || j1)
	  {
	    GET_VALUE(a, PyArray_GETPTR2(a, k0, k1), v);
	    if (v<value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(dii)", value, i0, i1));
	LOOP_END; LOOP_END;
	break;
      }
    case 3:
      {
	LOOP(0); LOOP(1); LOOP(2);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1 ||
		i2<=0 || i2>=dims[2]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR3(a, i0, i1, i2), value);
	if (value >= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1); INNER_LOOP(2);
	if (j0 || j1 || j2)
	  {
	    GET_VALUE(a, PyArray_GETPTR3(a, k0, k1, k2), v);
	    if (v<value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(diii)", value, i0, i1, i2));
	count ++;
	LOOP_END; 
	if (verbose)
	  {
	    CALL_WRITE((write_func, PyFloat_FromDouble(((double)count)/sz), result, NULL));
	  }
	LOOP_END; LOOP_END;
	break;
      }
    case 4:
      {
	LOOP(0); LOOP(1); LOOP(2); LOOP(3);
	if (boundary==BC_FINITE)
	  {
	    if (i0<=0 || i0>=dims[0]-1 ||
		i1<=0 || i1>=dims[1]-1 ||
		i2<=0 || i2>=dims[2]-1 ||
		i3<=0 || i3>=dims[3]-1
		)
	      continue;
	  }
	GET_VALUE(a, PyArray_GETPTR4(a, i0, i1, i2, i3), value);
	if (value >= level)
	  continue;
	is_extremum = 1;
	INNER_LOOP(0); INNER_LOOP(1); INNER_LOOP(2); INNER_LOOP(3);
	if (j0 || j1 || j2 || j3)
	  {
	    GET_VALUE(a, PyArray_GETPTR4(a, k0,k1,k2,k3), v);
	    if (v<value) is_extremum = 0;
	  }
	if (!is_extremum) break;
	INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END; INNER_LOOP_END;
	if (is_extremum)
	  PyList_Append(result, Py_BuildValue("(diiii)", value, i0, i1, i2, i3));
	LOOP_END; LOOP_END; LOOP_END; LOOP_END;
	break;
      }
    default:
      PyErr_SetString(PyExc_ValueError,"local_extrema: unsupported array rank");
      goto fail;
    }
  
  return result;
fail:
  if (result != NULL)
    Py_DECREF(result);
  return NULL;
}


static PyMethodDef module_methods[] = {
  {"local_maxima", local_maxima , METH_VARARGS, "local_maxima(a)->value_indices"},
  {"local_minima", local_minima , METH_VARARGS, "local_minima(a)->value_indices"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initlocal_extrema_ext(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module local_extrema_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("local_extrema_ext", module_methods, "Provides functions for finding local extrema.");
}

