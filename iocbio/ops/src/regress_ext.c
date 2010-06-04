/*
  Implements regress function.
  Author: Pearu Peterson
  Created: September 2009
 */

/*

  TODO:
  - support different kernel types/smoothing methods/boundary conditions for different dimensions
  - support for rank>=4 arrays
  - implement polynomial regression
  - optionally return gradients of regression fits
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#define REGRESS_LOOP(index)				\
  for (i##index=0;i##index<dims[index];++i##index)	\
    {
#define REGRESS_LOOP_END }

#define REGRESS_INIT_KERNEL_LOOP		\
  for (i=0;i<=rank;i++)				\
    {						\
      for (j=0;j<=rank;j++) mat[i][j] = 0;	\
      rhs[i] = 0;				\
      if (i<rank) kkdims[i] = 0;		\
    }

#define KERNEL_KERNEL_LOOP(index) \
  for (ki##index=-di[index], i##index=0; ki##index<=di[index];++ki##index,++i##index) \
    {									\
      d##index = scales[index] * (ki##index);
#define KERNEL_KERNEL_LOOP_END }

#define REGRESS_KERNEL_LOOP(index) \
  for (ki##index=i##index-di[index]; ki##index<=i##index+di[index];++ki##index)	\
    {									\
      d##index = scales[index] * (i##index - ki##index);		\
      switch (boundary_condition)					\
	{								\
	case BC_CONSTANT: j##index = (ki##index>=dims[index] ? dims[index]-1: (ki##index<0?0:ki##index)); break; \
	case BC_FINITE:	j##index = ki##index; if ((ki##index>=dims[index])||(ki##index<0)) continue; break; \
	case BC_PERIODIC: j##index = (ki##index>=dims[index] ? ki##index-dims[index]: (ki##index<0?ki##index+dims[index]:ki##index)); break; \
	case BC_REFLECTIVE: j##index = (ki##index>=dims[index]?2*dims[index]-ki##index-2: (ki##index<0?-ki##index:ki##index)); break; \
	default:							\
	  PyErr_SetString(PyExc_ValueError,"regress:kernel_loop: unknown boundary condition"); \
	  goto fail;							\
	}

#define REGRESS_KERNEL_LOOP_END }

#define REGRESS_SET_VALUE(ARR, PTR, value)				\
  switch (PyArray_TYPE(ARR))						\
    {									\
    case PyArray_FLOAT64: *(npy_float64*)(PTR) = value; break;		\
    case PyArray_FLOAT32: *(npy_float32*)(PTR) = value; break;		\
    case PyArray_INT64: *(npy_int64*)(PTR) = value; break;		\
    case PyArray_INT32: *(npy_int32*)(PTR) = value; break;		\
    case PyArray_INT16: *(npy_int16*)(PTR) = value; break;		\
    case PyArray_INT8: *(npy_int8*)(PTR) = value; break;		\
    case PyArray_UINT64: *(npy_uint64*)(PTR) = value; break;		\
    case PyArray_UINT32: *(npy_uint32*)(PTR) = value; break;		\
    case PyArray_UINT16: *(npy_uint16*)(PTR) = value; break;		\
    case PyArray_UINT8: *(npy_uint8*)(PTR) = value; break;		\
    default:								\
      PyErr_SetString(PyExc_TypeError,"regress|kernel:set_value: unsupported array dtype"); \
      goto fail;							\
    }

#define REGRESS_GET_VALUE(ARR, PTR, value)				\
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
      PyErr_SetString(PyExc_TypeError,"regress:get_value: unsupported array dtype"); \
      goto fail;							\
    }


/* constant in gaussian kernel exponent is choosed such that the
   boundary value is 100x smaller than the center value */
#define REGRESS_EVAL_KERNEL						\
  switch (kernel_type)							\
    {									\
    case KT_EPANECHNIKOV: kv = 0.75*(1-r2); break;			\
    case KT_UNIFORM: kv = 0.5; break;			\
    case KT_TRIANGULAR: kv = 1.0 - sqrt(r2); break;			\
    case KT_QUARTIC: kv=1-r2; kv = 15.0/16.0*kv*kv; break;		\
    case KT_TRIWEIGHT: kv=1-r2; kv = 35.0/32.0*kv*kv*kv; break;		\
    case KT_TRICUBE: kv = 1-r2*sqrt(r2); kv=kv*kv*kv; break;		\
    case KT_GAUSSIAN: kv = exp((-4.6051701859880909) * r2); break;	\
    default:								\
      PyErr_SetString(PyExc_ValueError,"regress|kernel:eval_kernel: unknown kernel type"); \
      goto fail;							\
    }									\
  if (r2==1.0) kv *= 0.5;

#define REGRESS_UPDATE_KKDIMS(index)					\
  if (kkdims[index]==-1) kkdims[index] = j##index;			\
  else if ((kkdims[index]>=0) && (kkdims[index] != j##index)) kkdims[index] = -2;

#define REGRESS_APPLY_KKDIMS			\
  for (i=0; i<rank; ++i)			\
    if (kkdims[i]!=-2)				\
      {						\
	rhs[i+1] = 0;				\
	for (j=0; j<=rank; ++j)			\
	  mat[i+1][j] = mat[j][i+1] = 0;	\
	mat[i+1][i+1] = 1;			\
      }

#define CALL_WRITE(ARGS) \
  if (verbose && PyObject_CallFunctionObjArgs ARGS == NULL) return NULL;


double compute_dot2(double a[2][2], double b[2], int i0);
double compute_dot3(double a[3][3], double b[3], int i0, int i1);
double compute_dot4(double a[4][4], double b[4], int i0, int i1, int i2);

typedef enum {KT_EPANECHNIKOV=0, KT_UNIFORM=1, KT_TRIANGULAR=2, KT_QUARTIC=3, KT_TRIWEIGHT=4, KT_TRICUBE=5,
              KT_GAUSSIAN=6 } KernelType;
typedef enum {SM_AVERAGE=0, SM_LINEAR=1} SmoothingMethod;
typedef enum {BC_CONSTANT=0, BC_FINITE=1, BC_PERIODIC=2, BC_REFLECTIVE=3} BoundaryCondition;

double calc_kernel_sum(KernelType kernel_type, npy_intp rank, double* scales, npy_intp* di)
{
  double r = 0;
  int ki0, ki1, ki2;
  int i0, i1, i2;
  double d0, d1, d2;
  double kv, r2;
  switch (rank)
    {
    case 1:
      KERNEL_KERNEL_LOOP(0);
      r2 = d0*d0;
      REGRESS_EVAL_KERNEL; // sets kv
      r += kv;
      KERNEL_KERNEL_LOOP_END;
      break;
    case 2:
      KERNEL_KERNEL_LOOP(0);
      KERNEL_KERNEL_LOOP(1);
      r2 = d0*d0 + d1*d1;
      REGRESS_EVAL_KERNEL; // sets kv
      r += kv;
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      break;
    case 3:
      KERNEL_KERNEL_LOOP(0);
      KERNEL_KERNEL_LOOP(1);
      KERNEL_KERNEL_LOOP(2);
      r2 = d0*d0 + d1*d1 + d2*d2;
      REGRESS_EVAL_KERNEL; // sets kv
      r += kv;
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      break;
    default:
      PyErr_SetString(PyExc_ValueError,"kernel: unsupported array rank");
      goto fail;
    }
  return r;
 fail:
  return 0;
}

static PyObject *kernel(PyObject *self, PyObject *args)
{
  npy_intp i,rank;
  PyObject* scales_obj = NULL;
  npy_float64* scales = NULL;
  npy_intp* kdims = NULL;
  npy_intp* di = NULL;
  KernelType kernel_type;
  PyObject* r=NULL;
  double kv, r2;
  int ki0, ki1, ki2;
  int i0, i1, i2;
  double d0, d1, d2;
  double kernel_sum;
  if (!PyArg_ParseTuple(args, "Oi", &scales_obj, &kernel_type))
    return NULL;
  if (!PyTuple_Check(scales_obj))
    {
      PyErr_SetString(PyExc_TypeError,"first argument must be tuple object");
      return NULL;
    }
  rank = PyTuple_Size(scales_obj);
  if (rank > 3)
    {
      PyErr_SetString(PyExc_NotImplementedError,"only rank <=3 arrays are supported");
      return NULL;
    }
  scales = (npy_float64*)malloc(sizeof(npy_float64)*rank);
  kdims = (npy_intp*)malloc(sizeof(npy_intp)*rank);
  di = (npy_intp*)malloc(sizeof(npy_intp)*rank);

  for (i=0; i<rank; ++i)
    {
      scales[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(scales_obj, i));
      di[i] = (npy_intp)(ceil(1 / scales[i]));
      kdims[i] = 2*di[i]+1;
    }

  r = PyArray_SimpleNew(rank, kdims, PyArray_FLOAT64);

  kernel_sum = calc_kernel_sum(kernel_type, rank, scales, di);
  if (kernel_sum==0.0)
    goto fail;

  switch (rank)
    {
    case 1:
      KERNEL_KERNEL_LOOP(0);
      r2 = d0*d0;
      REGRESS_EVAL_KERNEL; // sets kv
      REGRESS_SET_VALUE(r, PyArray_GETPTR1(r, i0), kv/kernel_sum);
      KERNEL_KERNEL_LOOP_END;
      break;
    case 2:
      KERNEL_KERNEL_LOOP(0);
      KERNEL_KERNEL_LOOP(1);
      r2 = d0*d0 + d1*d1;
      REGRESS_EVAL_KERNEL; // sets kv
      REGRESS_SET_VALUE(r, PyArray_GETPTR2(r, i0, i1), kv/kernel_sum);
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      break;
    case 3:
      KERNEL_KERNEL_LOOP(0);
      KERNEL_KERNEL_LOOP(1);
      KERNEL_KERNEL_LOOP(2);
      r2 = d0*d0 + d1*d1 + d2*d2;
      REGRESS_EVAL_KERNEL; // sets kv
      REGRESS_SET_VALUE(r, PyArray_GETPTR3(r, i0, i1, i2), kv/kernel_sum);
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      KERNEL_KERNEL_LOOP_END;
      break;
    default:
      PyErr_SetString(PyExc_ValueError,"kernel: unsupported array rank");
      goto fail;
    }

  free(scales);
  free(kdims);
  free(di);
  return Py_BuildValue("N", r);  
 fail:
  Py_DECREF(r);
  free(scales);
  free(kdims);
  free(di);
  return NULL;
}

static PyObject *regress(PyObject *self, PyObject *args)
{
  PyObject* a = NULL;
  npy_intp sz = 0, i, j, rank=0;
  npy_intp *dims = NULL;
  PyObject* scales_obj = NULL;
  npy_float64* scales = NULL;
  PyObject* r=NULL;
  int *kdims = NULL;
  npy_intp* di = NULL;
  double eta;
  int count;
  int kn;
  int i0, i1, i2;
  int j0, j1, j2;
  int ki0, ki1, ki2;
  double d0, d1, d2;
  double r2, kv, value;
  double kernel_sum;
  int verbose;
  PyObject* write_func = NULL;
  clock_t start_clock = clock();

  KernelType kernel_type;
  SmoothingMethod smoothing_method;
  BoundaryCondition boundary_condition;

  if (!PyArg_ParseTuple(args, "OOiiiO", &a, &scales_obj, &kernel_type, &smoothing_method, &boundary_condition, &write_func))
    return NULL;
  //printf("kernel_type=%d\n", kernel_type);
  //printf("smoothing_method=%d\n", smoothing_method);
  //printf("boundary_condition=%d\n", boundary_condition);
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
  if (write_func == Py_None)
    verbose = 0;
  else if (PyCallable_Check(write_func))
    verbose = 1;
  else
    {
      PyErr_SetString(PyExc_TypeError,"sixth argument must be None or callable object");
      return NULL;
    }
  sz = PyArray_SIZE(a);
  dims = PyArray_DIMS(a);
  rank = PyArray_NDIM(a);
  if (rank > 3)
    {
      PyErr_SetString(PyExc_NotImplementedError,"only rank <=3 arrays are supported");
      return NULL;
    }
  if (PyTuple_Size(scales_obj) != rank)
    {
      PyErr_SetString(PyExc_TypeError,"second argument must have size equal to the rank of the first argument");
      return NULL;
    }
  scales = (npy_float64*)malloc(sizeof(npy_float64)*rank);
  kdims = (int*)malloc(sizeof(int)*rank);
  di = (npy_intp*)malloc(sizeof(npy_intp)*rank);

  for (i=0; i<rank; ++i)
    {
      scales[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(scales_obj, i));
      di[i] = (npy_intp)(ceil(1 / scales[i]));
      kdims[i] = 2*di[i]+1;
      //printf("kdims[%d]=%d\n",(int)i,(int)kdims[i]);
    }

  r = PyArray_SimpleNew(rank, PyArray_DIMS(a), PyArray_TYPE(a));

  kernel_sum = calc_kernel_sum(kernel_type, rank, scales, di);
  if (kernel_sum==0.0)
    goto fail;

  kn = 0;
  count = 0;
  switch (rank)
    {
    case 1:
      {
	double mat[2][2] = {{0,0},{0,0}};
	double rhs[2] = {0,0};
	int kkdims[1] = {-1};
	REGRESS_LOOP(0);
	REGRESS_INIT_KERNEL_LOOP;
	REGRESS_KERNEL_LOOP(0);
	r2 = d0*d0;
	if (r2 <= 1.0)
	  {
	    REGRESS_GET_VALUE(a, PyArray_GETPTR1(a, j0), value);
	    REGRESS_EVAL_KERNEL;
	    kv /= kernel_sum;
	    kn++;			
	    REGRESS_UPDATE_KKDIMS(0);
	    switch (smoothing_method)
	      {
	      case SM_AVERAGE:
		rhs[0] += kv * value;
		mat[0][0] += kv;
		break;
	      case SM_LINEAR:
		rhs[0] += kv * value; rhs[1] += kv * value * ki0;
		mat[0][0] += kv; mat[0][1] += kv * ki0;
		mat[1][1] += kv * ki0 * ki0;
		break;
	      default:
		PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
		goto fail;
	      }
	    //printf("i0=%d, ki0=%d, j0=%d, kv=%f, value=%f\n",i0,ki0,j0,kv,value);
	  }
	REGRESS_KERNEL_LOOP_END;
	/* interpolating */
	switch (smoothing_method)
	  {
	  case SM_AVERAGE:
	    value = rhs[0] / mat[0][0];
	    break;
	  case SM_LINEAR:
	    REGRESS_APPLY_KKDIMS;
	    value = compute_dot2(mat, rhs, i0);
	    break;
	  default: 
	    PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
	    goto fail;
	  }
	REGRESS_SET_VALUE(r, PyArray_GETPTR1(r, i0), value);
	//printf("r[%d] = %f\n",i0, value);
	count ++;
	REGRESS_LOOP_END;
      }
      break;

    case 2:
      {	
	double mat[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
	double rhs[3] = {0,0,0};
	int kkdims[2] = {-1, -1};
	if (verbose)
	  {
	    if (PyObject_CallFunctionObjArgs(write_func,
					     PyString_FromString("\n"),
					     NULL)==NULL)
	      goto fail;
	  }
	REGRESS_LOOP(0);
	REGRESS_LOOP(1);
	/* computing neighborhood indices and kernel values
	 */
	REGRESS_INIT_KERNEL_LOOP;
	kn = 0;
	REGRESS_KERNEL_LOOP(0);
	REGRESS_KERNEL_LOOP(1);
	r2 = d0*d0 + d1*d1;
	if (r2 <= 1.0)
	  {
	    REGRESS_EVAL_KERNEL;
	    kv /= kernel_sum;
	    kn++;			
	    REGRESS_GET_VALUE(a, PyArray_GETPTR2(a, j0, j1), value);
	    REGRESS_UPDATE_KKDIMS(0);
	    REGRESS_UPDATE_KKDIMS(1);
	    switch (smoothing_method)
	      {
	      case SM_AVERAGE:
		rhs[0] += kv * value;
		mat[0][0] += kv;
		break;
	      case SM_LINEAR:
		rhs[0] += kv * value; rhs[1] += kv * value * ki0; rhs[2] += kv * value * ki1;
		mat[0][0] += kv; mat[0][1] += kv * ki0; mat[0][2] += kv * ki1;
		mat[1][1] += kv * ki0 * ki0; mat[1][2] += kv * ki0 * ki1;
		mat[2][2] += kv * ki1 * ki1;
		break;
	      default:
		PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
		goto fail;
	      }
	  }
	REGRESS_KERNEL_LOOP_END;
	REGRESS_KERNEL_LOOP_END;

	/* interpolating */
	switch (smoothing_method)
	  {
	  case SM_AVERAGE:
	    value = rhs[0] / mat[0][0];
	    break;
	  case SM_LINEAR:
	    REGRESS_APPLY_KKDIMS;
	    value = compute_dot3(mat, rhs, i0, i1);
	    break;
	  default: 
	    PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
	    goto fail;
	  }
	REGRESS_SET_VALUE(r, PyArray_GETPTR2(r, i0, i1), value);
	count ++;
      }
      if (verbose)
	{
	  eta = (clock() - start_clock) * (sz/((double)count)-1.0) / CLOCKS_PER_SEC;
	  CALL_WRITE((write_func,
		      PyString_FromString("\rComputing regression: %6.2f%% done, #kernel points:%d, ETA:%4.1fs  "),
		      PyFloat_FromDouble((count*100.0)/sz),
		      PyInt_FromLong(kn),
		      PyFloat_FromDouble(eta),
		      NULL));
	}
      REGRESS_LOOP_END;
      REGRESS_LOOP_END;
      CALL_WRITE((write_func, PyString_FromString("\n"), NULL));
      break;
    case 3:
      {
	double mat[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
	double rhs[4] = {0,0,0,0};
	int kkdims[3] = {-1, -1, -1};

	if (verbose)
	  {
	    if (PyObject_CallFunctionObjArgs(write_func,
					     PyString_FromString("\n"),
					     NULL)==NULL)
	      goto fail;
	  }
	REGRESS_LOOP(0);
	REGRESS_LOOP(1);
	REGRESS_LOOP(2);
	/* computing neighborhood indices and kernel values
	 */
	REGRESS_INIT_KERNEL_LOOP;
	kn = 0;
	REGRESS_KERNEL_LOOP(0);
	REGRESS_KERNEL_LOOP(1);
	REGRESS_KERNEL_LOOP(2);
	r2 = d0*d0 + d1*d1 + d2*d2;
	if (r2 <= 1.0)
	  {
	    REGRESS_EVAL_KERNEL;
	    kv /= kernel_sum;
	    kn++;			
	    REGRESS_GET_VALUE(a, PyArray_GETPTR3(a, j0, j1, j2), value);
	    REGRESS_UPDATE_KKDIMS(0);
	    REGRESS_UPDATE_KKDIMS(1);
	    REGRESS_UPDATE_KKDIMS(2);			
	    switch (smoothing_method)
	      {
	      case SM_AVERAGE:
		rhs[0] += kv * value;
		mat[0][0] += kv;
		break;
	      case SM_LINEAR:
		rhs[0] += kv * value; rhs[1] += kv * value * ki0; rhs[2] += kv * value * ki1; rhs[3] += kv * value * ki2;
		mat[0][0] += kv; mat[0][1] += kv * ki0; mat[0][2] += kv * ki1; mat[0][3] += kv * ki2;
		mat[1][1] += kv * ki0 * ki0; mat[1][2] += kv * ki0 * ki1; mat[1][3] += kv * ki0 * ki2;
		mat[2][2] += kv * ki1 * ki1; mat[2][3] += kv * ki1 * ki2; mat[3][3] += kv * ki2 * ki2;
		break;
	      default:
		PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
		goto fail;
	      }
	  }
	REGRESS_KERNEL_LOOP_END;
	REGRESS_KERNEL_LOOP_END;
	REGRESS_KERNEL_LOOP_END;

	/* interpolating */
	if (kn>1)
	  switch (smoothing_method)
	    {
	    case SM_AVERAGE:
	      value = rhs[0] / mat[0][0];
	      break;
	    case SM_LINEAR:
	      REGRESS_APPLY_KKDIMS;
	      value = compute_dot4(mat, rhs, i0, i1, i2);
	      break;
	    default: 
	      PyErr_SetString(PyExc_ValueError,"regress: unknown smoothing method");
	      goto fail;
	    }
	REGRESS_SET_VALUE(r, PyArray_GETPTR3(r, i0, i1, i2), value);
	count ++;
      }
      REGRESS_LOOP_END;
      if (verbose)
	{
	  eta = (clock() - start_clock) * (sz/((double)count)-1.0) / CLOCKS_PER_SEC;
	  CALL_WRITE((write_func,
		      PyString_FromString("\rComputing regression: %6.2f%% done, #kernel points:%d, ETA:%4.1fs  "),
		      PyFloat_FromDouble((count*100.0)/sz),
		      PyInt_FromLong(kn),
		      PyFloat_FromDouble(eta),
		      NULL));
	}
      REGRESS_LOOP_END;
      REGRESS_LOOP_END;
      CALL_WRITE((write_func, PyString_FromString("\n"), NULL));
      break;
    default:
      PyErr_SetString(PyExc_ValueError,"regress: unsupported array rank");
      goto fail;
    }
  free(scales);
  free(kdims);
  free(di);
  return Py_BuildValue("N", r);
 fail:
  free(scales);
  free(kdims);
  free(di);
  return NULL;

}

static PyMethodDef module_methods[] = {
  {"kernel", kernel, METH_VARARGS, "kernel(scales, kernel_type_code)"},
  {"regress", regress, METH_VARARGS, "regress(a, scales, kernel_type_code, smoother_method_code, boundary_condition_code, verbose)"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initregress_ext(void) 
{
  PyObject* m = NULL;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module regress_ext (failed to import numpy)"); return;}
  m = Py_InitModule3("regress_ext", module_methods, "Provides regress and kernel functions.");
}

void compute_inverse2(double imat[2][2], double a[2][2])
{
  double t2 = a[0][1]*a[0][1];
  double t4 = 1/(-a[0][0]*a[1][1]+t2);
  double t6 = a[0][1]*t4;
  imat[0][0] = -a[1][1]*t4;
  imat[0][1] = t6;
  imat[1][0] = t6;
  imat[1][1] = -a[0][0]*t4;

}

void compute_inverse3(double imat[3][3], double a[3][3])
{
  double t2 = a[1][2]*a[1][2];
  double t4 = a[0][0]*a[1][1];
  double t7 = a[0][1]*a[0][1];
  double t9 = a[0][2]*a[0][1];
  double t12 = a[0][2]*a[0][2];
  double t15 = 1/(-t4*a[2][2]+a[0][0]*t2+t7*a[2][2]-2.0*t9*a[1][2]+t12*a[1][1]);
  
  double t20 = (-a[0][1]*a[2][2]+a[0][2]*a[1][2])*t15;
  double t24 = (-a[0][1]*a[1][2]+a[0][2]*a[1][1])*t15;
  double t30 = (-a[0][0]*a[1][2]+t9)*t15;
  imat[0][0] = (-a[1][1]*a[2][2]+t2)*t15;
  imat[0][1] = -t20;
  imat[0][2] = t24;
  imat[1][0] = -t20;
  imat[1][1] = (-a[0][0]*a[2][2]+t12)*t15;
  imat[1][2] = -t30;
  imat[2][0] = t24;
  imat[2][1] = -t30;
  imat[2][2] = (-t4+t7)*t15;
}

void compute_inverse4(double imat[4][4], double a[4][4])
{
  double t3 = a[2][3]*a[2][3];
  double t5 = a[1][2]*a[1][2];
  double t10 = a[1][3]*a[1][3];
  double t13 = a[0][0]*a[1][1];
  double t17 = a[0][0]*t5;
  double t19 = a[0][0]*a[1][2];
  double t23 = a[0][0]*t10;
  double t25 = a[0][1]*a[0][1];
  double t26 = t25*a[2][2];
  double t29 = a[0][1]*a[1][2];
  double t33 = a[0][3]*a[2][3];
  double t36 = a[0][1]*a[1][3];
  double t43 = a[0][2]*a[0][2];
  double t44 = a[1][1]*t43;
  double t46 = a[0][2]*a[1][1];
  double t50 = a[0][2]*a[1][3];
  double t51 = a[0][3]*a[1][2];
  double t54 = a[0][3]*a[0][3];
  double t55 = a[1][1]*t54;
  double t58 = t13*a[2][2]*a[3][3]-t13*t3-t17*a[3][3]+2.0*t19*a[1][3]*a[2][3]-t23*
    a[2][2]-t26*a[3][3]+t25*t3+2.0*t29*a[0][2]*a[3][3]-2.0*t29*t33-2.0*t36*a[0][2]*
    a[2][3]+2.0*t36*a[0][3]*a[2][2]-t44*a[3][3]+2.0*t46*t33+t43*t10-2.0*t50*t51-t55
    *a[2][2]+t54*t5;
  double t59 = 1/t58;
  double t64 = a[0][2]*a[1][2];
  double t68 = a[0][3]*a[1][3];
  double t71 = (a[0][1]*a[2][2]*a[3][3]-a[0][1]*t3-t64*a[3][3]+t51*a[2][3]+t50*a
	 [2][3]-t68*a[2][2])*t59;
  double t75 = a[0][3]*a[1][1];
  double t80 = (-t29*a[3][3]+t36*a[2][3]+t46*a[3][3]-t75*a[2][3]-a[0][2]*t10+t68*a
	 [1][2])*t59;
  
  double t88 = (-t29*a[2][3]+t36*a[2][2]+t46*a[2][3]-t75*a[2][2]-t64*a[1][3]+a[0]
	 [3]*t5)*t59;
  double t93 = a[0][2]*a[0][3];
  double t100 = a[0][0]*a[1][3];
  double t102 = a[0][2]*a[0][1];
  double t105 = a[0][3]*a[0][1];
  double t109 = (-t19*a[3][3]+t100*a[2][3]+t102*a[3][3]-t93*a[1][3]-t105*a[2][3]+
	  t54*a[1][2])*t59;
  double t117 = (-t19*a[2][3]+t100*a[2][2]+t102*a[2][3]-t43*a[1][3]-t105*a[2][2]+
	  t93*a[1][2])*t59;
  double t131 = (-t13*a[2][3]+t100*a[1][2]+t25*a[2][3]-t102*a[1][3]-t105*a[1][2]+
	  t93*a[1][1])*t59;
  imat[0][0] = -(-a[1][1]*a[2][2]*a[3][3]+a[1][1]*t3+t5*a[3][3]-2.0*a[1][2]
		 *a[1][3]*a[2][3]+t10*a[2][2])*t59;
  imat[0][1] = -t71;
  imat[0][2] = -t80;
  imat[0][3] = t88;
  imat[1][0] = -t71;
  imat[1][1] = -(-a[0][0]*a[2][2]*a[3][3]+a[0][0]*t3+t43*a[3][3]-2.0*t93*a
		 [2][3]+a[2][2]*t54)*t59;
  imat[1][2] = t109;
  imat[1][3] = -t117;
  imat[2][0] = -t80;
  imat[2][1] = t109;
  imat[2][2] = -(-t13*a[3][3]+t23+t25*a[3][3]-2.0*t105*a[1][3]+t55)*t59;
  imat[2][3] = t131;
  imat[3][0] = t88;
  imat[3][1] = -t117;
  imat[3][2] = t131;
  imat[3][3] = -(-t13*a[2][2]+t17+t26-2.0*t102*a[1][2]+t44)*t59;
}

void compute_solution2(double x[2], double a[2][2], double b[2])
{
  double imat[2][2];
  int i, j;
  compute_inverse2(imat, a);
  for (i=0;i<2;++i)
    for (x[i]=0, j=0; j<2; ++j)
      x[i] += imat[i][j] * b[j];
}

void compute_solution3(double x[3], double a[3][3], double b[3])
{
  double imat[3][3];
  int i, j;
  compute_inverse3(imat, a);
  for (i=0;i<3;++i)
    for (x[i]=0, j=0; j<3; ++j)
      x[i] += imat[i][j] * b[j];
}

void compute_solution4(double x[4], double a[4][4], double b[4])
{
  double imat[4][4];
  int i, j;
  compute_inverse4(imat, a);
  for (i=0;i<4;++i)
    for (x[i]=0, j=0; j<4; ++j)
      x[i] += imat[i][j] * b[j];
}

double compute_dot2(double a[2][2], double b[2], int i0)
{
  double y[2];
  compute_solution2(y, a, b);
  //printf("i0,y[0],y[1]=%d,%f,%f\n",i0,y[0],y[1]);
  return y[0] + y[1] * i0;
}

double compute_dot3(double a[3][3], double b[3], int i0, int i1)
{
  double y[3];
  compute_solution3(y, a, b);
  return y[0] + y[1] * i0 + y[2] * i1;
}

double compute_dot4(double a[4][4], double b[4], int i0, int i1, int i2)
{
  double y[4];
  compute_solution4(y, a, b);
  return y[0] + y[1] * i0 + y[2] * i1 + y[3] * i2;
}
