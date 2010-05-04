/* tifffile.c

A Python C extension module for decoding PackBits and LZW encoded TIFF data.

Refer to the tifffile.py module for documentation and tests.

Tested on Python 2.6 and 3.1, 32-bit and 64-bit.

Authors:
  Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>,
  Laboratory for Fluorescence Dynamics. University of California, Irvine.

Copyright (c) 2008-2010, The Regents of the University of California
Produced by the Laboratory for Fluorescence Dynamics.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holders nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*****************************************************************************/
/* setup.py

"""A Python script to build the _tifffile extension module.

Usage:: ``python setup.py build_ext --inplace``

"""

from distutils.core import setup, Extension
import numpy

setup(name='_tifffile', ext_modules=[Extension('_tifffile', ['tifffile.c'],
    include_dirs=[numpy.get_include()], extra_compile_args=[])],)


******************************************************************************/

#define _VERSION_ "2010.04.10"

#define WIN32_LEAN_AND_MEAN

#include "Python.h"
#include "string.h"
#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION < 3
#ifndef PyBytes_FromFormat
#define PyBytes_CheckExact(o)              PyString_CheckExact(o)
#define PyBytes_Check(o)                   PyString_Check(o)
#define PyBytes_FromStringAndSize(s, len)  PyString_FromStringAndSize(s, len)
#define PyBytes_FromFormat                 PyString_FromFormat
#define PyBytes_GET_SIZE(s)                PyString_GET_SIZE(s)
#define PyBytes_AS_STRING(s)               PyString_AS_STRING(s)
#endif
#endif

#define SWAP2BYTES(x) ((((x)>>8)&0xff) | (((x)&0xff)<<8))
#define SWAP4BYTES(x) ((((x)>>24)&0xff) | (((x)&0xff)<<24) | \
                       (((x)>>8)&0xff00) | (((x)&0xff00)<<8))

struct BYTE_STRING {
    unsigned int ref; /* reference count */
    unsigned int len; /* length of string */
    char *str;        /* pointer to bytes */
};

/*****************************************************************************/
/* C functions */

/*****************************************************************************/
/* Python functions */

/*
Decode TIFF PackBits encoded string.
*/
char py_decodepackbits_doc[] = "Return TIFF PackBits decoded string.";

static PyObject *
py_decodepackbits(PyObject *obj, PyObject *args)
{
    int n;
    char e;
    char *decoded = NULL;
    char *encoded = NULL;
    char *encoded_end = NULL;
    char *encoded_pos = NULL;
    unsigned int encoded_len;
    unsigned int decoded_len;
    PyObject *byteobj = NULL;
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(args, "O", &byteobj))
        return NULL;

    if (!PyBytes_Check(byteobj)) {
        PyErr_Format(PyExc_TypeError, "expected byte string as input");
        goto _fail;
    }

    Py_INCREF(byteobj);
    encoded = PyBytes_AS_STRING(byteobj);
    encoded_len = (unsigned int)PyBytes_GET_SIZE(byteobj);

    /* release GIL: byte/string objects are immutable */
    Py_BEGIN_ALLOW_THREADS

    /* determine size of decoded string */
    encoded_pos = encoded;
    encoded_end = encoded + encoded_len;
    decoded_len = 0;
    while (encoded_pos < encoded_end) {
        n = (int)*encoded_pos++;
        if (n >= 0) {
            n++;
            if (encoded_pos+n > encoded_end)
                n = (int)(encoded_end - encoded_pos);
            encoded_pos += n;
            decoded_len += n;
        } else if (n > -128) {
            encoded_pos++;
            decoded_len += 1-n;
        }
    }
    Py_END_ALLOW_THREADS

    result = PyBytes_FromStringAndSize(0, decoded_len);
    if (result == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to allocate decoded string");
        goto _fail;
    }
    decoded = PyBytes_AS_STRING(result);

    Py_BEGIN_ALLOW_THREADS

    /* decode string */
    encoded_end = encoded + encoded_len;
    while (encoded < encoded_end) {
        n = (int)*encoded++;
        if (n >= 0) {
            n++;
            if (encoded+n > encoded_end)
                n = (int)(encoded_end - encoded);
            /* memmove(decoded, encoded, n); decoded += n; encoded += n; */
            while (n--)
                *decoded++ = *encoded++;
        } else if (n > -128) {
            n = 1 - n;
            e = *encoded++;
            /* memset(decoded, e, n); decoded += n; */
            while (n--)
                *decoded++ = e;
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(byteobj);
    return result;

  _fail:
    Py_XDECREF(byteobj);
    Py_XDECREF(result);
    return NULL;
}

/*
Decode TIFF LZW encoded string.
*/
char py_decodelzw_doc[] = "Return TIFF LZW decoded string.";

static PyObject *
py_decodelzw(PyObject *obj, PyObject *args)
{
    PyThreadState *_save = NULL;
    PyObject *byteobj = NULL;
    PyObject *result = NULL;
    int i, j;
    unsigned int encoded_len = 0;
    unsigned int decoded_len = 0;
    unsigned int result_len = 0;
    unsigned int table_len = 0;
    unsigned int len;
    unsigned int code, c, oldcode, mask, bitw, shr, bitcount;
    char *encoded = NULL;
    char *result_ptr = NULL;
    char *table2 = NULL;
    char *cptr;
    struct BYTE_STRING *decoded = NULL;
    struct BYTE_STRING *table[4096];
    struct BYTE_STRING *decoded_ptr = NULL, *newentry, *newresult, *t;
    int little_endian = 0;

    if (!PyArg_ParseTuple(args, "O", &byteobj))
        return NULL;

    if (!PyBytes_Check(byteobj)) {
        PyErr_Format(PyExc_TypeError, "expected byte string as input");
        goto _fail;
    }

    Py_INCREF(byteobj);
    encoded = PyBytes_AS_STRING(byteobj);
    encoded_len = (unsigned int)PyBytes_GET_SIZE(byteobj);

    /* release GIL: byte/string objects are immutable */
    _save = PyEval_SaveThread();

    if ((*encoded != -128) || ((*(encoded+1) & 128))) {
        PyEval_RestoreThread(_save);
        PyErr_Format(PyExc_ValueError,
            "strip must begin with CLEAR code");
        goto _fail;
    }
    little_endian = (*(unsigned short *)encoded) & 128;

    /* allocate buffer for codes and pointers */
    decoded_len = 0;
    len = (encoded_len + encoded_len/9) * sizeof(decoded);
    decoded = PyMem_Malloc(len * sizeof(void *));
    if (decoded == NULL) {
        PyEval_RestoreThread(_save);
        PyErr_Format(PyExc_MemoryError, "failed to allocate decoded");
        goto _fail;
    }
    memset((void *)decoded, 0, len * sizeof(void *));
    decoded_ptr = decoded;

    /* cache strings of length 2 */
    cptr = table2 = PyMem_Malloc(256*256*2 * sizeof(char));
    if (table2 == NULL) {
        PyEval_RestoreThread(_save);
        PyErr_Format(PyExc_MemoryError, "failed to allocate table2");
        goto _fail;
    }
    for (i = 0; i < 256; i++) {
        for (j = 0; j < 256; j++) {
            *cptr++ = (char)i;
            *cptr++ = (char)j;
        }
    }

    memset(table, 0, sizeof(table));
    table_len = 258;
    bitw = 9;
    shr = 23;
    mask = 4286578688;
    bitcount = 0;
    result_len = 0;
    code = 0;
    oldcode = 0;

    while ((unsigned int)((bitcount + bitw) >> 3) <= encoded_len) {
        /* read next code */
        code = *((unsigned int *)((void *)(encoded + (bitcount / 8))));
        if (little_endian)
            code = SWAP4BYTES(code);
        code <<= bitcount % 8;
        code &= mask;
        code >>= shr;
        bitcount += bitw;

        if (code == 257) /* end of information */
            break;

        if (code == 256) {  /* clearcode */
            /* initialize table and switch to 9 bit */
            while (table_len > 258) {
                t = table[--table_len];
                t->ref--;
                if (t->ref == 0) {
                    if (t->len > 2)
                        PyMem_Free(t->str);
                    PyMem_Free(t);
                }
            }
            bitw = 9;
            shr = 23;
            mask = 4286578688;

            /* read next code */
            code = *((unsigned int *)((void *)(encoded + (bitcount / 8))));
            if (little_endian)
                code = SWAP4BYTES(code);
            code <<= bitcount % 8;
            code &= mask;
            code >>= shr;
            bitcount += bitw;

            if (code == 257) /* end of information */
                break;

            /* decoded.append(table[code]) */
            if (code < 256) {
                result_len++;
                *((int *)decoded_ptr++) = code;
            } else {
                newresult = table[code];
                newresult->ref++;
                result_len += newresult->len;
                 *(struct BYTE_STRING **)decoded_ptr++ = newresult;
            }
        } else {
            if (code < table_len) {
                /* code is in table */
                /* newresult = table[code]; */
                /* newentry = table[oldcode] + table[code][0] */
                /* decoded.append(newresult); table.append(newentry) */
                if (code < 256) {
                    c = code;
                    *((unsigned int *)decoded_ptr++) = code;
                    result_len++;
                } else {
                    newresult = table[code];
                    newresult->ref++;
                    c = (unsigned int) *newresult->str;
                    *(struct BYTE_STRING **)decoded_ptr++ = newresult;
                    result_len += newresult->len;
                }
                newentry = PyMem_Malloc(sizeof(struct BYTE_STRING));
                newentry->ref = 1;
                if (oldcode < 256) {
                    newentry->len = 2;
                    newentry->str = table2 + (oldcode << 9) +
                                    ((unsigned char)c << 1);
                } else {
                    len = table[oldcode]->len;
                    newentry->len = len + 1;
                    newentry->str = PyMem_Malloc(newentry->len);
                    if (newentry->str == NULL)
                        break;
                    memmove(newentry->str, table[oldcode]->str, len);
                    newentry->str[len] = c;
                }
                table[table_len++] = newentry;
            } else {
                /* code is not in table */
                /* newentry = newresult = table[oldcode] + table[oldcode][0] */
                /* decoded.append(newresult); table.append(newentry) */
                newresult = PyMem_Malloc(sizeof(struct BYTE_STRING));
                newentry = newresult;
                newentry->ref = 2;
                if (oldcode < 256) {
                    newentry->len = 2;
                    newentry->str = table2 + 514*oldcode;
                } else {
                    len = table[oldcode]->len;
                    newentry->len = len + 1;
                    newentry->str = PyMem_Malloc(newentry->len);
                    if (newentry->str == NULL)
                        break;
                    memmove(newentry->str, table[oldcode]->str, len);
                    newentry->str[len] = *table[oldcode]->str;
                }
                table[table_len++] = newentry;
                *(struct BYTE_STRING **)decoded_ptr++ = newresult;
                result_len += newresult->len;
            }
        }
        oldcode = code;
        /* increase bit-width if necessary */
        switch (table_len) {
            case 511:
                bitw = 10;
                shr = 22;
                mask = 4290772992;
                break;
            case 1023:
                bitw = 11;
                shr = 21;
                mask = 4292870144;
                break;
            case 2047:
                bitw = 12;
                shr = 20;
                mask = 4293918720;
        }
    }

    PyEval_RestoreThread(_save);

    if (code != 257) {
      printf("py_decodelzw: unexpected end of stream (code=%d), ignoring\n", code);
      //PyErr_Format(PyExc_TypeError, "unexpected end of stream");
      //goto _fail;
    }

    /* result = ''.join(decoded) */
    decoded_len = (unsigned int)(decoded_ptr - decoded);
    decoded_ptr = decoded;
    result = PyBytes_FromStringAndSize(0, result_len);
    if (result == NULL) {
        PyErr_Format(PyExc_MemoryError, "failed to allocate decoded string");
        goto _fail;
    }
    result_ptr = PyBytes_AS_STRING(result);

    _save = PyEval_SaveThread();

    while (decoded_len--) {
        code = *((unsigned int *)decoded_ptr);
        if (code < 256) {
            *result_ptr++ = (char)code;
        } else {
            t = *((struct BYTE_STRING **)decoded_ptr);
            memmove(result_ptr, t->str, t->len);
            result_ptr +=  t->len;
            if (--t->ref == 0) {
                if (t->len > 2)
                    PyMem_Free(t->str);
                PyMem_Free(t);
            }
        }
        decoded_ptr++;
    }
    PyMem_Free(decoded);

    while (table_len-- > 258) {
        t = table[table_len];
        if (t->len > 2)
            PyMem_Free(t->str);
        PyMem_Free(t);
    }
    PyMem_Free(table2);

    PyEval_RestoreThread(_save);

    Py_DECREF(byteobj);
    return result;

  _fail:
    if (table2 != NULL)
        PyMem_Free(table2);
    if (decoded != NULL) {
        while (decoded_len--) {
            code = *((unsigned int *) decoded_ptr);
            if (code > 258) {
                t = *((struct BYTE_STRING **) decoded_ptr);
                if (--t->ref == 0) {
                    if (t->len > 2)
                        PyMem_Free(t->str);
                    PyMem_Free(t);
                }
            }
        }
        PyMem_Free(decoded);
    }
    while (table_len-- > 258) {
        t = table[table_len];
        if (t->len > 2)
            PyMem_Free(t->str);
        PyMem_Free(t);
    }

    Py_XDECREF(byteobj);
    Py_XDECREF(result);

    return NULL;
}

/*****************************************************************************/
/* Create Python module */

char module_doc[] =
    "A Python C extension module for decoding PackBits and LZW encoded "
    "TIFF data.\n\n"
    "Refer to the tifffile.py module for documentation and tests.\n\n"
    "Authors:\n  Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>\n"
    "  Laboratory for Fluorescence Dynamics, University of California, Irvine."
    "\n\nVersion: %s\n";

static PyMethodDef module_methods[] = {
    {"decodelzw", (PyCFunction)py_decodelzw, METH_VARARGS,
        py_decodelzw_doc},
    {"decodepackbits", (PyCFunction)py_decodepackbits, METH_VARARGS,
        py_decodepackbits_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_tifffile",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        module_traverse,
        module_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit__tifffile(void)

#else

#define INITERROR return

PyMODINIT_FUNC 
init_tifffile(void) 

#endif
{
    PyObject *module;

    char *doc = (char *)PyMem_Malloc(sizeof(module_doc) + sizeof(_VERSION_));
    sprintf(doc, module_doc, _VERSION_);

#if PY_MAJOR_VERSION >= 3
    moduledef.m_doc = doc;
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3("_tifffile", module_methods, doc);
#endif

    PyMem_Free(doc);

    if (module == NULL)
        INITERROR;

    if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    {    
#if PY_MAJOR_VERSION < 3
    PyObject *s = PyString_FromString(_VERSION_);
#else
    PyObject *s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
