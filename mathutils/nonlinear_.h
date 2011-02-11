/* This file interfaces with useful non-Python routines. 
   To use them however, see nonlinear.py. */

static PyObject *sigmoid_(PyObject *self, PyObject *args);
static PyObject *dsigmoid_(PyObject *self, PyObject *args);
static PyObject *softmax_vec_(PyObject *self, PyObject *args);
static PyObject *reclin_(PyObject *self, PyObject *args);
static PyObject *dreclin_(PyObject *self, PyObject *args);
static PyObject *softplus_(PyObject *self, PyObject *args);
