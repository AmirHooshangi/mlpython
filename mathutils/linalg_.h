/* This file interfaces with useful non-Python routines. 
   To use them however, see linalg.py. */

static PyObject *product_matrix_matrix_(PyObject *self, PyObject *args);
static PyObject *product_matrix_vector_(PyObject *self, PyObject *args);
static PyObject *getdiag_(PyObject *self, PyObject *args);
static PyObject *solve_(PyObject *self, PyObject *args);
static PyObject *lu_(PyObject *self, PyObject *args);

