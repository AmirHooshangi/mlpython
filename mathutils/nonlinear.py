"""
The ``mathutils.nonlinear`` module contains several useful nonlinear
functions on Numpy arrays. All functions avoid memory allocation, by
requiring the Numpy array in which to write the answer. All arrays
should be double arrays.

This module defines the following functions:

* ``sigmoid``:         Computes the sigmoid function.
* ``dsigmoid``:        Computes the derivative of a sigmoid function with respect to its input.
* ``reclin``:          Computes the rectified linear function.
* ``dreclin``:         Computes the derivative of a rectified linear function with respect to its input.
* ``softplus``:        Computes the softplus function.
* ``softmax``:         Computes the softmax function.

"""

import numpy as np
import scipy.weave
import nonlinear_

def sigmoid(input,output):
    """
    Computes the sigmoid function sigm(input) = 1/(1+exp(-input)) = output
    """
    nonlinear_.sigmoid_(input,output)

def dsigmoid(output,doutput,dinput):
    """
    Computes the derivative of a sigmoid function with respect to its input, 
    given the output of the sigmoid and the derivative on the output.
    """
    nonlinear_.dsigmoid_(output,doutput,dinput)

def reclin(input,output):
    """
    Computes the rectified linear function reclin(input) = 1_{input>0}*input = output
    """
    nonlinear_.reclin_(input,output)

def dreclin(output,doutput,dinput):
    """
    Computes the derivative of a rectified linear function with respect to its input, 
    given its output and the derivative on the output.
    """
    nonlinear_.dreclin_(output,doutput,dinput)

def softplus(input,output):
    """
    Computes the softplus function softplus(input) = log(1+exp(input))
    """
    nonlinear_.softplus_(input,output)

def softmax(input,output):
    """
    Computes the softmax function softmax(input) = exp(input)/sum(exp(input)) = output.
    """
    nonlinear_.softmax_vec_(input,output)

