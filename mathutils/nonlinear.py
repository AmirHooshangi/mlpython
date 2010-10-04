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

def softmax(input,output):
    """
    Computes the softmax function softmax(input) = exp(input)/sum(exp(input)) = output.
    """
    nonlinear_.softmax_vec_(input,output)

