"""
The ``learners.generic`` module contains Learners that are not designed for a specific
type of problem or data. They mostly serve as interfaces to derive new
Learners.

This module contains the following classes:

* Learner:         Root class for all learning algorithms.
* OnlineLearner:   Interface for Learners that can be traiend "online".

"""

class Learner:
    """
    Root class or interface for a learning algorithm.

    All Learner objects inherit from this class. It is meant to
    standardize the behavior of all learning algorithms.

    """

    #def __init__():
    
    def train(self,trainset):
        """
        Runs the learning algorithm on ``trainset``
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def forget(self):
        """
        Resets the Learner to its original state.
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def use(self,dataset):
        """
        Computes and returns the output of the Learner for
        ``dataset``. The method should return an iterator over these
        outputs.
        """
        raise NotImplementedError("Subclass should have implemented this method.")

    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the cost of 
        those outputs for ``dataset``. The method should return a pair of two iterators, the first
        being over the outputs and the second over the costs.
        """
        raise NotImplementedError("Subclass should have implemented this method.")


class OnlineLearner(Learner):
    """
    Class (interface) for Learners that can be traiend "online".
    
    This class interface makes it easier to construct a learner. All
    that must be defined are four following methods:
    
    * ``initialize_learner(self,metadata)``
    * ``update_learner(self,example)``
    * ``use_learner(self,example)``
    * ``cost(self,output,example)``
    
    Method ``initialize_learner()`` simply initializes
    the learner. The training set's 'metadata' is also available.
    
    Method ``update_learner()`` updates the learner's parameters
    using the given 'example'.
    
    Method ``use_learner()`` should return the output
    for the given 'example'. The output should be a sequence
    (even if it has just one element in it), to allow
    for multiple outputs. Make sure not to return an object
    that is referenced internally and is still
    being used by the class object.
    
    Method ``cost()`` should return the cost associated
    to some 'output' for the given 'example'. The returned 
    cost should be a sequence (even if it has just one 
    element in it), to allow for multiple costs.
    
    Option ``n_stages`` specifies how many iterations over the 
    training set is done at every call of ``train()``.
    
    All other hyper-parameters for the learner supplied
    through the constructor will be assigned as attributes
    to the object, and hence will be accessible
    by all methods.
    
    Example of methods for a linear perceptron. ::

       import numpy as np
       import mlpython

       class Perceptron(mlpython.learners.OnlineLearner):
       
          def initialize_learner(self,metadata):
             self.w = np.zeros((metadata['input_size']))
             self.b = 0.
          
          def update_learner(self,example):
             input = example[0]
             target = 2*example[1]-1 # Targets are 0/1
             output = np.dot(self.w,input)+self.b
             if np.sign(output) != target:
                self.w += self.lr * target * input
                self.b += self.lr * target
          
          def use_learner(self,example):
             return [np.sign(np.dot(self.w,example[0])+self.b)]
          
          def cost(self,output,example):
             return [int(output != 2*example[1]-1)]

    When creating an instance, must provide the
    value of the hyper-parameter lr: ::

       perceptron = Perceptron(1,lr=0.01)

    Alternatively, one could override the constructor to
    specify some default hyper-parameters: ::
    
       class Perceptron(mlpython.learners.OnlineLearner):
             
          def __init__(self, n_stages, 
                       lr = 0.01):
              self.n_stages = n_stages
              self.stage = 0
              self.lr = lr

    """
    
    def __init__(self, n_stages, **kw):
    
        self.n_stages = n_stages
        self.stage = 0
        for k,v in kw.iteritems():
            setattr(self,k,v)

    def train(self,trainset):
        if self.stage == 0:
            self.initialize_learner(trainset.metadata)
        for it in range(self.stage,self.n_stages):
            for example in trainset:
                self.update_learner(example)
        self.stage = self.n_stages

    def forget(self):
        self.stage = 0

    def use(self,dataset):
        outputs = []
        for example in dataset:
            outputs += [self.use_learner(example)]
        return outputs
            
    def test(self,dataset):
        outputs = self.use(dataset)
        costs = []
        for example,output in zip(dataset,outputs):
            costs += [self.cost(output,example)]
        return outputs,costs

    def initialize_learner(self,metadata):
        raise NotImplementedError("Subclass should have implemented this method.")

    def update_learner(self,example):
        raise NotImplementedError("Subclass should have implemented this method.")

    def use_learner(self,example):
        raise NotImplementedError("Subclass should have implemented this method.")

    def cost(self,output,example):
        raise NotImplementedError("Subclass should have implemented this method.")
