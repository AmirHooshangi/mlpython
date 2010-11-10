"""
The ``learners.density`` module contains Learners meant for density or
distribution estimation problems.  The MLProblems for these Learners
should be iterators over inputs.

The currently implemented algorithms are:

* ``BagDensity``: a density estimation learner where each example is a bag of inputs.

"""

from generic import Learner
import numpy as np
import mlpython.mlproblems.generic as mlpb

class BagDensity(Learner):
    """
    A density estimation learner where each example is a bag of inputs.

    Given a density learner (given by the user), this learner
    will train it on all inputs in all bags. It is
    assumed that the density learner outputs its estimate
    of the log-density (when calling ``use(...)``).

    """
    def __init__(   self,
                    estimator=None,# The density learner to be trained
                    ):
        self.stage = 0
        self.estimator = estimator

    def train(self,trainset):
        """
        Trains the estimator on all examples in all bags.
        Each call to train increments ``self.stage`` by 1.
        """

        self.density_trainset = mlpb.MergedProblem(data=trainset,metadata=trainset.metadata)
        self.estimator.train(self.density_trainset)
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        self.estimator.forget()

    def use(self,dataset):
        """
        Outputs the sum of the density learning outputs for 
        all inputs in each bag (example).
        """
        outputs = np.zeros((len(dataset),1))
        for bag,pred in zip(dataset,outputs):
            out = 0
            for x in bag:
                out += self.estimator.use([x])[0]
            pred[0] = out
            
        return outputs

    def test(self,dataset):
        """
        Outputs the NLLs of each example, normalized
        by the size of the example's bag.
        """
        outputs = self.use(dataset)
        costs = zeros(len(dataset),1)
        for bag,o,c in zip(dataset,outputs,costs):
            c[0] = -o[0]/len(bag)

        return outputs,costs
