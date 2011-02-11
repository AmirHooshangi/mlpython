"""
The ``learners.features`` module contains Learners meant for feature
or representation learning. The MLProblems for these Learners should
be iterators over inputs. Their output should be a new feature
representation of the input.

The currently implemented algorithms are:

* ``PCA``: Principal Component Analysis learner.

"""

from generic import Learner
import numpy as np
import mlpython.mlproblems.generic as mlpb

class PCA(Learner):
    """
    Principal Component Analysis.

    Outputs the input's projection on the principal components, so as
    to obtain a representation with mean zero and identity covariance.

    Option ``n_components`` is the number of principal components to
    compute.

    Option ``regularizer`` is a small constant to add to the diagonal
    of the estimated covariance matrix (default=1e-10).

    **Required metadata:**

    * ``'input_size'``: size of the inputs

    """
    def __init__( self,
                  n_components,
                  regularizer=1e-10
                  ):
        self.n_components = n_components
        self.regularizer = regularizer

    def train(self,trainset):
        """
        Extract principal components.
        """

        # Put data in Numpy matrix
        input_size = trainset.metadata['input_size']
        trainmat = np.zeros((len(trainset),input_size))
        t = 0
        for input in trainset:
            trainmat[t,:] = input
            t+=1

        # Compute mean and covariance
        self.mean = trainmat.mean(axis=0)
        train_cov = np.cov(trainmat,rowvar=0)
        # Add a small constant on the diagonal, to regularize
        train_cov += np.diag(self.regularizer*np.ones(input_size))

        ## Compute principal components
        w,v = np.linalg.eigh(train_cov)
        s = (-w).argsort()
        w = w[s]
        v = v[:,s]

        self.transform = (1./np.sqrt(w[:self.n_components])).reshape((1,-1))*v[:,:self.n_components]

    def forget(self):
        del self.transform
        del self.mean

    def use(self,dataset):
        """
        Outputs the projection on the principal components, so as to obtain
        a representation with mean zero and identity covariance.
        """
        return [ np.dot(input-self.mean,self.transform) for input in dataset ]

    def test(self,dataset):
        """
        Outputs the squared error of the reconstructed inputs.
        """
        outputs = self.use(dataset)
        costs = zeros(len(dataset),1)
        for input,output,cost in zip(dataset,outputs,costs):
            cost[0] = np.sum((input-self.mean -np.dot(output,self.transform.T))**2)

        return outputs,costs
