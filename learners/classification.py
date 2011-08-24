"""
The ``learners.classification`` module contains Learners meant for classification problems. 
They normally will require (at least) the metadata ``'targets'``.
The MLProblems for these Learners should be iterators over pairs
of inputs and targets, with the target being a class index.

The currently implemented algorithms are:

* ``BayesClassifier``: a Bayes classifier obtained from density estimators.

"""

from generic import Learner
import numpy as np
import mlpython.mlproblems.classification as mlpb


class BayesClassifier(Learner):
    """ 
    Bayes classifier from density estimators
 
    Given one density learner per class (option ``estimators``), this
    learner will train each one on a separate class and classify
    examples using Bayes' rule.

    **Required metadata:**
    
    * ``'targets'``

    """
    def __init__(   self,
                    estimators=[],# The density learners to be trained
                    ):
        self.stage = 0
        self.estimators = estimators

    def train(self,trainset):
        """
        Trains each estimator. Each call to train increments ``self.stage`` by 1.
        If ``self.stage == 0``, first initialize the model.
        """

        self.n_classes = len(trainset.metadata['targets'])

        # Initialize model
        if self.stage == 0:
            # Split data according to classes
            self.class_trainset = []
            tot_len = len(trainset)
            self.prior = np.zeros((self.n_classes))
            for c in xrange(self.n_classes):
                trainset_c = mlpb.ClassSubsetProblem(data=trainset,metadata=trainset.metadata,
                                                     subset=set([c]),
                                                     include_class=False)
                trainset_c.setup()
                self.class_trainset += [ trainset_c ]
                self.prior[c] = float(len(trainset_c))/tot_len

        # Training each estimators
        for c in xrange(self.n_classes):
            self.estimators[c].train(self.class_trainset[c])
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        # Initialize estimators
        for c in xrange(self.n_classes):
            self.estimators[c].forget()
        self.prior = 1./self.n_classes * np.ones((self.n_classes))

    def use(self,dataset):
        """
        Outputs the class_id chosen by the algorithm, for each
        example in the dataset.
        """
        outputs = -1*np.ones((len(dataset),1))
        for xy,pred in zip(dataset,outputs):
            x,y = xy
            max_prob = -np.inf
            max_prob_class = -1
            for c in xrange(self.n_classes):
                prob_c = self.estimators[c].use([x])[0] + np.log(self.prior[c])
                if max_prob < prob_c:
                    max_prob = prob_c
                    max_prob_class = c
                
            pred[0] = max_prob_class
            
        return outputs

    def test(self,dataset):
        """
        Outputs the class_id chosen by the algorithm and
        the classification error cost for each example in the dataset
        """
        outputs = self.use(dataset)
        costs = np.ones((len(outputs),1))
        # Compute classification error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            if y == pred[0]:
                cost[0] = 0

        return outputs,costs
