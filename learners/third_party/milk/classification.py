# Copyright 2011 David Brouillard - Guillaume Roy-Fontaine. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY David Brouillard - Guillaume Roy-Fontaine ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL David Brouillard - Guillaume Roy-Fontaine OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of David Brouillard - Guillaume Roy-Fontaine.

"""
The ``learners.third_party.milk.classification`` module contains 
an Tree Classifier based on the MILK library:
"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import milk as libmilk
except ImportError:
    print 'Warning in mlpython.learners.third_party.milk.classification:''import milk'' failed. The MILK library is not properly installed. See mlpython/misc/third_party/milk/README for instructions.'


class TreeClassifier(Learner):
    """ 
    Decision Tree Classifier using MILK library
 
    A decision tree classifier (currently, implements the greedy ID3
    algorithm without any pruning).

    Attributes
    ----------
    criterion : function, optional
        criterion to use for tree construction,
        this should be a function that receives a set of labels
        (default: information_gain).

        Information Gain
        See http://en.wikipedia.org/wiki/Information_gain_in_decision_trees

        The function calculated here does not include the original entropy unless
        you explicitly ask for it (by passing include_entropy=True)

        z1_loss
        zero-one loss split for tree learning

    min_split : integer, optional
        minimum size to split on (default: 4).
        
    Other options
    (see http://packages.python.org/milk/index.html for more details).

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    """
    def __init__(self, criterion='information_gain', min_split=4, include_entropy=False, weights0=None, weights1=None, return_label=True):
        self.criterion = 'criterion'
        self.min_split = min_split
        self.include_entropy = include_entropy
        self.weights0 = weights0
        self.weights1 = weights1
        self.return_label = return_label
        # HUGO: ajouter les autres options associee aux criterions

        #self.subsample = subsample
        #self.R = R

    def train(self,trainset):
        """
        Trains the data with the Milk Tree Learner.
        """  
        print '\nTRAIN BEGIN!\n'

        self.n_classes = len(trainset.metadata['targets'])
        # HUGO: raise error if number of classes > 2
        if self.n_classes > 2:
            raise ValueError('Invalid. Should have 2 classes.')
        
        
        features = np.zeros((len(trainset),trainset.metadata['input_size']))
        labels = np.zeros((len(trainset)))
        for i,xy in enumerate(trainset):
            x,y = xy
            features[i] = x
            labels[i] = y

        if self.criterion == 'information_gain':
            def criterion_fcn(labels0=self.labels0, labels1=self.labels1):
                return libmilk.supervised.tree.information_gain(labels0, labels1, include_entropy=self.include_entropy)
        elif self.criterion == 'z1_loss':
            return libmilk.supervised.tree.z1_loss(labels0, labels1, weights0=self.weights0, weights1=self.weights1)
        else:
                raise ValueError('Invalid parameter: '+self.criterion+'. Should be either \'information_gain\' or \'z1_loss\'')

        learner = libmilk.supervised.tree_learner(criterion=criterion_fcn,min_split=self.min_split,return_label=self.return_label) # HUGO: passer des options!
        #self.subsample = subsample
        #self.R = R
        model = learner.train(features, labels)
        
        self.tree = model
        
        print '\nTRAIN DONE!\n'

    def use(self,dataset):
        features = []
        outputs = np.zeros((len(dataset),1))
        for xy in dataset:
            x,y = xy
            features += [x]

        for test,out in zip(features,outputs):
            out[0] = self.tree.apply(test)
        
        return outputs

    def forget(self):
        self.tree = None

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
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
        



