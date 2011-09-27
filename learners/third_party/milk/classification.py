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
    Tree Classifier using MILK library
    
    TODO: REFAIRE POUR MILK
 
    Examples should be input and target pairs. The input can use
    either a sparse or coarse representation, as returned by the
    ``mlpython.misc.io.libsvm_load`` function.  Option ``kernel`` (which
    can be either ``'linear'``, ``'polynomial'``, ``'rbf'`` or
    ``'sigmoid'``) determines the type of kernel.

    Weights to examples of different classes can be given using 
    option ``label_weights``, which must be a dictionary mapping from 
    the label (string) to the weight (float).

    The SVM will also output probabilities if option ``'output_probabilities'``
    is True.

    Other options are the same as those in the LIBSVM implementation
    (see http://www.csie.ntu.edu.tw/~cjlin/libsvm for more details).

    **Required metadata:**

    * ``'targets'``
    * ``'class_to_id'``

    """
    #def __init__(self,
                 #kernel='linear',
                 #degree=3,
                 #gamma=1,
                 #coef0=0,
                 #C=1,
                 #tolerance=0.001,
                 #cache_size=100,
                 #shrinking=True,
                 #output_probabilities=False,
                 #label_weights = None
                 #):
        
        #self.kernel = kernel
        #self.degree = degree
        #self.gamma = float(gamma)
        #self.coef0 = float(coef0)
        #self.C = float(C)
        #self.tolerance = float(tolerance)
        #self.cache_size = cache_size
        #self.shrinking = shrinking
        #self.output_probabilities = output_probabilities
        #self.label_weights = label_weights
        
    def __init__(self, criterion='information_gain', min_split=4, return_label=True, subsample=None, R=None):
        self.criterion = 'criterion'
        self.min_split = min_split
        self.return_label = return_label
        #self.subsample = subsample
        #self.R = R

    def train(self,trainset):
        """
        Trains the data with the Milk Tree Learner.
        """  
        print '\nTRAIN BEGIN!\n'     
        #features = np.random.rand(100,10)
        #labels = np.zeros(100)
        #features[50:] += .5
        #labels[50:] = 1
        
        features = []
        labels = []
        for input,target in trainset:
            features += [input]
            labels += [target]
            
        #print '\nFeatures : \n'
        #for target2 in features:
            #print target2
            
        #print '\nLabels : \n'
        #for target3 in labels:
            #print target3
        
        learner = libmilk.supervised.tree_learner()
        model = learner.train(features, labels)
        
        self.tree = model
        
        print '\nTRAIN DONE!\n'

    #def use(self,dataset):
        #"""
        #Outputs the class_id chosen by the algorithm. If 
        #``output_probabilities`` is True, also outputs the vector
        #of probabilities.
        #"""
        
        #features = []
        #labels = []
        #for input,target in dataset:
            #features += [input]
            #labels += [target]
            
        #myOutput = np.zeros((len(datasets),1))
        #print '\nTest Features : \n'
        #for target2 in features:
            ##print self.tree.apply(target2)
            ##target3 += [self.tree.apply(target2)]
            #if [self.tree.apply(target2)] == [True]:
                #myOutput += [0]
            #else:
                #myOutput += [1]
        
        #print myOutput
            
        #outputs = myOutput
            
        #return outputs
        
        
    def use(self,dataset):
        features = []
        outputs = np.zeros((len(dataset),1))
        for xy in dataset:
            x,y = xy
            features += [x]

        for test,out in zip(features,outputs):
            temp = [self.tree.apply(test)]
            if temp == [True]:
                out[0] = 0
            else:
                out[0] = 1
        
        return outputs



    def forget(self):
        #self.svm = None 
        self.tree = None
        self.n_classes = None


    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
        the classification error cost for each example in the dataset
        """
        outputs = self.use(dataset)
        
        costs = np.ones((len(outputs),1))
        # Compute normalized NLLs
        for xy,pred,cost in zip(dataset,outputs,costs):
            
            print pred
            
            x,y = xy
            if y == pred[0]:
                cost[0] = 0

        return outputs,costs
        
    def test1(self):
        """
        Outputs the result of ``use(dataset)`` and 
        the classification error cost for each example in the dataset
        """
        print 'DONE - test1'
    
        



