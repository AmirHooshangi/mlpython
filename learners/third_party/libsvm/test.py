# Copyright 2011 Hugo Larochelle. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

import mlpython.misc.io as mlio
import mlpython.mlproblems.classification as mlpb
import mlpython.learners.third_party.libsvm.classification as mlclassif
import numpy as np

train_data = [([1,0,0,1],'0'),([0,1,1,0],'1')]
train_metadata = {'input_size':4,'targets':set(['0','1'])}
test_data = [([1,0,0,0],'0'),([0,0,1,0],'1')]
test_metadata = {'input_size':4,'targets':set(['0','1'])}

trainset = mlpb.ClassificationProblem(train_data,train_metadata)
trainset.setup()
testset = trainset.apply_on(test_data,test_metadata)

print 'Training SVM'
svm = mlclassif.SVMClassifier(
    kernel='polynomial',
    degree=3,
    gamma=1,
    coef0=0,
    C=1,
    tolerance=0.001,
    cache_size=100,
    shrinking=True,
    output_probabilities=False,
    label_weights = None
    )
svm.train(trainset)
print 'Evaluating on test set'
outputs, costs = svm.test(testset)
print 'Classification error =',np.mean(costs)
print 'LIBSVM installation is working!'
