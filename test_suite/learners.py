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

import mlpython.learners.generic as learners
import mlpython.mlproblems.generic as mlpb
import numpy as np

class Perceptron(learners.OnlineLearner):

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
      return np.sign(np.dot(self.w,example[0])+self.b)
   
   def cost(self,output,example):
      return int(output != 2*example[1]-1)

inputs = np.array([[0,0,1,1],[1,1,0,0]])
targets = np.array([0,1]).T
metadata = {'input_size':4,'targets':set([0,1])}
trainset = mlpb.MLProblem(zip(inputs,targets),metadata)

perceptron = Perceptron(2,lr=0.01)

perceptron.train(trainset)
print perceptron.use(trainset)
print perceptron.test(trainset)

inputs = np.array([[0,0,0,1],[1,0,0,0]])
targets = np.array([0,1]).T
metadata = {'input_size':4,'targets':set([0,1])}
testset = mlpb.MLProblem(zip(inputs,targets),metadata)
print perceptron.use(testset)
print perceptron.test(testset)
