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
