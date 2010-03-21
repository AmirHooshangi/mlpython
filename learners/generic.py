

class Learner:
    """
    Base class for a learning algorithm.

    This class is meant to standardize the creation of learners.

    """

    #def __init__():
    
    def train(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def forget(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def use(self):
        raise NotImplementedError("Subclass should have implemented this method.")

    def test(self):
        raise NotImplementedError("Subclass should have implemented this method.")


class OnlineLearner(Learner):
    """
    A learner that is trained online.

    This class makes it easier to construct a learner. All
    that must be defined are four following methods:

    - initialize_learner(self,metadata)
    - update_learner(self,example)
    - use_learner(self,example)
    - cost(self,output,example)

    Funciton initialize_learner(...) simply initializes
    the learner. The training set's 'metadata' is also available.

    Function update_learner(...) updates the learner's parameters
    using the given 'example'.

    Function use_learner(...) should return the output
    for the given 'example'. The output should be a sequence
    (even if it has just one element in it), to allow
    for multiple outputs.

    Function cost(...) should return the cost associated
    to some 'output' for the given 'example'. The returned 
    cost should be a sequence (even if it has just one 
    element in it), to allow for multiple costs.

    Option n_stages specifies how many iterations over the 
    training set is done at every call of train(...).

    All other hyper-parameters for the learner supplied
    through the constructor will be assigned as attributes
    to the object, and hence will be accessible
    by all methods.

    Example of methods for a linear perceptron.

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

    When creating an instance, must provide the
    value of the hyper-parameter lr:

    perceptron = Perceptron(1,lr=0.01)

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
        self.stage = it+1

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
