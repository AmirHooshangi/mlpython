from mlpython.learners.generic import Learner
import numpy as np

#### Clasifiers ####

class MultinomialNaiveBayesClassifier(Learner):
    """
    Multinomial Naive Bayes Classifier.

    This simple classifier has been found useful for text classification.
    Each non-zero input feature is treated as indication of the presence
    of a word, and its value is treated as the frequency of that word.

    Options:
    - 'dirichlet_prior_parameter'

    Required metadata: 
    - 'targets'
    - 'input_size'

    Reference: A Comparison of Event Models for Naive Bayes Text Classification
               McCallum and Nigam
               link: http://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf

    """

    def __init__(self, dirichlet_prior_parameter=1):
        self.dirichlet_prior_parameter = dirichlet_prior_parameter

    def train(self, trainset):
        self.input_size = trainset.metadata['input_size']
        self.n_classes = len(trainset.metadata['targets'])

        # Initialize the model
        self.p_w_given_c = np.ones((self.input_size,self.n_classes))*self.dirichlet_prior_parameter
        self.p_c = np.zeros((self.n_classes))

        # Train the model
        for input,target in trainset:
            values,indices = input
            self.p_w_given_c[indices,target] += values
            self.p_c[target] += 1

        # Normalize counts
        self.p_w_given_c /= np.sum(self.p_w_given_c,0)
        self.log_p_w_given_c = np.log(self.p_w_given_c)
        self.p_c /= np.sum(self.p_c)
        self.log_p_c = np.log(self.p_c)

    def forget(self):        
        self.p_w_given_c[:] = 1./self.dirichlet_prior_parameter
        self.p_c[:] = 1./self.n_classes

    def use(self,dataset):
        probs = np.zeros((len(dataset),self.n_classes))
        count = 0
        outputs = []
        for example in dataset:
            values,indices = example[0]
            probs[count,:] = np.dot(values,self.log_p_w_given_c[indices,:])+self.log_p_c
            max_output = np.max(probs[count,:])
            probs[count,:] -= max_output
            probs[count,:] = np.exp(probs[count,:])
            probs[count,:] /= np.sum(probs[count,:])
            pred = np.argmax(probs[count,:])
            outputs += [[pred,probs[count,:]]]
            count += 1
        return outputs

    def test(self,dataset):
        outputs = self.use(dataset)
        costs = np.zeros((len(outputs),1))
        count = 0
        for input,target in dataset:
            pred = int(np.argmax(outputs[count,:]))
            costs[count,:] = int(pred!=target)
            count += 1

        return outputs,costs
