from generic import Learner
from numpy import dot,ones,zeros,log,argmax,sum

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
        self.p_w_given_c = ones((self.input_size,self.n_classes))*self.dirichlet_prior_parameter
        self.p_c = zeros((self.n_classes))

        # Train the model
        for input,target in trainset:
            values,indices = input
            self.p_w_given_c[indices,target] += values
            self.p_c[target] += 1

        # Normalize counts
        self.p_w_given_c /= sum(self.p_w_given_c,0)
        self.log_p_w_given_c = log(self.p_w_given_c)
        self.p_c /= sum(self.p_c)
        self.log_p_c = log(self.p_c)

    def forget(self):        
        self.p_w_given_c[:] = 1./self.dirichlet_prior_parameter
        self.p_c[:] = 1./self.n_classes

    def use(self,dataset):
        outputs = zeros((len(dataset),self.n_classes))
        count = 0
        for example in dataset:
            values,indices = example[0]
            outputs[count,:] = dot(values,self.log_p_w_given_c[indices,:])+self.log_p_c
            max_output = np.max(outputs[count,:])
            outputs[count,:] -= max_output
            outputs[count,:] = np.exp(outputs[count,:])
            outputs[count,:] /= np.sum(outputs[count,:])
            count += 1
        return outputs

    def test(self,dataset):
        outputs = self.use(dataset)
        costs = zeros((len(outputs),1))
        count = 0
        for input,target in dataset:
            pred = int(argmax(outputs[count,:]))
            costs[count,:] = int(pred!=target)
            count += 1

        return outputs,costs
