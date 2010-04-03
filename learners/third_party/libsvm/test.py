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
