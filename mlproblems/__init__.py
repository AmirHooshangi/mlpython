"""
Before a dataset can be fed to a Learner, it must first be converted
into an MLProblem. 

MLProblem objects are simply iterators with some extra properties.
Hence, from an MLProblem, examples can be obtained by iterating over
the MLProblem. 

MLProblem objects also contain metadata, i.e. "data about the
data". For instance, the metadata could contain information about the
size of the input or the set of all possible values for the
target. The metadata (field ``metadata`` of an MLProblem) is
represented by a dictionary mapping strings to arbitrary objects.  

To create an MLProblem, simply give the raw data over which
iterating should be done and the dictionary containing the
metadata: ::

   >>> from mlpython.mlproblems.generic import MLProblem
   >>> import numpy as np
   >>>
   >>> data = np.arange(30).reshape((10,3)) 
   >>> metadata = {'input_size':3}
   >>> mlpb = MLProblem(data,metadata)
   >>> for example in mlpb:
   ...     print example
   ...
   [0 1 2]
   [3 4 5]
   [6 7 8]
   [ 9 10 11]
   [12 13 14]
   [15 16 17]
   [18 19 20]
   [21 22 23]
   [24 25 26]
   [27 28 29]


Each Learners will require a specific structure within each example. For instance,
a supervised learning algorithms will require the examples decompose into two 
parts, an input and a target::

   >>> data = [ (input,np.sum(input)) for input in np.arange(30).reshape((10,3))]
   >>> metadata = {'input_size':3}
   >>> mlpb = MLProblem(data,metadata)
   >>> for input,target in mlpb:
   ...     print input,target
   ...
   [0 1 2] 3
   [3 4 5] 12
   [6 7 8] 21
   [ 9 10 11] 30
   [12 13 14] 39
   [15 16 17] 48
   [18 19 20] 57
   [21 22 23] 66
   [24 25 26] 75
   [27 28 29] 84

Each Learner object will expect a certain structure within the
example, and it is the job of MLProblem to be compatible with the
Learner's expected example structure.

Some MLProblem objects might require that specific metadata keys be
given and might even add some more. Here's an example to illustrate
this concept. Imagine we wish to create a training set for a
classification problem. To do this, we create a ClassificationProblem
object, which requires the metadata ``'targets'`` that corresponds to
the set of values that the target can take.  Moreover, after having
created this new MLProblem, its metadata will now contain a key
``'class_to_id'``. This key will be associated to a dictionary that
maps target symbols to class IDs: ::

   >>> from mlpython.mlproblems.classification import ClassificationProblem
   >>> data = [ (input,str(input<5)) for input in range(10) ]
   >>> metadata = {'targets':set(['False','True'])}
   >>> trainset = ClassificationProblem(data,metadata)
   >>> print trainset.metadata['class_to_id']
   {'False': 1, 'True': 0}

Learners will also expect different kinds of metadata. The user should
look at the MLProblem and Learner class docstrings in order to figure
out which metadata are expected by these objects.

Once the training set has been processed as desired, then
the same processing should be applied to the other
datasets, such as the test set, as follows: ::

   >>> test_data = [ (input,str(input<5)) for input in [-2,-1,10,11] ]
   >>> test_metadata = {'targets':set(['False','True'])}
   >>> testset = trainset.apply_on(test_data,test_metadata)
   >>> print testset.metadata['class_to_id']
   {'False': 1, 'True': 0}

This ensures that all datasets are coherent and share the relevant
metadata, such as the ``class_to_id`` mapping for classification
problems.  

MLProblems can also be combined. For instance, to obtain
a subset of the examples of a classification problem,
a ClassificationProblem object can be fed to a SubsetProblem::

   >>> from mlpython.mlproblems.generic import SubsetProblem
   >>> subsetpb = SubsetProblem(trainset,subset=set(range(0,10,2)))
   >>> for example in subsetpb:
   ...     print example
   ... 
   (0, 0)
   (2, 0)
   (4, 0)
   (6, 1)
   (8, 1)
   >>> print subsetpb.metadata
   {'targets': set(['True', 'False']), 'class_to_id': {'False': 1, 'True': 0}}

The new MLProblem will inherit the metadata from the previous
MLProblem. Additional metadata can also be given explicitly in the
constructor, as before. If there is overlap between the keys of the
previous MLProblem and the metadata explicitly given to the
constructor, the later will have priority.

Finally, an MLProblem has a length, as defined by the output of
``__len__(self)``.

The ``mlproblems`` package is divided into different modules, 
based on the nature of the machine learning problem or task
being implemented. 

The modules are:

* ``mlproblems.generic``:          MLProblems not specific to a particular task.
* ``mlproblems.classification``:   classification MLProblems.
* ``mlproblems.ranking``:          ranking MLProblems.

"""
