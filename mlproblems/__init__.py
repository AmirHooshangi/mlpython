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
