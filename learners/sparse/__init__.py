"""
The ``learners.sparse`` package containts Learners meant for sparse data. By sparse data, we
mean data where the non-zero inputs are not given explicitly.  

The MLProblems for these Learners should have inputs which decompose into
two parts, the input variable values and the corresponding indices 
of those non-zero variables. For instance, a non-sparse input vector
``[0,-1,0,0,0.5]`` would correspond to the pair ``([-1,0.5],[1,4])``
in a sparse format.

This package decomposes into different modules, based on the type of task
the Learners are solving. Currently, the available modules are:

* ``learners.sparse.classification``: classification Learners on sparse inputs.

"""
