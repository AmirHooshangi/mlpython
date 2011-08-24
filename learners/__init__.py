"""
MLPython supports many types of learning algorithms or "Learners".
A Learner object will always define the four following methods:

* ``train(self,trainset)``: runs the learning algorithm on ``trainset``.
* ``forget(self)``: resets the Learner to it's original state.
* ``use(self,dataset)``: computes and returns the output of the Learner for ``dataset``. 
  The method should return an iterator over these outputs.
* ``test(self,dataset)``: computes and returns the outputs of the Learner as well as the cost of 
  those outputs for ``dataset``. The method should return a pair of two iterators, the first
  being over the outputs and the second over the costs.

Of course, a constructor ``__init__(self,...)`` also needs to be defined, taking as argument
the different options or "hyper-parameters" this learning algorithm requires.

The ``learners`` package is divided into different modules or
subpackages, based on the task the associated Learners are trying to
solve of the type of data they require. 

The modules are:

* ``learners.generic``:          Learners not specific to a particular task or type of data.
* ``learners.classification``:   Learners for classification problems.
* ``learners.density``:          Learners for density or distrubtion estimation.
* ``learners.dynamic``:          Learners for sequential data.

The subpackages are:

* ``learners.sparse``:           learners for data in sparse format.
* ``learners.third_party``:      learners based on third-party libraries.
* ``learners.gpu``:              learners running on GPUs.

"""
