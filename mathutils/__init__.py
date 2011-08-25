"""
The ``mathutils`` package provides modules for different mathematical
functions that are not directly accessible through Numpy. A particular
focus is put on allowing to call these functions such that no memory
allocation is required within. This is useful when one wants to call
the same function several times. All modules rely on C++ code that
must be compiled (see README in mlpython/mathutils/).

This package is divided into the following modules: 

* ``mathutils.linalg``:          Linear algebra operations (dot product, linear system solver, etc.).
* ``mathutils.nonlinear``:       Nonlinear operations (sigmoid, softmax, rectified linear, etc.).

"""
