"""
The package ``learners.third_party.libsvm`` contains modules for learning
algorithms using the LIBSVM library. These modules all require that the
LIBSVM library be installed.

To use LIBSVM through mlpython, do the following:

1. download LIBSVM from here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
2. install LIBSVM (see LIBSVM instructions)
3. install the included python interface (see LIBSVM intrusctions)
4. put path to the python interface in PYTHONPATH

That should do it. Try 'python test.py' to see if your installation is working.

Here is an example of what steps 1 to 3 can look like, where LIBSVMDIR
is the path where you wish to install LIBSVM and
PYTHON_INCLUDEDIR is the path of your python include directory
(use at your own risk!): ::

   tcsh
   set LIBSVMDIR=~/python
   set PYTHON_INCLUDEDIR=/usr/include/python2.5
   wget -O $LIBSVMDIR/libsvm-2.9.zip http://www.csie.ntu.edu.tw/~cjlin/libsvm/libsvm-2.9.zip
   cd $LIBSVMDIR/
   unzip libsvm-2.9.zip
   cd libsvm-2.9
   make
   cd python
   make PYTHON_INCLUDEDIR=$PYTHON_INCLUDEDIR all
   exit

Finally, you'll need to add $MLIBSVMDIR/python to your PYTHONPATH.

Currently, ``learner.third_party.libsvm`` contains the following modules:

* ``learning.third_party.libsvm.classification``:    SVM classifier based on the LIBSVM library.

"""
