from distutils.core import setup, Extension

module1 = Extension('linalg_',
                    sources = ['linalg_.c'],
                    libraries = ['blas','lapack'],
                    include_dirs = ['/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/numpy/core/include/numpy'])
                    # include_dirs = ['/opt/lisa/os/lib/python2.5/site-packages/numpy/core/include/numpy'])

module2 = Extension('nonlinear_',
                    sources = ['nonlinear_.c'],
                    libraries = [],
                    include_dirs = ['/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/numpy/core/include/numpy'])
                    # include_dirs = ['/opt/lisa/os/lib/python2.5/site-packages/numpy/core/include/numpy'])

setup (name = 'MLPythonMath',
       version = '1.0',
       description = 'Simple interfaces to useful math routines',
       ext_modules = [module1,module2])
