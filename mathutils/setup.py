from distutils.core import setup, Extension

module1 = Extension('linalg_',
                    sources = ['linalg_.c'],
                    libraries = ['blas','lapack'],
                    include_dirs = ['/opt/lisa/os/lib/python2.5/site-packages/numpy/core/include/numpy'])

setup (name = 'MLPythonMath',
       version = '1.0',
       description = 'Simple interfaces to useful math routines',
       ext_modules = [module1])
