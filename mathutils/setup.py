from distutils.core import setup, Extension

module1 = Extension('linalg_',
                    sources = ['linalg_.c'],
                    libraries = ['blas','lapack'])

module2 = Extension('nonlinear_',
                    sources = ['nonlinear_.c'],
                    libraries = [])

setup (name = 'MLPythonMath',
       version = '1.0',
       description = 'Simple interfaces to useful math routines',
       ext_modules = [module1,module2])
