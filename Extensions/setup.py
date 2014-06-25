from distutils.core import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]
setup(name='Correlation', version='1.0',
      ext_modules=[Extension('correlation',
                             extra_compile_args=['-std=c99'],
                             include_dirs = include_dirs_numpy,
                             sources=['correlation.c'])])

setup(name='Derivative', version='1.0',
      ext_modules=[Extension('derivative',
                             extra_compile_args=['-std=c99'],
                             include_dirs = include_dirs_numpy,
                             sources=['derivative.c'])])



"""
correlation = Extension('DynaPhoPy._correlation',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['correlation.c'])

derivative  = Extension('DynaPhoPy._derivative',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['derivative.c'])

setup(name='DynaPhoPy',
      version='0.1',
      description='DynaPhoPy extensions',
      author='Abel Carreras',
      author_email='abelcarreras83@gmail.com',
      xt_modules=[correlation, derivative])

"""
