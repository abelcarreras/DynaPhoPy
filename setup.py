from distutils.core import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]

correlation = Extension('DynaPhoPy._correlation',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['Extensions/correlation.c'])

derivative  = Extension('DynaPhoPy._derivative',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['Extensions/derivative.c'])

setup(name='DynaPhoPy',
      version='0.9',
      description='DynaPhoPy extensions',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/DynaPhoPy',
      author_email='abelcarreras83@gmail.com',
#      packages=['Classes','Functions'],            #Enable for full install
      ext_modules=[correlation, derivative])
