from distutils.core import setup, Extension
import numpy

include_dirs_numpy = [numpy.get_include()]

correlation = Extension('dynaphopy.correlation',
                        extra_compile_args=['-std=c99', '-fopenmp'],
                        extra_link_args=['-lgomp'],
                        include_dirs = include_dirs_numpy,
                        sources=['Extensions/correlation.c'])

derivative  = Extension('dynaphopy.derivative',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['Extensions/derivative.c'])

mem = Extension('dynaphopy.mem',
                extra_compile_args=['-std=c99', '-fopenmp'],
                extra_link_args=['-lgomp'],
                include_dirs = include_dirs_numpy,
                sources=['Extensions/mem.c'])


displacements = Extension('dynaphopy.displacements',
                extra_compile_args=['-std=c99', '-fopenmp'],
                extra_link_args=['-lgomp'],
                include_dirs = include_dirs_numpy,
                sources=['Extensions/displacements.c'])


setup(name='dynaphopy',
      version='1.5.1',
      description='dynaphopy module',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/DynaPhoPy',
      author_email='abelcarreras83@gmail.com',
      packages=['dynaphopy',
                'dynaphopy.classes',
                'dynaphopy.analysis',
                'dynaphopy.interface'],
      scripts=['scripts/dynaphopy'],
      ext_modules=[correlation, derivative, mem, displacements])


exit()