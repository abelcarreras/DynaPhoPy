try:
    from setuptools import setup, Extension
    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print("distutils is used.")


import numpy

include_dirs_numpy = [numpy.get_include()]


def get_version_number():
    __version__ = None
    for l in  open('dynaphopy/__init__.py', 'r').readlines():
        if not(l.find('__version__')):
            exec(l)
            return __version__

correlation = Extension('dynaphopy.power_spectrum.correlation',
                        extra_compile_args=['-std=c99'],
                        include_dirs = include_dirs_numpy,
                        sources=['c/correlation.c'])

mem = Extension('dynaphopy.power_spectrum.mem',
                extra_compile_args=['-std=c99'],
                include_dirs = include_dirs_numpy,
                sources=['c/mem.c'])

displacements = Extension('dynaphopy.displacements',
                extra_compile_args=['-std=c99'],
                include_dirs = include_dirs_numpy,
                sources=['c/displacements.c'])

setup(name='dynaphopy',
      version=get_version_number(),
      description='dynaphopy module',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/DynaPhoPy',
      author_email='abelcarreras83@gmail.com',
      packages=['dynaphopy',
                'dynaphopy.orm',
                'dynaphopy.power_spectrum',
                'dynaphopy.analysis',
                'dynaphopy.analysis.fitting',
                'dynaphopy.interface',
                'dynaphopy.interface.iofile'],
      scripts=['scripts/dynaphopy'],
      requires=['phonopy', 'numpy', 'scipy', 'matplotlib'],
      license='MIT License',
      ext_modules=[correlation, mem, displacements])


exit()