try:
    from setuptools import setup, Extension
    use_setuptools = True
    print('setuptools is used')
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print('distutils is used')

import numpy

include_dirs_numpy = [numpy.get_include()]


def check_compiler():
    import subprocess
    output = subprocess.Popen(['gcc'], stderr=subprocess.PIPE).communicate()[1]
    if b'clang' in output:
        return 'clang'
    if b'gcc' in output:
        return 'gcc'


def get_version_number():
    for l in open('dynaphopy/__init__.py', 'r').readlines():
        if not(l.find('__version__')):
            exec(l, globals())
            return __version__


if check_compiler() == 'clang':
    correlation = Extension('dynaphopy.power_spectrum.correlation',
                            extra_compile_args=['-std=c99'],
                            include_dirs=include_dirs_numpy,
                            sources=['c/correlation.c'])

    mem = Extension('dynaphopy.power_spectrum.mem',
                    extra_compile_args=['-std=c99'],
                    include_dirs=include_dirs_numpy,
                    sources=['c/mem.c'])

else:
    print ('openmp is used')
    correlation = Extension('dynaphopy.power_spectrum.correlation',
                            extra_compile_args=['-std=c99', '-fopenmp'],
                            extra_link_args=['-lgomp'],
                            include_dirs=include_dirs_numpy,
                            sources=['c/correlation.c'])

    mem = Extension('dynaphopy.power_spectrum.mem',
                    extra_compile_args=['-std=c99', '-fopenmp'],
                    extra_link_args=['-lgomp'],
                    include_dirs=include_dirs_numpy,
                    sources=['c/mem.c'])

displacements = Extension('dynaphopy.displacements',
                          extra_compile_args=['-std=c99'],
                          include_dirs=include_dirs_numpy,
                          sources=['c/displacements.c'])

setup(name='dynaphopy',
      version=get_version_number(),
      description='dynaphopy module',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/DynaPhoPy',
      author_email='abelcarreras83@gmail.com',
      packages=['dynaphopy',
                'dynaphopy.power_spectrum',
                'dynaphopy.analysis',
                'dynaphopy.analysis.fitting',
                'dynaphopy.interface',
                'dynaphopy.interface.iofile'],
      scripts=['scripts/dynaphopy',
               'scripts/concath5',
               'scripts/fitdata',
               'scripts/qha_extract',
               'scripts/rfc_calc'],
      install_requires=['phonopy', 'numpy', 'scipy', 'matplotlib'],
      license='MIT License',
      ext_modules=[correlation, mem, displacements])
