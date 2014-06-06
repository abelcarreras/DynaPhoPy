from distutils.core import setup, Extension
setup(name='Correlation', version='1.0',  \
      ext_modules=[Extension('correlation', ['correlation.c'])])
