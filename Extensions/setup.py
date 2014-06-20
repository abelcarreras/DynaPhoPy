from distutils.core import setup, Extension

setup(name='Correlation', version='1.0',
      ext_modules=[Extension('correlation',
                             extra_compile_args=['-std=c99'],
                             include_dirs = ['/Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/Python.framework/Versions/2.6/Extras/lib/python/numpy/core/include',
                                             '/usr/include/python2.7'],
                             sources=['correlation.c'])])

setup(name='Derivative', version='1.0',
      ext_modules=[Extension('derivative',
                             extra_compile_args=['-std=c99'],
                             include_dirs = ['/Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/Python.framework/Versions/2.6/Extras/lib/python/numpy/core/include',
                                             '/usr/include/python2.7'],
                             sources=['derivative.c'])])
