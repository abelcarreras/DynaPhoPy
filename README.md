[![PyPI version](https://badge.fury.io/py/dynaphopy.svg)](https://pypi.python.org/pypi/dynaphopy)
[![Build Status](https://app.travis-ci.com/abelcarreras/DynaPhoPy.svg)](https://app.travis-ci.com/github/abelcarreras/DynaPhoPy)
[![Coverage Status](https://coveralls.io/repos/github/abelcarreras/DynaPhoPy/badge.svg)](https://coveralls.io/github/abelcarreras/DynaPhoPy)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2017.08.017-blue)](https://doi.org/10.1016/j.cpc.2017.08.017)

DynaPhoPy
=========
Software to calculate crystal microscopic anharmonic properties
from molecular dynamics (MD) using the normal-mode-decomposition technique.
These properties include the phonon frequency shifts and linewidths,
as well as the renormalized force constanst and thermal properties
by using quasiparticle theory. This code includes interfaces for MD
outputs from VASP and LAMMPS. PHONOPY code is used to obtain harmonic
phonon modes.

Online manual: http://abelcarreras.github.io/DynaPhoPy/


Installation instructions
---------------------------------------------------------

1. Requirements
  - Python 2.7.x/3.5 or higher
  - Phonopy 2.0 or higher (https://phonopy.github.io/phonopy)
  - Matplotlib
  - Scipy
  - h5py
  - pyYAML
  - (optional) pyFFTW (http://www.fftw.org/)
  - (optional) cuda_functions (https://github.com/abelcarreras/cuda_functions)

2a. Install from pypi repository
   ```
   pip install dynaphopy --user
   ```

2b. Install from source (requires c compiler)
   - Install requirements from requirements.txt:
   ```
   pip install -r requirements.txt --user
   ```
   - Run setup.py to install dynaphopy
   ```
   python setup.py install --user
   ```

Executing this software
---------------------------------------------------------

1. Command line method
  - execute ***dynaphopy -h*** for detailed description of available options
    ```
    dynaphopy input_file MD_file [Options]
    ```

2. Interactive mode
  - Use -i option from command line method and follow the instructions
    ```
    dynaphopy input_file MD_file -i
    ```
3. Scripting method (as a module)
  - Dynaphopy can be imported as a python module
  - In examples/api_scripts directory an example script is available (script_silicon.py)
  - The comments in the script makes it (hopefully) self explained.

Input files for several materials can be found in the same example/inputs directory.
More information in the online manual at: http://abelcarreras.github.io/DynaPhoPy


Contact info
---------------------------------------------------------
Abel Carreras  
abelcarreras83@gmail.com

Donostia International Physics Center (DIPC)  
Donostia-San Sebastian (Spain)