[![PyPI version](https://badge.fury.io/py/dynaphopy.svg)](https://pypi.python.org/pypi/dynaphopy)
[![Build Status](https://travis-ci.org/abelcarreras/DynaPhoPy.svg)](https://travis-ci.org/abelcarreras/DynaPhoPy)
[![Coverage Status](https://coveralls.io/repos/github/abelcarreras/DynaPhoPy/badge.svg)](https://coveralls.io/github/abelcarreras/DynaPhoPy)

DynaPhoPy
=========
Software to calculate crystal microscopic anharmonic properties
from molecular dynamics (MD) using the normal-mode-decomposition technique.
These properties include the phonon frequency shifts and linewidths, 
as well as the renormalized force constanst and thermal properties 
by using quasiparticle theory. This code includes interfaces for MD 
outputs from VASP and LAMMPS .PHONOPY code is used to obtain harmonic
phonon modes.

Installation instructions
---------------------------------------------------------

1. Requirements
  - Python 2.7/3.4 or higher
  - Phonopy 1.9.6 or higher (http://phonopy.sourceforge.net)
  - Matplotlib
  - Scipy
  - h5py
  - pyYAML
  - (optional) pyFFTW (http://www.fftw.org/)
  - (optional) cuda_functions (https://github.com/abelcarreras/cuda_functions)

2. Download the source code from GitHub (https://github.com/abelcarreras/DynaPhoPy/) 
   and place it in the installation directory

3. Install requirements manually or using pip:
   ```
   pip install -r requirements.txt --user
   ```
4. Run setup.py to install dynaphopy
   ```
   python setup.py install --user
   ```
* NEW! Now you can use pip to install/update dynaphopy module
   ```
   pip install dynaphopy --user
   ```
* NEW! Added compatibility with python 3.4

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

Files and directories included in DynaPhoPy distribution
--------------------------------------------------------

~~~
  README             this file 
  LICENSE            the MIT license 
  setup.py           installation script
  requirements.txt   list of required packages

  /dynaphopy          main code in python
  /c                  additional functions in c
  /doc                documentation
  /examples           simple examples
  /script             executable scripts to run dynaphopy in command line
  /unittest           unit tests for checking the integrity of the code
~~~~

Contact info
---------------------------------------------------------
Abel Carreras
<br>abelcarreras83@gmail.com

Donostia International Physics Center (DIPC)
<br>Donostia-San Sebastian (Spain)