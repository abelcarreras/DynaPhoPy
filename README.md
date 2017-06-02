[![PyPI version](https://badge.fury.io/py/dynaphopy.svg)](https://pypi.python.org/pypi/dynaphopy)
[![Build Status](https://travis-ci.org/abelcarreras/DynaPhoPy.svg?branch=development)](https://travis-ci.org/abelcarreras/DynaPhoPy)


DynaPhoPy
=========
Software to calculate crystal microscopical anharmonic properties
from molecular dynamics using the normal-mode-decomposition technique.
VASP or LAMMPS codes can be used to calculate MD. PHONOPY code
is used to obtain harmonic phonon modes.

Installation instructions
---------------------------------------------------------

1. Requirements
  - Python 2.7 or higher
  - Phonopy 1.9.6 or higher (http://phonopy.sourceforge.net)
  - Matplotlib
  - Scipy
  - h5py

2. Download the source code and place it in the installation directory

3. (optional) Install requirements using pip
   <br>pip install -r requirements.txt --user

4. Run setup.py to install dynaphopy
  <br>python setup.py install --user

* NEW! Now you can use pip to install/update dynaphopy module
   <br>pip install dynaphopy --user


Executing this software
---------------------------------------------------------

1. Command line method
  - $dynaphopy input_file MD_file [Options]
  - execute $dynaphopy -h for detailed description of available options

2. Interactive mode
  - Use -i option from command line method and follow the instructions
  - $dynaphopy input_file MD_file -i

3. Scripting method (as a module)
  - Dynaphopy can be imported as a python module
  - In Example directory an example script is available (script_silicon.py)
  - The comments in the script makes it (hopefully) self explained.

Input files for several materials can be found in the same Example directory
More information at: http://abelcarreras.github.io/DynaPhoPy


Contact info
---------------------------------------------------------
Abel Carreras
<br>abelcarreras83@gmail.com

Department of Materials Science and Engineering
<br>Kyoto University