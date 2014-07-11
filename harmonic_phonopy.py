__author__ = 'abel'
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
import phonopy.file_IO as file_IO
import numpy as np
import Functions.reading as reading
import Functions.phonopy_interface as phoin
import matplotlib.pyplot as plt
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS

#Starting test program

q_vector = np.array([0.0, 0.0, 0.0])

#Reading structure
structure = reading.read_from_file_structure('/home/abel/VASP/Si-test_petit/OUTCAR')
structure.set_force_set( parse_FORCE_SETS(64,filename='/home/abel/VASP/Si-test_petit/FORCE_SETS'))

#force_constants = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test_petit/vasprun.xml')[0]

#structure.set_super_cell([2,2,2])

structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])
structure.set_super_cell_matrix([[2, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 2]])

print(structure.get_cell())
#print(structure.get_unit_cell())
#structure.set_force_constants(force_constants)

print(structure.get_number_of_primitive_atoms())
print('---------------------')
print(structure.get_number_of_atoms())
print(structure.get_number_of_cell_atoms())

arranged_EV, frequencies = phoin.obtain_eigenvectors_from_phonopy(structure,q_vector)


#Phonopy bands calculation
bands = []
q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.0, 0.5, 0.0])
band = []
for i in range(51):
    print(i)
    q_vector = (q_start + (q_end - q_start) / 50 * i)
    frequencies = phoin.obtain_eigenvectors_from_phonopy(structure,q_vector)[1]
    print(frequencies)
    band.append(frequencies)


q_start  = np.array([0.0, 0.5, 0.5])
q_end    = np.array([0.0, 0.0, 0.0])
for i in range(1,51):
    print(i)
    q_vector = (q_start + (q_end - q_start) / 50 * i)
    frequencies = phoin.obtain_eigenvectors_from_phonopy(structure,q_vector)[1]
    print(frequencies)
    band.append(frequencies)


q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.5, 0.5, 0.5])
for i in range(1,51):
    print(i)
    q_vector = (q_start + (q_end - q_start) / 50 * i)
    frequencies = phoin.obtain_eigenvectors_from_phonopy(structure,q_vector)[1]
    print(frequencies)
    band.append(frequencies)


plt.plot(band)
plt.show()
