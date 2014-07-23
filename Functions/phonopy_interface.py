# Lots of stuff I really don't need!!! (just here for testing)
import numpy as np
#import phonopy.file_IO as file_IO
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
#import eigenvectors as eigen
#import Functions.reading as reading
#import matplotlib.pyplot as plt
import scitools.numpyutils as numpyutils
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.interface.vasp import read_vasp
import copy
import random
#Direct force constants read from file 'FORCE_CONSTANTS' (test, but could be useful)

def eigenvectors_normalization(eigenvector):
    for i in range(eigenvector.shape[0]):
#        eigenvector[i,:] = eigenvector[i,:]/np.sqrt(np.sum(pow(abs(eigenvector[i,:]),2)))
        eigenvector[i,:] = eigenvector[i,:]/np.linalg.norm(eigenvector[i,:])
    return eigenvector


def get_force_constants_from_file (file_name):

    f = open(file_name, 'r')
    # Change according to the system dimensions!!
    force_constants = np.zeros((8,8,3,3))  #needs to be read from somewhere
    f.readline()
    for i in range(8):
        for j in range(8):
            f.readline()
            for x in range(3):
                row = f.readline().split()
                for y in range(len(row)): force_constants[i,j,x,y] = float(row[y])

#   print(force_constants[0,0,:,:])

    return  force_constants

# The only actually (very) important function in this module!!
def obtain_eigenvectors_from_phonopy(structure,q_vector):

#   Needs to be cleaned!!!

#    print('atomic',structure.get_atomic_types())
#   Preparing the bulk type
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions())
    bulk.set_cell(structure.get_cell())

    phonon = Phonopy(bulk,structure.get_super_cell_phonon(),
                     primitive_matrix= structure.get_primitive_matrix(),
                     is_auto_displacements=False)

    phonon.set_displacement_dataset(copy.deepcopy(structure.get_force_set()))
    phonon.produce_force_constants()

########################################################################

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

#    print('Testing Orthogonality (diagonal elements only)')
    if True:
        eigenvectors = eigenvectors_normalization(eigenvectors)
        print('Testing Orthogonality')
        np.set_printoptions(precision=3,suppress=True)
        print(np.dot(eigenvectors.T,np.ma.conjugate(eigenvectors)))
        np.set_printoptions(suppress=False)

#    eigenvectors =np.mat(eigen.orthogonalize(eigenvectors))
#    print([(eigenvectors.T*eigenvectors.conj())[i,i] for i in range(eigenvectors.shape[0])])

    #Arranging eigenvectors by atoms and dimensions
    number_of_dimensions = structure.get_number_of_dimensions()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()

#    print(number_of_dimensions,number_of_primitive_atoms)

    arranged_ev = np.array([[[eigenvectors [i,j*number_of_dimensions+k]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_primitive_atoms)]
                                    for i in range(number_of_primitive_atoms*number_of_dimensions)])

    return arranged_ev, frequencies

'''
#Starting test program

q_vector = np.array([0.0, 0.0, 0.0])

#Reading structure
structure = reading.read_from_file_structure('/home/abel/VASP/Si-test/OUTCAR')
force_constants = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test/vasprun.xml')[0]
#force_constants = get_force_constants_from_file('/home/abel/VASP/Si-test/FORCE_CONSTANTS')

structure.set_force_constants(force_constants)

arranged_EV, frequencies = obtain_eigenvectors_from_phonopy(structure,q_vector)

#Phonopy bands calculation
bands = []
q_start  = np.array([0.5, 0.5, 0.0])
q_end    = np.array([0.0, 0.0, 0.0])
band = []
for i in range(51):
    print(i)
    q_vector = (q_start + (q_end - q_start) / 50 * i)
    frequencies = obtain_eigenvectors_from_phonopy(structure,q_vector)[1]
    print(frequencies)
    band.append(frequencies)

q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.5, 0.0, 0.0])
for i in range(1,51):
    print(i)
    q_vector = (q_start + (q_end - q_start) / 50 * i)
    frequencies = obtain_eigenvectors_from_phonopy(structure,q_vector)[1]
    print(frequencies)
    band.append(frequencies)

plt.plot(band)
plt.show()


#Restore Phonopy eigenvectors structure (for test only)
eigenvectors = arranged_EV.flatten().reshape(arranged_EV.shape[0],arranged_EV.shape[1]*arranged_EV.shape[2])

#Bands definition
bands = []
q_start  = np.array([0.5, 0.5, 0.0])
q_end    = np.array([0.0, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.5, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

#Bands calculation
#phonon.set_band_structure(bands)
#phonon.plot_band_structure().show()

'''