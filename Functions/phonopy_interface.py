import numpy as np
#import phonopy.file_IO as file_IO
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
#import eigenvectors as eigen
#import Functions.reading as reading
#import matplotlib.pyplot as plt
import scitools.numpyutils as numpyutils


#Direct force constants read from file 'FORCE_CONSTANTS' (test, but could be useful)
def get_force_constants_from_file (file_name):

    f = open(file_name, 'r')
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


def obtain_eigenvectors_from_phonopy(structure,q_vector):


#   Preparing the bulk type
    bulk = PhonopyAtoms(symbols=structure.atomic_types,scaled_positions=structure.get_scaled_positions())
    bulk.set_cell(structure.get_cell())

#   Preparing the phonon type
    phonon = Phonopy(bulk, [[1,0,0],[0,1,0],[0,0,1]], distance=0.01)
    phonon.set_force_constants(structure.get_force_constants())

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

#    print('Testing Orthogonality (diagonal elements only)')
    eigenvectors =np.mat(numpyutils.Gram_Schmidt(eigenvectors.real,normalize=True))
#    eigenvectors =np.mat(eigen.orthogonalize(eigenvectors))
#    print([(eigenvectors.T*eigenvectors.conj())[i,i] for i in range(eigenvectors.shape[0])])

    #Arranging eigenvectors by atoms and dimensions
    number_of_dimensions = structure.get_number_of_dimensions()
    number_of_cell_atoms = structure.get_number_of_cell_atoms()
    arranged_ev = np.array([[[eigenvectors [i,j*number_of_dimensions+k]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_cell_atoms)]
                                    for i in range(number_of_cell_atoms*number_of_dimensions)])

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