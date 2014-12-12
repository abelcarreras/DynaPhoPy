import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.file_IO import parse_BORN
import copy


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
def obtain_eigenvectors_from_phonopy(structure,q_vector,NAC=False):

#   Preparing the bulk type
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell().T)

    phonon = Phonopy(bulk,structure.get_super_cell_phonon(),
                     primitive_matrix= structure.get_primitive_matrix(),
                     is_auto_displacements=False)

    #Non Analitical Corrections (NAC) from Phonopy  (just for testing MgO)
    if NAC:
        print("Phonopy warning: Using Non Analitical Corrections")
        get_is_symmetry = True  #sfrom phonopy:   settings.get_is_symmetry()
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, get_is_symmetry)
        phonon.set_nac_params(nac_params=nac_params)

    phonon.set_displacement_dataset(copy.deepcopy(structure.get_force_set()))
    phonon.produce_force_constants()

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

    #Making sure for eigenvectors to be orthonormal (can be omitted)
    if True:
        eigenvectors = eigenvectors_normalization(eigenvectors)
        print('Testing eigenvectors orthonormality')
        np.set_printoptions(precision=3,suppress=True)
        print(np.dot(eigenvectors.T,np.ma.conjugate(eigenvectors)).real)
        np.set_printoptions(suppress=False)


    #Arranging eigenvectors by atoms and dimensions
    number_of_dimensions = structure.get_number_of_dimensions()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()


    arranged_ev = np.array([[[eigenvectors [j*number_of_dimensions+k,i]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_primitive_atoms)]
                                    for i in range(number_of_primitive_atoms*number_of_dimensions)])

    return arranged_ev, frequencies

def obtain_phonon_dispersion_spectra(structure,bands_ranges,NAC=False):

    print('Calculating phonon dispersion spectra...')
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell().T)

    phonon = Phonopy(bulk,structure.get_super_cell_phonon(),
                     primitive_matrix= structure.get_primitive_matrix(),
                     is_auto_displacements=False)

    if NAC:
        print("Phonopy warning: Using Non Analitical Corrections")
        print("BORN file is needed to do this")
        get_is_symmetry = True  #sfrom phonopy:   settings.get_is_symmetry()
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, get_is_symmetry)
        phonon.set_nac_params(nac_params=nac_params)


    phonon.set_displacement_dataset(copy.deepcopy(structure.get_force_set()))
    phonon.produce_force_constants()


    band_resolution =30
    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)


    phonon.set_band_structure(bands)

    return phonon.get_band_structure()
