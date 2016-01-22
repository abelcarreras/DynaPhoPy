import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS, write_FORCE_CONSTANTS
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.units import VaspToTHz
from phonopy.phonon.degeneracy import degenerate_sets

def eigenvectors_normalization(eigenvector):
    for i in range(eigenvector.shape[0]):
        eigenvector[i, :] = eigenvector[i, :]/np.linalg.norm(eigenvector[i, :])
    return eigenvector

def get_force_sets_from_file(file_name='FORCE_SETS'):
    #Just a wrapper to phonopy function
    force_sets = parse_FORCE_SETS(filename=file_name)
    return force_sets

def save_force_constants_to_file(force_constants, filename='FORCE_CONSTANTS'):
    #Just a wrapper to phonopy function
    write_FORCE_CONSTANTS(force_constants, filename=filename)


def get_phonon(structure, NAC=False):

    force_atoms_file = structure.get_force_set()['natom']
    force_atoms_input = np.product(np.diagonal(structure.get_super_cell_phonon()))*structure.get_number_of_atoms()

    if force_atoms_file != force_atoms_input:
        print("Error: FORCE_SETS file does not match with SUPERCELL MATRIX")
        exit()


    #Preparing the bulk type object
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell().T)

    phonon = Phonopy(bulk, structure.get_super_cell_phonon(),
                     primitive_matrix=structure.get_primitive_matrix(),
                     is_auto_displacements=False)

    #Non Analytical Corrections (NAC) from Phonopy [Frequencies only, eigenvectors no affected by this option]
    if NAC:
        print("Phonopy warning: Using Non Analytical Corrections")
        get_is_symmetry = True  #from phonopy:   settings.get_is_symmetry()
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, get_is_symmetry)
        phonon.set_nac_params(nac_params=nac_params)


    phonon.set_displacement_dataset(structure.get_force_set())
    phonon.produce_force_constants(computation_algorithm="svd")


    return phonon



def obtain_eigenvectors_from_phonopy(structure, q_vector, NAC=False):

    force_atoms_file = structure.get_force_set()['natom']
    force_atoms_input = np.product(np.diagonal(structure.get_super_cell_phonon()))*structure.get_number_of_atoms()

    if force_atoms_file != force_atoms_input:
        print("Error: FORCE_SETS file does not match with SUPERCELL MATRIX")
        exit()

    phonon = get_phonon(structure)

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

    print('Eigenvectors')
    print(eigenvectors)

    #Making sure eigenvectors are orthonormal (can be omitted)
    if True:
        eigenvectors = eigenvectors_normalization(eigenvectors)
        print('Testing eigenvectors orthonormality')
        np.set_printoptions(precision=3,suppress=True)
        print(np.dot(eigenvectors.T, np.ma.conjugate(eigenvectors)).real)
        np.set_printoptions(suppress=False)

    #Arranging eigenvectors by atoms and dimensions
    number_of_dimensions = structure.get_number_of_dimensions()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()


    arranged_ev = np.array([[[eigenvectors [j*number_of_dimensions+k, i]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_primitive_atoms)]
                                    for i in range(number_of_primitive_atoms*number_of_dimensions)])

    print("Harmonic frequencies:")
    print(frequencies)

    return arranged_ev, frequencies

def obtain_phonon_dispersion_bands(structure, bands_ranges, NAC=False, band_resolution=30):

    print('Getting phonon dispersion bands')
    phonon = get_phonon(structure, NAC=False)

    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)
    phonon.set_band_structure(bands)

    return phonon.get_band_structure()

def obtain_renormalized_phonon_dispersion_bands(structure, bands_ranges, force_constants, NAC=False, band_resolution=30):

    print('Getting renormalized phonon dispersion bands')
    phonon = get_phonon(structure, NAC=False)
    phonon.set_force_constants(force_constants)

    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)
    phonon.set_band_structure(bands)

    return phonon.get_band_structure()



def get_commensurate_points_info(structure):

    phonon = get_phonon(structure)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()

    phonon.set_qpoints_phonon(com_points, is_eigenvectors=True)

    return com_points, dynmat2fc, phonon

def get_renormalized_force_constants(normalized_frequencies, dynmat2fc, phonon, degenerate=True):

    frequencies, eigenvectors = phonon.get_qpoints_phonon()

    if degenerate:
        normalized_frequencies = get_degenerated_frequencies(frequencies, normalized_frequencies)

    dynmat2fc.set_dynamical_matrices(normalized_frequencies / VaspToTHz, eigenvectors)
    dynmat2fc.run()

    force_constants = dynmat2fc.get_force_constants()

    return force_constants

def get_degenerated_frequencies(frequencies, normalized_frequencies):

    num_phonons = frequencies.shape[1]
    normalized_frequencies_degenerated = np.zeros_like(normalized_frequencies)

    for i, q_frequencies in enumerate(frequencies):
        degenerate_index = degenerate_sets(q_frequencies)
        weight_matrix = get_weights_from_index_list(num_phonons, degenerate_index)

        for j, weight in enumerate(weight_matrix):
            normalized_frequencies_degenerated[i, j]= np.average(normalized_frequencies[i, :], weights=weight)

    return normalized_frequencies_degenerated

def get_weights_from_index_list(size, index_list):

    weight = np.zeros([size, size])
    for i in range(size):
        for group in index_list:
            if i in group:
                for j in group:
                    weight[i, j] = 1

    return weight