import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS, write_FORCE_CONSTANTS
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.harmonic.force_constants import set_tensor_symmetry_PJ
from phonopy.units import VaspToTHz


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


def get_phonon(structure, NAC=False, set_force=True):

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

    if set_force:
        phonon.set_displacement_dataset(structure.get_force_set())
        phonon.produce_force_constants(computation_algorithm="svd")


    return phonon


def obtain_eigenvectors_from_phonopy(structure, q_vector, NAC=False, test_orthonormal=False):

    phonon = get_phonon(structure)

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)


    if False:
        print('Eigenvectors')
        print(eigenvectors)

    #Making sure eigenvectors are orthonormal (can be omitted)
    if test_orthonormal:
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
    phonon = get_phonon(structure, NAC=False, set_force=False)

    phonon.set_force_constants(force_constants)

    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)
    phonon.set_band_structure(bands)

    return phonon.get_band_structure()


def get_commensurate_points(structure):

    phonon = get_phonon(structure, set_force=False)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()

    return com_points


def get_equivalent_q_points_by_symmetry(q_point, structure):

    from phonopy.structure.symmetry import Symmetry
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell().T)

    tot_points = []
    for operation_matrix in Symmetry(bulk).get_reciprocal_operations():
        operation_matrix_q = np.dot(np.linalg.inv(structure.get_primitive_matrix()), operation_matrix.T)
        operation_matrix_q = np.dot(operation_matrix_q, structure.get_primitive_matrix().T)

        q_point_test = np.dot(q_point, operation_matrix_q)

        if (q_point_test >= 0).all():
                tot_points.append(q_point_test)

#    print tot_points
#    print(np.vstack({tuple(row) for row in tot_points}))

    return np.vstack({tuple(row) for row in tot_points})


def get_renormalized_force_constants(renormalized_frequencies, com_points, structure, symmetrize=False):

    phonon = get_phonon(structure)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)

    phonon.set_qpoints_phonon(com_points, is_eigenvectors=True)
    frequencies, eigenvectors = phonon.get_qpoints_phonon()


    dynmat2fc.set_dynamical_matrices(renormalized_frequencies / VaspToTHz, eigenvectors)
    dynmat2fc.run()

    force_constants = dynmat2fc.get_force_constants()

    # Symmetrize force constants using crystal symmetry
    if symmetrize:
        print('Symmetrizing force constants')
        set_tensor_symmetry_PJ(force_constants,
                               phonon.supercell.get_cell().T,
                               phonon.supercell.get_scaled_positions(),
                               phonon.symmetry)

    return force_constants
