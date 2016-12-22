import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS, write_FORCE_CONSTANTS, parse_FORCE_CONSTANTS
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.harmonic.force_constants import set_tensor_symmetry_PJ
from phonopy.units import VaspToTHz


def eigenvectors_normalization(eigenvector):
    for i in range(eigenvector.shape[0]):
        eigenvector[i, :] = eigenvector[i, :]/np.linalg.norm(eigenvector[i, :])
    return eigenvector


def get_force_sets_from_file(file_name='FORCE_SETS'):
    # Just a wrapper to phonopy function
    force_sets = parse_FORCE_SETS(filename=file_name)
    return force_sets


def get_force_constants_from_file(file_name='FORCE_CONSTANTS'):
    # Just a wrapper to phonopy function
    force_constants = parse_FORCE_CONSTANTS(filename=file_name)
    return force_constants


def save_force_constants_to_file(force_constants, filename='FORCE_CONSTANTS'):
    # Just a wrapper to phonopy function
    write_FORCE_CONSTANTS(force_constants, filename=filename)


def get_phonon(structure, NAC=False, setup_forces=True, custom_supercell=None):

    super_cell_phonon = structure.get_supercell_phonon()
    if not(isinstance(custom_supercell, type(None))):
        super_cell_phonon = custom_supercell

    #Preparing the bulk type object
    bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell().T)

    phonon = Phonopy(bulk, super_cell_phonon,
                     primitive_matrix=structure.get_primitive_matrix())

    # Non Analytical Corrections (NAC) from Phonopy [Frequencies only, eigenvectors no affected by this option]
    if NAC:
        print("Phonopy warning: Using Non Analytical Corrections")
        get_is_symmetry = True  #from phonopy:   settings.get_is_symmetry()
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, get_is_symmetry)
        phonon.set_nac_params(nac_params=nac_params)

    if setup_forces:
        if not structure.forces_available():
    #    if not np.array(structure.get_force_constants()).any() and not np.array(structure.get_force_sets()).any():
            print('No force sets/constants available!')
            exit()
        if np.array(structure.get_force_constants()).any():
            phonon.set_force_constants(structure.get_force_constants())
        else:
            phonon.set_displacement_dataset(structure.get_force_sets())
            phonon.produce_force_constants(computation_algorithm="svd")

    return phonon


def obtain_eigenvectors_and_frequencies(structure, q_vector, NAC=False, test_orthonormal=False, print_data=True):

    phonon = get_phonon(structure, NAC=NAC)

    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

    if False:
        print('Eigenvectors')
        print(eigenvectors)

    # Making sure eigenvectors are orthonormal (can be omitted)
    if test_orthonormal:
        eigenvectors = eigenvectors_normalization(eigenvectors)
        print('Testing eigenvectors orthonormality')
        np.set_printoptions(precision=3, suppress=True)
        print(np.dot(eigenvectors.T, np.ma.conjugate(eigenvectors)).real)
        np.set_printoptions(suppress=False)

    #Arranging eigenvectors by atoms and dimensions
    number_of_dimensions = structure.get_number_of_dimensions()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()

    arranged_ev = np.array([[[eigenvectors [j*number_of_dimensions+k, i]
                                    for k in range(number_of_dimensions)]
                                    for j in range(number_of_primitive_atoms)]
                                    for i in range(number_of_primitive_atoms*number_of_dimensions)])

    if print_data:
        print("Harmonic frequencies (THz):")
        print(frequencies)

    return arranged_ev, frequencies


def obtain_phonopy_dos(structure, mesh=(40, 40, 40), force_constants=None, freq_min=None, freq_max=None, projected_on_atom=-1):

    if force_constants is None:
        phonon = get_phonon(structure,
                            setup_forces=True,
                            custom_supercell=None)
    else:
        phonon = get_phonon(structure,
                            setup_forces=False,
                            custom_supercell=structure.get_supercell_phonon_renormalized())
        phonon.set_force_constants(force_constants)

    if projected_on_atom < 0:
        phonon.set_mesh(mesh)
        phonon.set_total_DOS(freq_min=freq_min, freq_max=freq_max, tetrahedron_method=True)
        total_dos = np.array(phonon.get_total_DOS())

    else:
        phonon.set_mesh(mesh, is_eigenvectors=True, is_mesh_symmetry=False)
        phonon.set_partial_DOS(freq_min=freq_min, freq_max=freq_max)

        if projected_on_atom >= len(phonon.get_partial_DOS()[1]):
            print('No atom type {0}'.format(projected_on_atom))
            exit()

        total_dos = np.array([phonon.get_partial_DOS()[0], phonon.get_partial_DOS()[1][projected_on_atom]])


    #Normalize to unit cell
    total_dos[1, :] *= float(structure.get_number_of_atoms())/structure.get_number_of_primitive_atoms()
    return total_dos


def obtain_phonopy_thermal_properties(structure, temperature, mesh=(40, 40, 40), force_constants=None):

    if force_constants is None:
        phonon = get_phonon(structure,
                            setup_forces=True,
                            custom_supercell=None)
    else:
        phonon = get_phonon(structure,
                            setup_forces=False,
                            custom_supercell=structure.get_supercell_phonon_renormalized())
        phonon.set_force_constants(force_constants)

    phonon.set_mesh(mesh)
    phonon.set_thermal_properties(t_step=1, t_min=temperature, t_max=temperature)
    t, free_energy, entropy, cv = np.array(phonon.get_thermal_properties()).T[0]

    # Normalize to unit cell
    unit_cell_relation = float(structure.get_number_of_atoms())/structure.get_number_of_primitive_atoms()
    free_energy *= unit_cell_relation
    entropy *= unit_cell_relation
    cv *= unit_cell_relation

    return free_energy, entropy, cv


def obtain_phonon_dispersion_bands(structure, bands_ranges, force_constants=None, NAC=False, band_resolution=30):

    if force_constants is not None:
#        print('Getting renormalized phonon dispersion relations')
        phonon = get_phonon(structure, NAC=False, setup_forces=False,
                            custom_supercell=structure.get_supercell_phonon_renormalized())

        phonon.set_force_constants(force_constants)
    else:
 #       print('Getting phonon dispersion relations')
        phonon = get_phonon(structure, NAC=NAC)

    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)
    phonon.set_band_structure(bands)

    return phonon.get_band_structure()


def get_commensurate_points(structure, custom_supercell=None):

    phonon = get_phonon(structure, setup_forces=False,
                        custom_supercell=custom_supercell)

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
        operation_matrix_q = np.dot(operation_matrix_q, structure.get_primitive_matrix())

        q_point_test = np.dot(q_point, operation_matrix_q)

        if (q_point_test >= 0).all():
                tot_points.append(q_point_test)

#    print tot_points
#    print(np.vstack({tuple(row) for row in tot_points}))

    return np.vstack({tuple(row) for row in tot_points})


def get_renormalized_force_constants(renormalized_frequencies, eigenvectors, structure, symmetrize=False):

    phonon = get_phonon(structure, setup_forces=False, custom_supercell=structure.get_supercell_phonon_renormalized())

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)

    size = structure.get_number_of_dimensions() * structure.get_number_of_primitive_atoms()
    eigenvectors = np.array([eigenvector.reshape(size, size, order='C').T for eigenvector in eigenvectors ])

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


if __name__ == "__main__":

    import dynaphopy.interface.iofile as reading
    input_parameters = reading.read_parameters_from_input_file('/home/abel/VASP/Ag2Cu2O4/MD/input_dynaphopy')
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])
    structure.set_primitive_matrix(input_parameters['_primitive_matrix'])
    structure.set_supercell_phonon(input_parameters['_supercell_phonon'])
    structure.set_force_set(get_force_sets_from_file(file_name=input_parameters['force_constants_file_name']))
    obtain_phonopy_dos(structure)

