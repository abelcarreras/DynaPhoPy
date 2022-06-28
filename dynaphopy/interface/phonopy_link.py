import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS, write_FORCE_CONSTANTS, parse_FORCE_CONSTANTS
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.harmonic.force_constants import set_tensor_symmetry_PJ
from phonopy.units import VaspToTHz
from phonopy.structure.symmetry import Symmetry

# support old phonopy versions
try:
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:
    from phonopy.structure.atoms import Atoms as PhonopyAtoms


class ForceConstants:
    def __init__(self, force_constants, supercell=None):
        self._force_constants = np.array(force_constants)
        self._supercell = supercell

    def get_array(self):
        return self._force_constants

    def get_supercell(self):
        return self._supercell

    def set_supercell(self, supercell):
        self._supercell = supercell


class ForceSets:
    def __init__(self, force_sets, supercell=None):
        self._forces = force_sets
        self._supercell = supercell

    def get_dict(self):
        return self._forces

    def get_supercell(self):
        return self._supercell

    def set_supercell(self, supercell):
        self._supercell = supercell


def eigenvectors_normalization(eigenvector):
    for i in range(eigenvector.shape[0]):
        eigenvector[i, :] = eigenvector[i, :]/np.linalg.norm(eigenvector[i, :])
    return eigenvector


def get_force_sets_from_file(file_name='FORCE_SETS', fs_supercell=None):
    # Just a wrapper to phonopy function
    force_sets = ForceSets(parse_FORCE_SETS(filename=file_name))

    if fs_supercell is not None:
        force_sets.set_supercell(fs_supercell)
    else:
        print('No force sets supercell defined, set to identity')
        force_sets.set_supercell(np.identity(3))

    return force_sets


def get_force_constants_from_file(file_name='FORCE_CONSTANTS', fc_supercell=None):
    # Just a wrapper to phonopy function
    force_constants = ForceConstants(np.array(parse_FORCE_CONSTANTS(filename=file_name)))
    if fc_supercell is not None:
        force_constants.set_supercell(fc_supercell)
    else:
        print('No force sets supercell defined, set to identity')
        force_constants.set_supercell(np.identity(3))

    return force_constants


def save_force_constants_to_file(force_constants, filename='FORCE_CONSTANTS'):
    # Just a wrapper to phonopy function
    write_FORCE_CONSTANTS(force_constants.get_array(), filename=filename)


def get_phonon(structure, NAC=False, setup_forces=True, custom_supercell=None, symprec=1e-5):

    if custom_supercell is None:
        super_cell_phonon = structure.get_supercell_phonon()
    else:
        super_cell_phonon = custom_supercell

    # Preparing the bulk type object
    bulk = PhonopyAtoms(symbols=structure.get_atomic_elements(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell())

    phonon = Phonopy(bulk, super_cell_phonon,
                     primitive_matrix=structure.get_primitive_matrix(),
                     symprec=symprec)

    # Non Analytical Corrections (NAC) from Phonopy [Frequencies only, eigenvectors no affected by this option]

    if setup_forces:
        if structure.get_force_constants() is not None:
            phonon.set_force_constants(structure.get_force_constants().get_array())
        elif structure.get_force_sets() is not None:
            phonon.set_displacement_dataset(structure.get_force_sets().get_dict())
            phonon.produce_force_constants()
            structure.set_force_constants(ForceConstants(phonon.get_force_constants(),
                                                         supercell=structure.get_force_sets().get_supercell()))
        else:
            print('No force sets/constants available!')
            exit()

    if NAC:
        print("Warning: Using Non Analytical Corrections")
        primitive = phonon.get_primitive()
        nac_params = parse_BORN(primitive, is_symmetry=True)
        phonon.set_nac_params(nac_params=nac_params)

    return phonon


def obtain_eigenvectors_and_frequencies(structure, q_vector, test_orthonormal=False, print_data=True):

    phonon = get_phonon(structure)
    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)

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


def obtain_phonopy_dos(structure, mesh=(40, 40, 40), force_constants=None,
                       freq_min=None, freq_max=None, projected_on_atom=-1, NAC=False):

    if force_constants is None:
        phonon = get_phonon(structure,
                            setup_forces=True,
                            custom_supercell=None,
                            NAC=NAC)
    else:
        phonon = get_phonon(structure,
                            setup_forces=False,
                            custom_supercell=force_constants.get_supercell(),
                            NAC=NAC)
        phonon.set_force_constants(force_constants.get_array())

    if projected_on_atom < 0:
        phonon.run_mesh(mesh)
        phonon.run_total_dos(freq_min=freq_min, freq_max=freq_max, use_tetrahedron_method=True)
        total_dos = np.array([phonon.get_total_dos_dict()['frequency_points'],
                              phonon.get_total_dos_dict()['total_dos']])

    else:
        phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
        phonon.run_projected_dos(freq_min=freq_min, freq_max=freq_max)

        if projected_on_atom >= len(phonon.get_projected_dos_dict()['projected_dos']):
            print('No atom type {0}'.format(projected_on_atom))
            exit()

        # total_dos = np.array([phonon.get_partial_DOS()[0], phonon.get_partial_DOS()[1][projected_on_atom]])
        total_dos = np.array([phonon.get_projected_dos_dict()['frequency_points'],
                              phonon.get_projected_dos_dict()['projected_dos'][projected_on_atom]])

    #Normalize to unit cell
    total_dos[1, :] *= float(structure.get_number_of_atoms())/structure.get_number_of_primitive_atoms()
    return total_dos


def obtain_phonopy_thermal_properties(structure, temperature, mesh=(40, 40, 40), force_constants=None, NAC=False):

    if force_constants is None:
        phonon = get_phonon(structure,
                            setup_forces=True,
                            custom_supercell=None,
                            NAC=NAC)
    else:
        phonon = get_phonon(structure,
                            setup_forces=False,
                            custom_supercell=force_constants.get_supercell(),
                            NAC=NAC)
        phonon.set_force_constants(force_constants.get_array())

    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_step=1, t_min=temperature, t_max=temperature)
    # t, free_energy, entropy, cv = np.array(phonon.get_thermal_properties()).T[0]
    thermal_dict = phonon.get_thermal_properties_dict()
    free_energy = thermal_dict['free_energy']
    entropy = thermal_dict['entropy']
    cv = thermal_dict['heat_capacity']

    # Normalize to unit cell
    unit_cell_relation = float(structure.get_number_of_atoms())/structure.get_number_of_primitive_atoms()
    free_energy *= unit_cell_relation
    entropy *= unit_cell_relation
    cv *= unit_cell_relation

    return free_energy, entropy, cv


def obtain_phonopy_group_velocity(structure, q_point, force_constants=None, NAC=False):

    if force_constants is None:
        phonon = get_phonon(structure,
                            setup_forces=True,
                            custom_supercell=None,
                            NAC=NAC)
    else:
        phonon = get_phonon(structure,
                            setup_forces=False,
                            custom_supercell=force_constants.get_supercell(),
                            NAC=NAC)
        phonon.set_force_constants(force_constants.get_array())

    return phonon.get_group_velocity_at_q(q_point)


def obtain_phonopy_mesh_from_force_constants(structure, force_constants, mesh=(40, 40, 40), NAC=False):

    phonon = get_phonon(structure,
                        setup_forces=False,
                        custom_supercell=force_constants.get_supercell(),
                        NAC=NAC)
    phonon.set_force_constants(force_constants.get_array())

    phonon.run_mesh(mesh)
    mesh_dict = phonon.get_mesh_dict()

    return mesh_dict['qpoints'], mesh_dict['weights'], mesh_dict['frequencies']


def obtain_phonon_dispersion_bands(structure, bands_ranges, force_constants=None,
                                   NAC=False, band_resolution=30, band_connection=False):

    if force_constants is not None:
        # print('Getting renormalized phonon dispersion relations')
        phonon = get_phonon(structure, NAC=NAC, setup_forces=False,
                            custom_supercell=force_constants.get_supercell())

        phonon.set_force_constants(force_constants.get_array())
    else:
        # print('Getting phonon dispersion relations')
        phonon = get_phonon(structure, NAC=NAC)

    bands =[]
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution+1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)
    phonon.run_band_structure(bands, is_band_connection=band_connection, with_eigenvectors=True)

    bands_dict = phonon.get_band_structure_dict()

    return (bands_dict['qpoints'],
            bands_dict['distances'],
            bands_dict['frequencies'],
            bands_dict['eigenvectors'])


def get_commensurate_points(structure, fc_supercell):

    phonon = get_phonon(structure, setup_forces=False, custom_supercell=fc_supercell)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)
    com_points = dynmat2fc.get_commensurate_points()

    return com_points


def get_equivalent_q_points_by_symmetry(q_point, structure, symprec=1e-5):

    bulk = PhonopyAtoms(symbols=structure.get_atomic_elements(),
                        scaled_positions=structure.get_scaled_positions(),
                        cell=structure.get_cell())

    tot_points = [list(q_point)]
    for operation_matrix in Symmetry(bulk, symprec=symprec).get_reciprocal_operations():
        operation_matrix_q = np.dot(np.linalg.inv(structure.get_primitive_matrix()), operation_matrix.T)
        operation_matrix_q = np.dot(operation_matrix_q, structure.get_primitive_matrix())

        q_point_test = np.dot(q_point, operation_matrix_q)

        if (q_point_test >= 0).all():
                tot_points.append(list(q_point_test))

    tot_points_unique = [list(x) for x in set(tuple(x) for x in tot_points)]
    return tot_points_unique


def get_renormalized_force_constants(renormalized_frequencies, eigenvectors, structure, fc_supercell, symmetrize=False):

    phonon = get_phonon(structure, setup_forces=False, custom_supercell=fc_supercell)

    primitive = phonon.get_primitive()
    supercell = phonon.get_supercell()

    dynmat2fc = DynmatToForceConstants(primitive, supercell)

    size = structure.get_number_of_dimensions() * structure.get_number_of_primitive_atoms()
    eigenvectors = np.array([eigenvector.reshape(size, size, order='C').T for eigenvector in eigenvectors ])
    renormalized_frequencies = np.array(renormalized_frequencies)

    try:
        dynmat2fc.set_dynamical_matrices(renormalized_frequencies / VaspToTHz, eigenvectors)

    except TypeError:
        frequencies_thz = renormalized_frequencies / VaspToTHz
        eigenvalues = frequencies_thz ** 2 * np.sign(frequencies_thz)
        dynmat2fc.create_dynamical_matrices(eigenvalues=eigenvalues,
                                            eigenvectors=eigenvectors)

    dynmat2fc.run()

    force_constants = ForceConstants(dynmat2fc.get_force_constants(), supercell=fc_supercell)

    # Symmetrize force constants using crystal symmetry
    if symmetrize:
        print('Symmetrizing force constants')
        set_tensor_symmetry_PJ(force_constants.get_array(),
                               phonon.supercell.get_cell(),
                               phonon.supercell.get_scaled_positions(),
                               phonon.symmetry)

    return force_constants


if __name__ == "__main__":

    import dynaphopy.interface.iofile as reading
    input_parameters = reading.read_parameters_from_input_file('/home/abel/VASP/Ag2Cu2O4/MD/input_dynaphopy')
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])
    structure.set_primitive_matrix(input_parameters['_primitive_matrix'])
    # structure.set_supercell_phonon(input_parameters['_supercell_phonon'])
    structure.set_force_set(get_force_sets_from_file(file_name=input_parameters['force_constants_file_name']))
    obtain_phonopy_dos(structure)

