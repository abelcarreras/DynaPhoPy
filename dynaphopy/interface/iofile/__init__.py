import mmap
import os
import numpy as np

import dynaphopy.dynamics as dyn
import dynaphopy.atoms as atomtest
from dynaphopy.interface import phonopy_link as pho_interface


def diff_matrix(array_1, array_2, cell_size):
    """
    :param array_1: supercell scaled positions respect unit cell
    :param array_2: supercell scaled positions respect unit cell
    :param cell_size: diference between arrays accounting for periodicity
    :return:
    """
    array_1_norm = np.array(array_1) / np.array(cell_size, dtype=float)[None,:]
    array_2_norm = np.array(array_2) / np.array(cell_size, dtype=float)[None,:]

    return array_2_norm - array_1_norm


def check_atoms_order(filename, trajectory_reading_function, structure):

    trajectory = trajectory_reading_function(filename,
                                             structure=structure,
                                             initial_cut=0,
                                             end_cut=1
                                             )

    # For now average_positions() depends on order of atoms so can't used for this at this time
    # In future however this should work
    # reference = trajectory.average_positions()

    # Only using first step
    reference = trajectory.trajectory[0]

    template = get_correct_arrangement(reference, structure)

    return template


def get_correct_arrangement(reference, structure):

    # print structure.get_scaled_positions()
    scaled_coordinates = []
    for coordinate in reference:
        trans = np.dot(coordinate, np.linalg.inv(structure.get_cell()))
        #print coordinate.real, trans.real
        scaled_coordinates.append(np.array(trans.real, dtype=float))

    number_of_cell_atoms = structure.get_number_of_atoms()
    number_of_supercell_atoms = len(scaled_coordinates)
    supercell_dim = np.array(np.round(np.max(scaled_coordinates, axis=0)), dtype=int)

    unit_cell_scaled_coordinates = scaled_coordinates - np.array(scaled_coordinates, dtype=int)

    atom_unit_cell_index = []
    for coordinate in unit_cell_scaled_coordinates:
        # Only works for non symmetric cell (must be changed)

        diff = np.abs(np.array([coordinate]*number_of_cell_atoms) - structure.get_scaled_positions())

        diff[diff >= 0.5] -= 1.0
        diff[diff < -0.5] += 1.0

        # print 'diff', diff
        # print 'postions', structure.get_scaled_positions()
        index = np.argmin(np.linalg.norm(diff, axis=1))

        # print 'test', coordinate, index
        atom_unit_cell_index.append(index)
    atom_unit_cell_index = np.array(atom_unit_cell_index)
    # np.savetxt('index.txt', np.sort(atom_unit_cell_index))

    # np.savetxt('test.txt', unit_coordinates)
    # np.savetxt('test2.txt', np.array([type_0(j, cell_size, number_of_cell_atoms)[:3] for j in range(number_of_supercell_atoms)]))

    # print supercell_dim, number_of_supercell_atoms
    original_conf = np.array([dynaphopy_order(j, supercell_dim)[:3] for j in range(number_of_supercell_atoms)])

    # np.savetxt('original.txt', original_conf)
    # np.savetxt('unitcoor.txt', scaled_coordinates)

    # print np.array(scaled_coordinates).shape
    # print original_conf.shape

    template = []
    lp_coordinates = []
    for i, coordinate in enumerate(scaled_coordinates):
        lattice_points_coordinates = coordinate - structure.get_scaled_positions()[atom_unit_cell_index[i]]
        # print 'c', i, coordinate, coordinate2

        for k in range(3):
            if lattice_points_coordinates[k] > supercell_dim[k] - 0.5:
                lattice_points_coordinates[k] = lattice_points_coordinates[k] - supercell_dim[k]
            if lattice_points_coordinates[k] < -0.5:
                lattice_points_coordinates[k] = lattice_points_coordinates[k] + supercell_dim[k]

        comparison_cell = np.array([lattice_points_coordinates]*number_of_supercell_atoms)
        diference = np.linalg.norm(diff_matrix(original_conf, comparison_cell, supercell_dim), axis=1)
        template.append(np.argmin(diference) + atom_unit_cell_index[i]*number_of_supercell_atoms/number_of_cell_atoms)

        lp_coordinates.append(lattice_points_coordinates)
    template = np.array(template)
    # lp_coordinates = np.array(lp_coordinates)
    # print original_conf.shape, lp_coordinates.shape, template.shape
    # np.savetxt('index2.txt', np.sort(template))
    # np.savetxt('index_tot.txt', np.sort(template*number_of_cell_atoms + atom_unit_cell_index))

    # inv_template = inverse_template(template)
    # inv_template = np.argsort(template)

    # dm = diff_matrix(original_conf, lp_coordinates[inv_template], supercell_dim)
    # dm = diff_matrix(original_conf[template], lp_coordinates, supercell_dim)

    # np.savetxt('template.txt', template)

    # np.savetxt('lp.txt', lp_coordinates[inv_template])
    # np.savetxt('diff.txt', dm)

    if len(np.unique(template)) < len(template):
        print ('template failed, auto-order will not be applied')
        print ('unique: {} / {}'.format(len(np.unique(template)), len(template)))
        return range(len(template))

    return template


def dynaphopy_order(i, size):
    x = np.mod(i, size[0])
    y = np.mod(i, size[0]*size[1])//size[0]
    z = np.mod(i, size[0]*size[1]*size[2])//(size[1]*size[0])
    k = i//(size[1]*size[0]*size[2])

    return np.array([x, y, z, k])


def get_trajectory_parser(file_name, bytes_to_check=1000000):
    from dynaphopy.interface.iofile import trajectory_parsers as tp

    parsers_keywords = {'vasp_outcar': {'function': tp.read_vasp_trajectory,
                                        'keywords': ['NIONS', 'POMASS', 'direct lattice vectors']},
                        'lammps_dump': {'function': tp.read_lammps_trajectory,
                                        'keywords': ['ITEM: TIMESTEP', 'ITEM: NUMBER OF ATOMS', 'ITEM: BOX BOUNDS']},
                        'vasp_xdatcar': {'function': tp.read_VASP_XDATCAR,
                                         'keywords': ['Direct configuration', 'Direct configuration', '=']}}

    # Check file exists
    if not os.path.isfile(file_name):
        print (file_name + ' file does not exist')
        exit()

    file_size = os.stat(file_name).st_size

    # Check available parsers
    for parser in parsers_keywords.values():
        with open(file_name, "r+b") as f:
            file_map = mmap.mmap(f.fileno(), np.min([bytes_to_check, file_size]))
            num_test = [file_map.find(keyword.encode()) for keyword in list(parser['keywords'])]

        if not -1 in num_test:
            return parser['function']

    return None


def read_from_file_structure_outcar(file_name):

    # Check file exists
    if not os.path.isfile(file_name):
        print('Structure file does not exist!')
        exit()

    # Read from VASP OUTCAR file
    print('Reading VASP structure')

    with open(file_name, "r+b") as f:
        # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)

        # Setting number of dimensions
        number_of_dimensions = 3

        # trash reading for guessing primitive cell (Not stable)
        if False:
            # Reading primitive cell (not sure about this, by default disabled)
            position_number = file_map.find(b'PRICEL')
            file_map.seek(position_number)
            position_number = file_map.find(b'A1')
            file_map.seek(position_number)

            primitive_cell = []    #Primitive Cell
            for i in range (number_of_dimensions):
                primitive_cell.append(file_map.readline()
                                          .replace(",", "")
                                          .replace(")", "")
                                          .replace(")","")
                                          .split()[3:number_of_dimensions+3])
            primitive_cell = np.array(primitive_cell,dtype="double")

        # Reading number of atoms
        position_number = file_map.find(b'NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())

        # Reading atoms per type
        position_number = file_map.find(b'ions per type')
        file_map.seek(position_number+15)
        atoms_per_type = np.array(file_map.readline().split(),dtype=int)

        # Reading atoms  mass
        position_number = file_map.find(b'POMASS =')
        atomic_mass_per_type = []
        for i in range(atoms_per_type.shape[0]):
            file_map.seek(position_number+9+6*i)
            atomic_mass_per_type.append(file_map.read(6))
        atomic_mass = sum([[atomic_mass_per_type[j]
                            for i in range(atoms_per_type[j])]
                           for j in range(atoms_per_type.shape[0])],[])
        atomic_mass = np.array(atomic_mass,dtype='double')

        # Reading cell
        position_number = file_map.find(b'direct lattice vectors')
        file_map.seek(position_number)
        file_map.readline()
        direct_cell = []    #Direct Cell
        for i in range (number_of_dimensions):
            direct_cell.append(file_map.readline().split()[0:number_of_dimensions])
        direct_cell = np.array(direct_cell,dtype='double')

        file_map.seek(position_number)
        file_map.readline()

        reciprocal_cell = []    #Reciprocal cell
        for i in range (number_of_dimensions):
            reciprocal_cell.append(file_map.readline().split()[number_of_dimensions:number_of_dimensions*2])
        reciprocal_cell = np.array(reciprocal_cell,dtype='double')

        # Reading positions fractional cartesian
        position_number=file_map.find(b'position of ions in fractional coordinates')
        file_map.seek(position_number)
        file_map.readline()

        positions_fractional = []
        for i in range (number_of_atoms):
            positions_fractional.append(file_map.readline().split()[0:number_of_dimensions])
        positions_fractional = np.array(positions_fractional,dtype='double')

        # Reading positions cartesian
        position_number=file_map.find(b'position of ions in cartesian coordinates')
        file_map.seek(position_number)
        file_map.readline()

        positions = []
        for i in range (number_of_atoms):
            positions.append(file_map.readline().split()[0:3])
        positions = np.array(positions,dtype='double')

    file_map.close()

    return atomtest.Structure(cell= direct_cell,
                              positions=positions,
                              masses=atomic_mass,
                              )


def read_from_file_structure_poscar(file_name, number_of_dimensions=3):
    # Check file exists
    if not os.path.isfile(file_name):
        print('Structure file does not exist!')
        exit()

    # Read from VASP POSCAR file
    print("Reading VASP POSCAR structure")
    poscar_file = open(file_name, 'r')
    data_lines = poscar_file.read().split('\n')
    poscar_file.close()

    multiply = float(data_lines[1])
    direct_cell = np.array([data_lines[i].split()
                            for i in range(2, 2+number_of_dimensions)], dtype=float)
    direct_cell *= multiply
    scaled_positions = None
    positions = None

    try:
        number_of_types = np.array(data_lines[3+number_of_dimensions].split(),dtype=int)

        coordinates_type = data_lines[4+number_of_dimensions][0]
        if coordinates_type == 'D' or coordinates_type == 'd' :

            scaled_positions = np.array([data_lines[8+k].split()[0:3]
                                         for k in range(np.sum(number_of_types))],dtype=float)
        else:
            positions = np.array([data_lines[8+k].split()[0:3]
                                  for k in range(np.sum(number_of_types))],dtype=float)

        atomic_types = []
        for i,j in enumerate(data_lines[5].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = [item for sublist in atomic_types for item in sublist]
        # atomic_types = np.array(atomic_types).flatten().tolist()

    # Old style POSCAR format
    except ValueError:
        print ("Reading old style POSCAR")
        number_of_types = np.array(data_lines[5].split(), dtype=int)
        coordinates_type = data_lines[6][0]
        if coordinates_type == 'D' or coordinates_type == 'd':
            scaled_positions = np.array([data_lines[7+k].split()[0:3]
                                         for k in range(np.sum(number_of_types))], dtype=float)
        else:
            positions = np.array([data_lines[7+k].split()[0:3]
                                  for k in range(np.sum(number_of_types))], dtype=float)

        atomic_types = []
        for i,j in enumerate(data_lines[0].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = [item for sublist in atomic_types for item in sublist]
        # atomic_types = np.array(atomic_types).flatten().tolist()

    return atomtest.Structure(cell=direct_cell,  # cell_matrix, lattice vectors in rows
                              scaled_positions=scaled_positions,
                              positions=positions,
                              atomic_elements=atomic_types,
                              #                              primitive_cell=primitive_cell
                              )


# Just for testing (use with care) Generates a harmonic trajectory using the harmonic eigenvectors.
# All phonon are set to have the same phase defined by phase_0. The aplitude of each phonon mode is
# ajusted for all to have the same energy. This amplitude is given in temperature units assuming that
# phonon energy follows a Maxwell-Boltzmann distribution
def generate_test_trajectory(structure, supercell=(1, 1, 1),
                             minimum_frequency=0.1,  # THz
                             total_time=2,  # picoseconds
                             time_step=0.002,  # picoseconds
                             temperature=400,  # Kelvin
                             silent=False,
                             memmap=False,
                             phase_0=0.0):

    import random
    from dynaphopy.power_spectrum import _progress_bar

    print('Generating ideal harmonic data for testing')
    kb_boltzmann = 0.831446 # u * A^2 / ( ps^2 * K )

    number_of_unit_cells_phonopy = np.prod(np.diag(structure.get_supercell_phonon()))
    number_of_unit_cells = np.prod(supercell)
    # atoms_relation = float(number_of_unit_cells)/ number_of_unit_cells_phonopy

    # Recover dump trajectory from file (test only)
    import pickle
    if False:

        dump_file = open( "trajectory.save", "r" )
        trajectory = pickle.load(dump_file)
        return trajectory

    number_of_atoms = structure.get_number_of_cell_atoms()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()
    number_of_dimensions = structure.get_number_of_dimensions()

    positions = structure.get_positions(supercell=supercell)
    masses = structure.get_masses(supercell=supercell)

    number_of_atoms = number_of_atoms*number_of_unit_cells

    number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms

    atom_type = structure.get_atom_type_index(supercell=supercell)

    # Generate additional wave vectors sample
#    structure.set_supercell_phonon_renormalized(np.diag(supercell))

    q_vector_list = pho_interface.get_commensurate_points(structure, np.diag(supercell))

    q_vector_list_cart = [ np.dot(q_vector, 2*np.pi*np.linalg.inv(structure.get_primitive_cell()).T)
                           for q_vector in q_vector_list]

    atoms_relation = float(len(q_vector_list)*number_of_primitive_atoms)/number_of_atoms

    # Generate frequencies and eigenvectors for the testing wave vector samples
    print('Wave vectors included in test (commensurate points)')
    eigenvectors_r = []
    frequencies_r = []
    for i in range(len(q_vector_list)):
        print(q_vector_list[i])
        eigenvectors, frequencies = pho_interface.obtain_eigenvectors_and_frequencies(structure, q_vector_list[i])
        eigenvectors_r.append(eigenvectors)
        frequencies_r.append(frequencies)
    number_of_frequencies = len(frequencies_r[0])

    # Generating trajectory
    if not silent:
        _progress_bar(0, 'generating')

    # Generating trajectory
    trajectory = []
    for time in np.arange(total_time, step=time_step):
        coordinates = np.array(positions[:, :], dtype=complex)

        for i_freq in range(number_of_frequencies):
            for i_long, q_vector in enumerate(q_vector_list_cart):

                if abs(frequencies_r[i_long][i_freq]) > minimum_frequency: # Prevent error due to small frequencies
                    amplitude = np.sqrt(number_of_dimensions * kb_boltzmann * temperature / number_of_primitive_cells * atoms_relation)/(frequencies_r[i_long][i_freq] * 2 * np.pi) # + random.uniform(-1,1)*0.05
                    normal_mode = amplitude * np.exp(-1j * frequencies_r[i_long][i_freq] * 2.0 * np.pi * time)
                    phase = np.exp(1j * np.dot(q_vector, positions.T) + phase_0)

                    coordinates += (1.0 / np.sqrt(masses)[None].T *
                                   eigenvectors_r[i_long][i_freq, atom_type] *
                                   phase[None].T *
                                   normal_mode).real

        trajectory.append(coordinates)
        if not silent:
            _progress_bar(float(time + time_step) / total_time, 'generating', )

    trajectory = np.array(trajectory)

    time = np.array([i * time_step for i in range(trajectory.shape[0])], dtype=float)
    energy = np.array([number_of_atoms * number_of_dimensions *
                       kb_boltzmann * temperature
                       for i in range(trajectory.shape[0])], dtype=float)

    # Save a trajectory object to file for later recovery (test only)
    if False:
        dump_file = open("trajectory.save", "w")
        pickle.dump(dyn.Dynamics(structure=structure,
                                 trajectory=np.array(trajectory, dtype=complex),
                                 energy=np.array(energy),
                                 time=time,
                                 supercell=np.dot(np.diagflat(supercell), structure.get_cell())),
                    dump_file)

        dump_file.close()

    # structure.set_supercell_phonon_renormalized(None)

    return dyn.Dynamics(structure=structure,
                        trajectory=np.array(trajectory,dtype=complex),
                        energy=np.array(energy),
                        time=time,
                        supercell=np.dot(np.diagflat(supercell), structure.get_cell()),
                        memmap=memmap)


# Testing function
def read_from_file_test():

    print('Reading structure from test file')

    # Test conditions
    number_of_dimensions = 2

    f_coordinates = open('Data Files/test.out', 'r')
    f_velocity = open('Data Files/test2.out', 'r')
    f_trajectory = open('Data Files/test3.out', 'r')

    # Coordinates reading
    positions = []
    while True:
        row = f_coordinates.readline().split()
        if not row: break
        for i in range(len(row)): row[i] = float(row[i])
        positions.append(row)

    atom_type = np.array(positions,dtype=int)[:, 2]
    positions = np.array(positions)[:,:number_of_dimensions]
    print('Coordinates reading complete')

    structure = atomtest.Structure(positions=positions,
                                   atomic_numbers=atom_type,
                                   cell=[[2,0],[0,1]],
                                   masses=[1] * positions.shape[0]) #all 1
    number_of_atoms = structure.get_number_of_atoms()

    structure.set_number_of_primitive_atoms(2)
    print('number of atoms in primitive cell')
    print(structure.get_number_of_primitive_atoms())
    print('number of total atoms in structure (super cell)')
    print(number_of_atoms)

    # Velocity reading section
    velocity = []
    while True:
        row = f_velocity.readline().replace('I','j').replace('*','').replace('^','E').split()
        if not row: break
        for i in range(len(row)): row[i] = complex('('+row[i]+')')
        velocity.append(row)
    # Velocity = velocity[:4000][:]  #Limitate the number of points (just for testing)

    time = np.array([velocity[i][0]  for i in range(len(velocity))]).real
    velocity = np.array([[[velocity[i][j*number_of_dimensions+k+1]
                           for k in range(number_of_dimensions)]
                          for j in range(number_of_atoms)]
                         for i in range (len(velocity))])
    print('Velocity reading complete')

    # Trajectory reading
    trajectory = []
    while True:
        row = f_trajectory.readline().replace('I','j').replace('*','').replace('^','E').split()
        if not row: break
        for i in range(len(row)): row[i] = complex('('+row[i]+')')
        trajectory.append(row)

    trajectory = np.array([[[trajectory[i][j*number_of_dimensions+k+1]
                             for k in range(number_of_dimensions)]
                            for j in range(number_of_atoms)]
                           for i in range (len(trajectory))])

    print('Trajectory reading complete')

    return dyn.Dynamics(trajectory=trajectory,
                        #velocity=velocity,
                        time=time,
                        structure=structure)


def write_curve_to_file(frequency_range, curve_matrix, file_name):
    output_file = open(file_name, 'w')

    for i in range(curve_matrix.shape[0]):
        output_file.write("{0:10.4f}\t".format(frequency_range[i]))
        for j in curve_matrix[i, :]:
            output_file.write("{0:.10e}\t".format(j))
        output_file.write("\n")

    output_file.close()
    return 0


def read_parameters_from_input_file(file_name, number_of_dimensions=3):

    input_parameters = {'structure_file_name_poscar': 'POSCAR'}

    # Check file exists
    if not os.path.isfile(file_name):
        print (file_name + ' file does not exist')
        exit()

    with open(file_name, "r") as f:
        input_file = f.readlines()

    for i, line in enumerate(input_file):
        if line[0] == '#':
            continue

        if "STRUCTURE FILE OUTCAR" in line:
            input_parameters.update({'structure_file_name_outcar': input_file[i+1].replace('\n','').strip()})

        if "STRUCTURE FILE POSCAR" in line:
            input_parameters.update({'structure_file_name_poscar': input_file[i+1].replace('\n','').strip()})

        if "FORCE SETS" in line:
            input_parameters.update({'force_sets_file_name': input_file[i+1].replace('\n','').strip()})

        if "FORCE CONSTANTS" in line:
            input_parameters.update({'force_constants_file_name': input_file[i+1].replace('\n','').strip()})
            # print('Warning!: FORCE CONSTANTS label in input has changed. Please use FORCE SETS instead')
            # exit()

        if "PRIMITIVE MATRIX" in line:
            primitive_matrix = [input_file[i+j+1].replace('\n','').split() for j in range(number_of_dimensions)]
            input_parameters.update({'_primitive_matrix': np.array(primitive_matrix, dtype=float)})

        if "SUPERCELL MATRIX" in line:
            super_cell_matrix = [input_file[i+j+1].replace('\n','').split() for j in range(number_of_dimensions)]

            super_cell_matrix = np.array(super_cell_matrix, dtype=int)
            input_parameters.update({'supercell_phonon': np.array(super_cell_matrix, dtype=int)})

        if "BANDS" in line:
            bands = []
            labels = []
            while i < len(input_file)-1:
                line = input_file[i + 1].replace('\n', '')
                try:
                    labels.append(line.split(':')[1].replace('\n','').split(','))
                    line = line.split(':')[0]
                except:
                    pass
                try:
                    band = np.array(line.replace(',',' ').split(), dtype=float).reshape((2,3))
                except IOError:
                    break
                except ValueError:
                    break
                i += 1
                bands.append(band)
            labels = [(label[0].replace(' ',''), label[1].replace(' ','')) for label in labels]

            if labels != []:
                input_parameters.update({'_band_ranges': {'ranges': bands,
                                                          'labels': labels}})
            else:
                input_parameters.update({'_band_ranges': {'ranges':bands}})

        if "MESH PHONOPY" in line:
            input_parameters.update({'_mesh_phonopy': np.array(input_file[i+1].replace('\n','').split(),dtype=int)})

    return input_parameters


def write_xsf_file(file_name,structure):

    xsf_file = open(file_name,"w")

    xsf_file.write("CRYSTAL\n")
    xsf_file.write("PRIMVEC\n")

    for row in structure.get_primitive_cell():
        xsf_file.write("{0:10.4f}\t{1:10.4f}\t{2:10.4f}\n".format(*row))
    xsf_file.write("CONVVEC\n")

    for row in structure.get_cell():
        xsf_file.write("{0:10.4f}\t{1:10.4f}\t{2:10.4f}\n".format(*row))
    xsf_file.write("PRIMCOORD\n")

    xsf_file.write("{0:10d} {1:10d}\n".format(structure.get_number_of_primitive_atoms(),1))

    counter = 0
    while counter < structure.get_number_of_atom_types():
        for i,value_type in enumerate(structure.get_atom_type_index()):
            if value_type == counter:
                xsf_file.write("{0:4d}\t{1:10.4f}\t{2:10.4f}\t{3:10.4f}\n".format(structure.get_atomic_numbers()[i],
                                                                                  *structure.get_positions()[i]))
                counter += 1
                break
    xsf_file.close()


# Save & load HDF5 data file
def save_data_hdf5(file_name, time, super_cell, trajectory=None, velocity=None, vc=None, reduced_q_vector=None):
    import h5py

    hdf5_file = h5py.File(file_name, "w")

    if trajectory is not None:
        hdf5_file.create_dataset('trajectory', data=trajectory)

    if velocity is not None:
        hdf5_file.create_dataset('velocity', data=velocity)

    if vc is not None:
        hdf5_file.create_dataset('vc', data=vc)

    if reduced_q_vector is not None:
        hdf5_file.create_dataset('reduced_q_vector', data=reduced_q_vector)


    hdf5_file.create_dataset('time', data=time)
    hdf5_file.create_dataset('super_cell', data=super_cell)

    # print("saved", velocity.shape[0], "steps")
    hdf5_file.close()


def initialize_from_hdf5_file(file_name, structure, read_trajectory=True, initial_cut=1, final_cut=None, memmap=False):
    import h5py

    print("Reading data from hdf5 file: " + file_name)

    trajectory = None
    velocity = None
    vc = None
    reduced_q_vector = None

    # Check file exists
    if not os.path.isfile(file_name):
        print(file_name + ' file does not exist!')
        exit()

    hdf5_file = h5py.File(file_name, "r")
    if "trajectory" in hdf5_file and read_trajectory is True:
        trajectory = hdf5_file['trajectory'][:]
        if final_cut is not None:
            trajectory = trajectory[initial_cut-1:final_cut]
        else:
            trajectory = trajectory[initial_cut-1:]

    if "velocity" in hdf5_file:
        velocity = hdf5_file['velocity'][:]
        if final_cut is not None:
            velocity = velocity[initial_cut-1:final_cut]
        else:
            velocity = velocity[initial_cut-1:]

    if "vc" in hdf5_file:
        vc = hdf5_file['vc'][:]
        if final_cut is not None:
            vc = vc[initial_cut-1:final_cut]
        else:
            vc = vc[initial_cut-1:]

    if "reduced_q_vector" in hdf5_file:
        reduced_q_vector = hdf5_file['reduced_q_vector'][:]
        print("Load trajectory projected onto {0}".format(reduced_q_vector))

    time = hdf5_file['time'][:]
    supercell = hdf5_file['super_cell'][:]
    hdf5_file.close()

    if vc is None:
        return dyn.Dynamics(structure=structure,
                            trajectory=trajectory,
                            velocity=velocity,
                            time=time,
                            supercell=np.dot(np.diagflat(supercell), structure.get_cell()),
                            memmap=memmap)
    else:
        return vc, reduced_q_vector, dyn.Dynamics(structure=structure,
                                                  time=time,
                                                  supercell=np.dot(np.diagflat(supercell), structure.get_cell()),
                                                  memmap=memmap)


def save_quasiparticle_data_to_file(quasiparticle_data, filename):

    import yaml

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    yaml.add_representer(float, float_representer)

    output_dict = []
    for i, q_point in enumerate(quasiparticle_data['q_points']):
        q_point_dict = {'reduced_wave_vector': q_point.tolist()}
        q_point_dict.update({'frequencies': quasiparticle_data['frequencies'][i].tolist()})
        q_point_dict.update({'linewidths': quasiparticle_data['linewidths'][i].tolist()})
        q_point_dict.update({'frequency_shifts': quasiparticle_data['frequency_shifts'][i].tolist()})
        # output_dict.update({'q_point_{}'.format(i): q_point_dict})

        if 'group_velocity' in quasiparticle_data:
            q_point_dict.update({'group_velocity': [{'x': gv[0].tolist(),
                                                     'y': gv[1].tolist(),
                                                     'z': gv[2].tolist()} for gv in quasiparticle_data['group_velocity'][i]]})

        output_dict.append(q_point_dict)

    with open(filename, 'w') as outfile:
        yaml.dump(output_dict, outfile, default_flow_style=False)


def save_mesh_data_to_yaml_file(mesh_data, filename):

    import yaml

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    yaml.add_representer(float, float_representer)

    qpoints, multiplicity, frequencies, linewidths = mesh_data

    output_dict = []
    for i, qp in enumerate(qpoints):
        mesh_dict = {}
        mesh_dict['reduced_wave_vector'] = qp.tolist()
        mesh_dict['frequencies'] = frequencies[i].tolist()
        mesh_dict['linewidths'] = linewidths[i].tolist()
        mesh_dict['multiplicity'] = int(multiplicity[i])

        output_dict.append(mesh_dict)

    with open(filename, 'w') as outfile:
        yaml.dump(output_dict, outfile, default_flow_style=False)


def save_bands_data_to_file(bands_data, filename):
    import yaml

    def float_representer(dumper, value):
        text = '{0:.8f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    yaml.add_representer(float, float_representer)

    with open(filename, 'w') as outfile:
        yaml.dump(bands_data, outfile, default_flow_style=False)
