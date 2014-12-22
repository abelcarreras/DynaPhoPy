import numpy as np
import mmap
import pickle
import os
import h5py

import dynaphopy.classes.dynamics as dyn
import dynaphopy.classes.atoms as atomtest
import dynaphopy.functions.phonopy_interface as pho_interface


def read_from_file_structure_outcar(file_name):

    #Check file exists
    if not os.path.isfile(file_name):
        print('Structure file does not exist!')
        exit()

    #Read from VASP OUTCAR file
    print("Reading VASP structure")

    with open(file_name, "r+") as f:
        # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)


        #Setting number of dimensions
        number_of_dimensions = 3

        #Test reading for guessing primitive cell (Not stable)
        if False:
           #Reading primitive cell (not sure about this, by default disabled)
            position_number = file_map.find('PRICEL')
            file_map.seek(position_number)
            position_number = file_map.find('A1')
            file_map.seek(position_number)

            primitive_cell = []    #Primitive Cell
            for i in range (number_of_dimensions):
                primitive_cell.append(file_map.readline()
                                          .replace(",", "")
                                          .replace(")", "")
                                          .replace(")","")
                                          .split()[3:number_of_dimensions+3])
            primitive_cell = np.array(primitive_cell,dtype="double").T


        #Reading number of atoms
        position_number = file_map.find('NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())


        #Reading atoms per type
        position_number = file_map.find('ions per type')
        file_map.seek(position_number+15)
        atoms_per_type = np.array(file_map.readline().split(),dtype=int)


        #Reading atoms  mass
        position_number = file_map.find('POMASS =')
        atomic_mass_per_type = []
        for i in range(atoms_per_type.shape[0]):
            file_map.seek(position_number+9+6*i)
            atomic_mass_per_type.append(file_map.read(6))
        atomic_mass = sum([[atomic_mass_per_type[j]
                            for i in range(atoms_per_type[j])]
                           for j in range(atoms_per_type.shape[0])],[])
        atomic_mass = np.array(atomic_mass,dtype='double')


        #Reading cell
        position_number = file_map.find('direct lattice vectors')
        file_map.seek(position_number)
        file_map.readline()
        direct_cell = []    #Direct Cell
        for i in range (number_of_dimensions):
            direct_cell.append(file_map.readline().split()[0:number_of_dimensions])
        direct_cell = np.array(direct_cell,dtype='double').T

        file_map.seek(position_number)
        file_map.readline()

        reciprocal_cell = []    #Reciprocal cell
        for i in range (number_of_dimensions):
            reciprocal_cell.append(file_map.readline().split()[number_of_dimensions:number_of_dimensions*2])
        reciprocal_cell = np.array(reciprocal_cell,dtype='double').T


        #Reading positions fractional cartesian
        position_number=file_map.find('position of ions in fractional coordinates')
        file_map.seek(position_number)
        file_map.readline()

        positions_fractional = []
        for i in range (number_of_atoms):
            positions_fractional.append(file_map.readline().split()[0:number_of_dimensions])
        positions_fractional = np.array(positions_fractional,dtype='double')


        #Reading positions cartesian
        position_number=file_map.find('position of ions in cartesian coordinates')
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
#                              primitive_cell=primitive_cell
                              )


def read_from_file_structure_poscar(file_name):
    #Check file exists
    if not os.path.isfile(file_name):
        print('Structure file does not exist!')
        exit()

    #Read from VASP OUTCAR file
    print("Reading VASP POSCAR structure")
    poscar_file = open(file_name, 'r')
    data_lines = poscar_file.read().split('\n')
    poscar_file.close()

    direct_cell = np.array([data_lines[i].split()
                            for i in range(2,5)],dtype=float).T
    try:
        number_of_types = np.array(data_lines[6].split(),dtype=int)
        scaled_positions = np.array([data_lines[8+k].split()
                                     for k in range(np.sum(number_of_types))],dtype=float)
        atomic_types = []

        for i,j in enumerate(data_lines[5].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = np.array(atomic_types).flatten().tolist()

    except ValueError:
        number_of_types = np.array(data_lines[5].split(),dtype=int)
        scaled_positions = np.array([data_lines[7+k].split()
                                     for k in range(np.sum(number_of_types))],dtype=float)
        atomic_types = []
        for i,j in enumerate(data_lines[0].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = np.array(atomic_types).flatten().tolist()


    return atomtest.Structure(cell= direct_cell,
                              scaled_positions=scaled_positions,
                              atomic_types=atomic_types,
#                              primitive_cell=primitive_cell
                              )


def read_from_file_trajectory(file_name,structure=None,
                              limit_number_steps=1000000,  #Maximum number of steps read
                              last_steps=1000000):         #Total number of read steps (deprecated)

    #Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    #Starting reading
    print("Reading VASP trajectory")
    print("This could take long, please wait..")

    #Dimensionality of VASP calculation
    number_of_dimensions = 3

    with open(file_name, "r+") as f:
        #Memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)
        position_number=file_map.find('NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())

        #Read time step
        position_number=file_map.find('POTIM  =')
        file_map.seek(position_number+8)
        time_step = float(file_map.readline().split()[0])* 1E-3 # in picoseconds

        #Reading super cell
        position_number = file_map.find('direct lattice vectors')
        file_map.seek(position_number)
        file_map.readline()
        super_cell = []
        for i in range (number_of_dimensions):
            super_cell.append(file_map.readline().split()[0:number_of_dimensions])
        super_cell = np.array(super_cell,dtype='double').T

        file_map.seek(position_number)
        file_map.readline()

    # Check if number of atoms is multiple of cell atoms
        if structure:
            if number_of_atoms % structure.get_number_of_cell_atoms() != 0:
                print('Warning: Number of atoms not matching, check VASP output files')
    #        structure.set_number_of_atoms(number_of_atoms)

#       Read coordinates and energy
        trajectory = []
        energy = []
        while True :
            position_number=file_map.find('POSITION')
            if position_number < 0 : break

            file_map.seek(position_number)
            file_map.readline()
            file_map.readline()

            read_coordinates = []
            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])
            position_number=file_map.find('energy(')
            file_map.seek(position_number)
            read_energy = file_map.readline().split()[2]
            trajectory.append(np.array(read_coordinates,dtype=float).flatten()) #in angstrom
            energy.append(np.array(read_energy,dtype=float))
            limit_number_steps -= 1

            if limit_number_steps < 0:
                break

        file_map.close()

        trajectory = np.array([[[trajectory[i][j*number_of_dimensions+k]
                                 for k in range(number_of_dimensions)]
                                for j in range(number_of_atoms)]
                               for i in range (len(trajectory))])

        trajectory = trajectory[-last_steps:,:,:]
        energy = energy[-last_steps:]

        print('Number of total steps read:',trajectory.shape[0])
        time = np.array([ i*time_step for i in range(trajectory.shape[0])],dtype=float)

        print('Trajectory file read')
        return dyn.Dynamics(structure = structure,
                            trajectory = np.array(trajectory,dtype=complex),
                            energy = np.array(energy),
                            time=time,
                            super_cell=super_cell)


#Just for testing
def generate_test_trajectory(structure,q_vector_o,super_cell=(1,1,1)):

    print('Generating ideal harmonic data for testing')

    #Getting data from file instead of calculating (has to be the same object type generated by this function)
    if False:
        dump_file = open( "trajectory.save", "r" )
        trajectory = pickle.load(dump_file)
        return  trajectory

    number_of_atoms = structure.get_number_of_cell_atoms()
    positions = structure.get_positions(super_cell=super_cell)
    masses = structure.get_masses(super_cell=super_cell)


    #Parameters used to generate harmonic trajectory
    total_time = 2.0
    time_step = 0.001
    amplitude = 50.0
#    print('Freq Num',number_of_frequencies)

    for i in range(structure.get_number_of_dimensions()):
        number_of_atoms *= super_cell[i]
#    print('At Num',number_of_atoms)

    atom_type = structure.get_atom_type_index(super_cell=super_cell)
#    print('At type',atom_type)


    print(structure.get_atomic_types(super_cell=super_cell))
    #Generate an xyz file for checking
    xyz_file = open('test.xyz','w')

    #Generate additional random wave vectors sample for further testing
    number_of_wave_vectors = 0
    q_vector_r=np.random.random([number_of_wave_vectors, 3])
  #  np.random.random(number_of_wave_vectors,3)
    q_vector_r=np.concatenate((q_vector_r,[q_vector_o]),axis=0)
    print('test wave vectors')
    print(q_vector_r)

    #Generate frequencies and eigenvectors for the testing wave vector samples
    eigenvectors_r = []
    frequencies_r = []
    for i in range(len(q_vector_r)):
        print(q_vector_r[i])
        eigenvectors, frequencies = pho_interface.obtain_eigenvectors_from_phonopy(structure,q_vector_r[i])
        eigenvectors_r.append(eigenvectors)
        frequencies_r.append(frequencies)
    number_of_frequencies = len(frequencies_r[0])
    print('obtained frequencies')
    print(frequencies_r)


    print(np.pi*2.0*np.linalg.inv(structure.get_primitive_cell()).T)
    #Generating trajectory
    trajectory = []
    for time in np.arange(total_time,step=time_step):
        print(time)
        xyz_file.write(str(number_of_atoms) + '\n\n')
        coordinates = []
        for i_atom in range(number_of_atoms):
       #     coordinate = map(complex,positions[i_atom])
            coordinate = np.array(positions[i_atom,:],dtype=complex)
            for i_freq in range(number_of_frequencies):
                for i_long in range(q_vector_r.shape[0]):
                    q_vector = np.dot(q_vector_r[i_long,:], 2*np.pi*np.linalg.inv(structure.get_primitive_cell()))
                    # Beware in the testing amplitude!! Normalized for all phonons to have the same height!!
                    if abs(frequencies_r[i_long][i_freq]) > 0.01: #Prevent dividing by 0
                        coordinate += amplitude / (np.sqrt(masses[i_atom]) * frequencies_r[i_long][i_freq]) * (
                                      eigenvectors_r[i_long][i_freq,atom_type[i_atom]] *
                                      np.exp(np.complex(0,-1) * frequencies_r[i_long][i_freq] * 2.0 * np.pi * time) *
                                      np.exp(np.complex(0,1) * np.dot(q_vector,positions[i_atom,:])))

            xyz_file.write(structure.get_atomic_types(super_cell=super_cell)[i_atom]+'\t' +
                           '\t'.join([str(item) for item in coordinate.real]) + '\n')
            coordinates.append(coordinate)
        trajectory.append(coordinates)
    xyz_file.close()

    trajectory = np.array(trajectory)
    print(trajectory.shape[0])


    time = np.array([ i*time_step for i in range(trajectory.shape[0])],dtype=float)
    energy = np.array([ 0*i for i in range(trajectory.shape[0])],dtype=float)

    #Save a trajectory object to file for later recovery
    dump_file = open("trajectory.save", "w")
    pickle.dump(dyn.Dynamics(structure=structure,
                             trajectory=np.array(trajectory, dtype=complex),
                             energy=np.array(energy),
                             time=time,
                             super_cell=np.dot(np.diagflat(super_cell),structure.get_cell())),
                dump_file)

    dump_file.close()

    print(np.dot(np.diagflat(super_cell),structure.get_cell()))

    return dyn.Dynamics(structure=structure,
                        trajectory=np.array(trajectory,dtype=complex),
                        energy=np.array(energy),
                        time=time,
                        super_cell=np.dot(np.diagflat(super_cell),structure.get_cell()))


#Testing function
def read_from_file_test():

    print('Reading structure from test file')

    #Condicions del test
    number_of_dimensions = 2

    f_coordinates = open('Data Files/test.out', 'r')
    f_velocity = open('Data Files/test2.out', 'r')
    f_trajectory = open('Data Files/test3.out', 'r')


    #Coordinates reading
    positions = []
    while True:
        row = f_coordinates.readline().split()
        if not row: break
        for i in range(len(row)): row[i] = float(row[i])
        positions.append(row)

    atom_type = np.array(positions,dtype=int)[:,2]
    positions = np.array(positions)[:,:number_of_dimensions]
    print('Coordinates reading complete')

    structure = atomtest.Structure(positions=positions,
                                   atomic_numbers=atom_type,
                                   cell=[[2,0],[0,1]],
                                   masses=[1 for i in range(positions.shape[0])]) #all 1
    number_of_atoms = structure.get_number_of_atoms()

    structure.set_number_of_primitive_atoms(2)
    print('number of atoms in primitive cell')
    print(structure.get_number_of_primitive_atoms())
    print('number of total atoms in structure (super cell)')
    print(number_of_atoms)

    #Velocity reading section
    velocity = []
    while True:
        row = f_velocity.readline().replace('I','j').replace('*','').replace('^','E').split()
        if not row: break
        for i in range(len(row)): row[i] = complex('('+row[i]+')')
        velocity.append(row)
  #  velocity = velocity[:4000][:]  #Limitate the number of points (just for testing)

    time = np.array([velocity[i][0]  for i in range(len(velocity))]).real
    velocity = np.array([[[velocity[i][j*number_of_dimensions+k+1]
                           for k in range(number_of_dimensions)]
                          for j in range(number_of_atoms)]
                         for i in range (len(velocity))])
    print('Velocity reading complete')


    #Trajectory reading
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


def write_correlation_to_file(frequency_range,correlation_vector,file_name):
    output_file = open(file_name, 'w')

    for i in range(correlation_vector.shape[0]):
        output_file.write("{0:10.4f}\t".format(frequency_range[i]))
        for j in correlation_vector[i,:]:
            output_file.write("{0:.10e}\t".format(j))
        output_file.write("\n")

    output_file.close()
    return 0


def read_parameters_from_input_file(file_name):

    input_parameters = {'structure_file_name_poscar': 'POSCAR'}

    input_file = open(file_name, "r").readlines()
    for i, line in enumerate(input_file):

        if "STRUCTURE FILE OUTCAR" in line:
            input_parameters.update({'structure_file_name_outcar': input_file[i+1].replace('\n','')})

        if "STRUCTURE FILE POSCAR" in line:
            input_parameters.update({'structure_file_name_poscar': input_file[i+1].replace('\n','')})

        if "FORCE CONSTANTS" in line:
            input_parameters.update({'force_constants_file_name': input_file[i+1].replace('\n','')})

        if "PRIMITIVE MATRIX" in line:
            primitive_matrix = [input_file[i+1].replace('\n','').split(),
                                input_file[i+2].replace('\n','').split(),
                                input_file[i+3].replace('\n','').split()]
            input_parameters.update({'primitive_matrix': np.array(primitive_matrix, dtype=float)})


        if "SUPERCELL MATRIX PHONOPY" in line:
            super_cell_matrix = [input_file[i+1].replace('\n','').split(),
                                 input_file[i+2].replace('\n','').split(),
                                 input_file[i+3].replace('\n','').split()]

            super_cell_matrix = np.array(super_cell_matrix, dtype=int)
            input_parameters.update({'super_cell_matrix': np.array(super_cell_matrix, dtype=int)})


        if "BANDS" in line:
            bands = []
            while i < len(input_file)-1:
                try:
                    band = np.array(input_file[i+1].replace(',',' ').split(),dtype=float).reshape((2,3))
                except IOError:
                    break
                except ValueError:
                    break
                i += 1
                bands.append(band)
            input_parameters.update ({'bands':bands})


    return input_parameters

def write_xsf_file(file_name,structure):

    xsf_file = open(file_name,"w")

    xsf_file.write("CRYSTAL\n")
    xsf_file.write("PRIMVEC\n")

    for row in structure.get_primitive_cell().T:
        xsf_file.write("{0:10.4f}\t{1:10.4f}\t{2:10.4f}\n".format(*row))
    xsf_file.write("CONVVEC\n")

    for row in structure.get_cell().T:
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


def save_data_hdf5(file_name, velocity, time, super_cell):
    hdf5_file = h5py.File(file_name, "w")

    hdf5_file.create_dataset('velocity', data=velocity)
    hdf5_file.create_dataset('time', data=time)
    hdf5_file.create_dataset('super_cell', data=super_cell)

    print("saved", velocity.shape[0], "steps")
    hdf5_file.close()


def read_data_hdf5(file_name):

    hdf5_file = h5py.File(file_name, "r")
    velocity = hdf5_file['velocity'][:]
    hdf5_file.close()

    return velocity


def initialize_from_file(file_name,structure):
    print("Reading velocity from hdf5 file: " + file_name)

    hdf5_file = h5py.File(file_name, "r")
    velocity = hdf5_file['velocity'][:]
    time = hdf5_file['time'][:]
    super_cell = hdf5_file['super_cell'][:]
    hdf5_file.close()

    return dyn.Dynamics(structure = structure,
                        velocity = velocity,
                        time=time,
                        super_cell=np.dot(np.diagflat(super_cell), structure.get_cell()))
