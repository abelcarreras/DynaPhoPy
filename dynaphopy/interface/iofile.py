import mmap
import pickle
import os
import numpy as np

import dynaphopy.classes.dynamics as dyn
import dynaphopy.classes.atoms as atomtest
from dynaphopy.interface import phonopy_link as pho_interface


def check_trajectory_file_type(file_name, bytes_to_check=1000000):

    #Check file exists
    if not os.path.isfile(file_name):
        print file_name + ' file does not exist'
        exit()

    file_size = os.stat(file_name).st_size

    #Check if LAMMPS file
    with open (file_name, "r+") as f:
        file_map = mmap.mmap(f.fileno(), np.min([bytes_to_check, file_size]))
        num_test = [file_map.find('ITEM: TIMESTEP'),
                    file_map.find('ITEM: NUMBER OF ATOMS'),
                    file_map.find('ITEM: BOX BOUNDS')]

    if not -1 in num_test:
            return read_lammps_trajectory

    #Check if VASP file
    with open (file_name, "r+") as f:
        file_map = mmap.mmap(f.fileno(), np.min([bytes_to_check, file_size]))
        num_test = [file_map.find('NIONS'),
                    file_map.find('POMASS'),
                    file_map.find('direct lattice vectors')]

    if not -1 in num_test:
            return read_vasp_trajectory

    print('Trajectory file not recognized')
    exit()
    return None


def read_from_file_structure_outcar(file_name):

    #Check file exists
    if not os.path.isfile(file_name):
        print('Structure file does not exist!')
        exit()

    #Read from VASP OUTCAR file
    print('Reading VASP structure')

    with open(file_name, "r+") as f:
        # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)


        #Setting number of dimensions
        number_of_dimensions = 3

        #trash reading for guessing primitive cell (Not stable)
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

    multiply = float(data_lines[1])
    direct_cell = np.array([data_lines[i].split()
                            for i in range(2,5)],dtype=float).T
    direct_cell *= multiply
    try:
        number_of_types = np.array(data_lines[6].split(),dtype=int)
        scaled_positions = np.array([data_lines[8+k].split()[0:3]
                                     for k in range(np.sum(number_of_types))],dtype=float)
        atomic_types = []

        for i,j in enumerate(data_lines[5].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = [item for sublist in atomic_types for item in sublist]
#        atomic_types = np.array(atomic_types).flatten().tolist()


    #Old style POSCAR format
    except ValueError:
        print "Reading old style POSCAR"
        number_of_types = np.array(data_lines[5].split(), dtype=int)
        scaled_positions = np.array([data_lines[7+k].split()[0:3]
                                     for k in range(np.sum(number_of_types))],dtype=float)
        atomic_types = []
        for i,j in enumerate(data_lines[0].split()):
            atomic_types.append([j]*number_of_types[i])
        atomic_types = [item for sublist in atomic_types for item in sublist]
       # atomic_types = np.array(atomic_types).flatten().tolist()
    return atomtest.Structure(cell= direct_cell,
                              scaled_positions=scaled_positions,
                              atomic_types=atomic_types,
#                              primitive_cell=primitive_cell
                              )


def read_vasp_trajectory(file_name, structure=None, time_step=None,
                         limit_number_steps=10000000,  # Maximum number of steps read (for security)
                         last_steps=None,
                         initial_cut=1,
                         end_cut=None):


    #Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    #Check time step
    if time_step is not None:
        print('Warning! Time step flag has no effect reading VASP OUTCAR file (time step will be read from OUTCAR)')

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
        counter = 0
        while True :

            counter +=1

            #Initial cut control
            if initial_cut > counter:
                continue


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
            energy.append(np.array(read_energy, dtype=float))

            #security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

        file_map.close()

        trajectory = np.array([[[trajectory[i][j*number_of_dimensions+k]
                                 for k in range(number_of_dimensions)]
                                for j in range(number_of_atoms)]
                               for i in range (len(trajectory))])

        if last_steps is not None:
            trajectory = trajectory[-last_steps:,:,:]
            energy = energy[-last_steps:]


        print('Number of total steps read: {0}'.format(trajectory.shape[0]))
        time = np.array([i*time_step for i in range(trajectory.shape[0])], dtype=float)

        print('Trajectory file read')
        return dyn.Dynamics(structure=structure,
                            trajectory=np.array(trajectory, dtype=complex),
                            energy=np.array(energy),
                            time=time,
                            super_cell=super_cell)


#Just for testing (use with care)
def generate_test_trajectory(structure, super_cell=(1, 1, 1),
                             save_to_file=None,
                             minimum_frequency=0.1,  # THz
                             total_time=2,           # picoseconds
                             time_step=0.002,        # picoseconds
                             temperature=400):       # Kelvin

    import random
    from dynaphopy.power_spectrum import progress_bar

    print('Generating ideal harmonic data for testing')
    kb_boltzmann = 0.831446 # u * A^2 / ( ps^2 * K )


    number_of_unit_cells_phonopy = np.prod(np.diag(structure.get_super_cell_phonon()))
    number_of_unit_cells = np.prod(super_cell)
    atoms_relation = float(number_of_unit_cells)/ number_of_unit_cells_phonopy

#    print(number_of_unit_cells_phonopy, number_of_unit_cells)


    #Recover dump trajectory from file (test only)
    if False:
        dump_file = open( "trajectory.save", "r" )
        trajectory = pickle.load(dump_file)
        return trajectory

    number_of_atoms = structure.get_number_of_cell_atoms()
    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()
    number_of_dimensions = structure.get_number_of_dimensions()

    positions = structure.get_positions(super_cell=super_cell)
    masses = structure.get_masses(super_cell=super_cell)


    number_of_atoms = number_of_atoms*number_of_unit_cells
 #   print('At Num',number_of_atoms)

 #   exit()

    number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms

    atom_type = structure.get_atom_type_index(super_cell=super_cell)
#    print('At type',atom_type)

    #Generate an xyz file for checking
    if save_to_file is None:
        xyz_file = open(os.devnull, 'w')
    else:
        xyz_file = open(save_to_file, 'w')

    #Generate additional wave vectors sample
    q_vector_list = pho_interface.get_commensurate_points(structure)

    #Generate frequencies and eigenvectors for the testing wave vector samples
    print('Wave vectors included in test (commensurate points)')
    eigenvectors_r = []
    frequencies_r = []
    for i in range(len(q_vector_list)):
        print(q_vector_list[i])
        eigenvectors, frequencies = pho_interface.obtain_eigenvectors_from_phonopy(structure, q_vector_list[i])
        eigenvectors_r.append(eigenvectors)
        frequencies_r.append(frequencies)
    number_of_frequencies = len(frequencies_r[0])

    #Generating trajectory
    progress_bar(0, 'generating')

    trajectory = []
    for time in np.arange(total_time, step=time_step):
      #  print(time)

        xyz_file.write('{0}\n\n'.format(number_of_atoms))
        coordinates = []
        for i_atom in range(number_of_atoms):
            coordinate = np.array(positions[i_atom, :], dtype=complex)
            for i_freq in range(number_of_frequencies):
                for i_long in range(q_vector_list.shape[0]):
                    q_vector = np.dot(q_vector_list[i_long,:], 2*np.pi*np.linalg.inv(structure.get_primitive_cell()))

                    if abs(frequencies_r[i_long][i_freq]) > minimum_frequency: # Prevent error due to small frequencies
                        # Amplitude is normalized to be equal area for all phonon projected power spectra.
                        amplitude = np.sqrt(2 * kb_boltzmann * temperature / number_of_primitive_cells * atoms_relation)/(frequencies_r[i_long][i_freq] * 2 * np.pi) # + random.uniform(-1,1)*0.05
                        normal_mode_coordinate = amplitude * np.exp(np.complex(0, -1) * frequencies_r[i_long][i_freq] * 2.0 * np.pi * time)
                        phase = np.exp(np.complex(0, 1) * np.dot(q_vector, positions[i_atom, :]))

                        coordinate += (1.0 / np.sqrt(masses[i_atom]) *
                                       eigenvectors_r[i_long][i_freq, atom_type[i_atom]] *
                                       phase *
                                       normal_mode_coordinate).real
                        coordinate = coordinate.real


            xyz_file.write(structure.get_atomic_types(super_cell=super_cell)[i_atom]+'\t' +
                           '\t'.join([str(item) for item in coordinate]) + '\n')

            coordinates.append(coordinate)
        trajectory.append(coordinates)
        progress_bar(float(time+time_step)/total_time,'generating', )


    xyz_file.close()

    trajectory = np.array(trajectory)
    print(trajectory.shape[0])

    time = np.array([i * time_step for i in range(trajectory.shape[0])], dtype=float)
    energy = np.array([number_of_atoms * number_of_dimensions *
                       kb_boltzmann * temperature
                       for i in range(trajectory.shape[0])], dtype=float)

    #Save a trajectory object to file for later recovery (test only)
    if False:
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

    atom_type = np.array(positions,dtype=int)[:, 2]
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



def read_lammps_trajectory(file_name, structure=None, time_step=None,
                           limit_number_steps=10000000,
                           last_steps=None,
                           initial_cut=1,
                           end_cut=None):

 #Time in picoseconds
 #Coordinates in Angstroms



    number_of_atoms = None
    bounds = None

    #Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    #Check time step
    if time_step is None:
        print('Warning! LAMMPS trajectory file does not contain time step information')
        print('Using default: 0.001 ps')
        time_step = 0.001

    #Starting reading
    print("Reading LAMMPS trajectory")
    print("This could take long, please wait..")

    #Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    time = []
    trajectory = []
    counter = 0

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        while True:

            counter += 1

            #Read time steps
            position_number=file_map.find('TIMESTEP')
            if position_number < 0: break

            file_map.seek(position_number)
            file_map.readline()
            time.append(float(file_map.readline()))


            if number_of_atoms is None:
                #Read number of atoms
                file_map = mmap.mmap(f.fileno(), 0)
                position_number=file_map.find('NUMBER OF ATOMS')
                file_map.seek(position_number)
                file_map.readline()
                number_of_atoms = int(file_map.readline())

                # Check if number of atoms is multiple of cell atoms
                if structure:
                    if number_of_atoms % structure.get_number_of_cell_atoms() != 0:
                        print('Warning: Number of atoms not matching, check LAMMPS output file')

            if bounds is None:
                #Read cell
                file_map = mmap.mmap(f.fileno(), 0)
                position_number=file_map.find('BOX BOUNDS')
                file_map.seek(position_number)
                file_map.readline()


                bounds = []
                for i in range(3):
                    bounds.append(file_map.readline().split())

                bounds = np.array(bounds, dtype=float)
                if bounds.shape[1] == 2:
                    bounds = np.append(bounds, np.array([0, 0, 0])[None].T ,axis=1)

                xy = bounds[0, 2]
                xz = bounds[1, 2]
                yz = bounds[2, 2]

                xlo = bounds[0, 0] - np.min([0.0, xy, xz, xy+xz])
                xhi = bounds[0, 1] - np.max([0.0, xy, xz, xy+xz])
                ylo = bounds[1, 0] - np.min([0.0, yz])
                yhi = bounds[1, 1] - np.max([0.0, yz])
                zlo = bounds[2, 0]
                zhi = bounds[2, 1]

                super_cell = np.array([[xhi-xlo, xy,  xz],
                                       [0,  yhi-ylo,  yz],
                                       [0,   0,  zhi-zlo]])

# Testing cell
                lx = xhi-xlo
                ly = yhi-ylo
                lz = zhi-zlo

                a = lx
                b = np.sqrt(pow(ly,2) + pow(xy,2))
                c = np.sqrt(pow(lz,2) + pow(xz,2) +  pow(yz,2))

                alpha = np.arccos((xy*xz + ly*yz)/(b*c))
                beta = np.arccos(xz/c)
                gamma = np.arccos(xy/b)
#End testing cell


            position_number = file_map.find('ITEM: ATOMS')

            file_map.seek(position_number)
            file_map.readline()

            #Initial cut control
            if initial_cut > counter:
                continue

            #Reading coordinates
            read_coordinates = []
            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])

            try:
                trajectory.append(np.array(read_coordinates, dtype=float)) #in angstroms

            except ValueError:
                print("Error reading step {0}".format(counter))
                break
        #        print(read_coordinates)

            #security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break


    file_map.close()

    time = np.array(time) * time_step
    trajectory = np.array(trajectory, dtype=complex)
    if last_steps is not None:
        trajectory = trajectory[-last_steps:, :, :]
        time = time[-last_steps:]


    return dyn.Dynamics(structure=structure,
                        trajectory=trajectory,
                        time=time,
                        super_cell=super_cell)



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

    #Check file exists
    if not os.path.isfile(file_name):
        print file_name + ' file does not exist'
        exit()

    input_file = open(file_name, "r").readlines()
    for i, line in enumerate(input_file):
        if line[0] == '#':
            continue

        if "STRUCTURE FILE OUTCAR" in line:
            input_parameters.update({'structure_file_name_outcar': input_file[i+1].replace('\n','')})

        if "STRUCTURE FILE POSCAR" in line:
            input_parameters.update({'structure_file_name_poscar': input_file[i+1].replace('\n','')})

        if "FORCE SETS" in line:
            input_parameters.update({'force_sets_file_name': input_file[i+1].replace('\n','')})

        if "FORCE CONSTANTS" in line:
            input_parameters.update({'force_constants_file_name': input_file[i+1].replace('\n','')})
       #     print('Warning!: FORCE CONSTANTS label in input has changed. Please use FORCE SETS instead')
       #     exit()

        if "PRIMITIVE MATRIX" in line:
            primitive_matrix = [input_file[i+1].replace('\n','').split(),
                                input_file[i+2].replace('\n','').split(),
                                input_file[i+3].replace('\n','').split()]
            input_parameters.update({'_primitive_matrix': np.array(primitive_matrix, dtype=float)})


        if "SUPERCELL MATRIX PHONOPY" in line:
            super_cell_matrix = [input_file[i+1].replace('\n','').split(),
                                 input_file[i+2].replace('\n','').split(),
                                 input_file[i+3].replace('\n','').split()]

            super_cell_matrix = np.array(super_cell_matrix, dtype=int)
            input_parameters.update({'_super_cell_phonon': np.array(super_cell_matrix, dtype=int)})


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
            input_parameters.update ({'_band_ranges':bands})

        if "MESH PHONOPY" in line:
            input_parameters.update({'_mesh_phonopy': input_file[i+1].replace('\n','').split()})


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

 #   print("saved", velocity.shape[0], "steps")
    hdf5_file.close()


def initialize_from_hdf5_file(file_name, structure, read_trajectory=True, initial_cut=1, final_cut=None):
    import h5py

    print("Reading data from hdf5 file: " + file_name)

    trajectory = None
    velocity = None
    vc = None
    reduced_q_vector = None

    #Check file exists
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
    super_cell = hdf5_file['super_cell'][:]
    hdf5_file.close()

    if vc is None:
        return dyn.Dynamics(structure=structure,
                            trajectory=trajectory,
                            velocity=velocity,
                            time=time,
                            super_cell=np.dot(np.diagflat(super_cell), structure.get_cell()))
    else:
        return vc, reduced_q_vector, dyn.Dynamics(structure=structure,
                                time=time,
                                super_cell=np.dot(np.diagflat(super_cell), structure.get_cell()))


