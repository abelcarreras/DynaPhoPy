import numpy as np
import Classes.dynamics as dyn
import Classes.atomstest as atomtest
import mmap
import phonopy.file_IO as file_IO

#import ase.io.vasp
#import ase.atoms as Atoms


"""
def read_from_file_structure2(file_name):
    estructura = Atoms.Atoms(ase.io.vasp.read_vasp_out(filename=file_name, index=-1))
 #   print(estructura.get_cell())
 #   print(estructura.get_forces())
 #   print(estructura.get_masses())
 #   print(estructura.get_positions())
 #   print(estructura.get_scaled_positions())
 #   print(estructura.get_atomic_numbers())
 #   print(estructura.get_number_of_atoms())
    return atomtest.Structure(cell= estructura.get_cell(),
                                   positions=estructura.get_positions(),
                                   masses=estructura.get_masses(),
                                   forces=estructura.get_forces(),
                                   atomic_numbers=estructura.get_atomic_numbers())

"""

def read_from_file_structure(file_name):
    #Read from VASP OUTCAR file

    with open(file_name, "r+") as f:
        # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)

        #Setting number of dimensions
        number_of_dimensions = 3

        #Reading number of atoms
        position_number = file_map.find('NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())
#        print('Number of atoms:',number_of_atoms)

        #Reading atoms per type
        position_number = file_map.find('ions per type')
        file_map.seek(position_number+15)
        atoms_per_type = np.array(file_map.readline().split(),dtype=int)


        #Reading atoms  mass
        position_number = file_map.find('POMASS =')
#        file_map.seek(position_number)
        atomic_mass_per_type = []
#        print('pos:',position_number)
        for i in range(atoms_per_type.shape[0]):
            file_map.seek(position_number+9+6*i)
            atomic_mass_per_type.append(file_map.read(6))
        atomic_mass = sum([ [atomic_mass_per_type[j] for i in range(atoms_per_type[j])] for j in range(atoms_per_type.shape[0])],[])
        atomic_mass = np.array(atomic_mass,dtype='double')


        #Reading cell
        position_number = file_map.find('direct lattice vectors')
        file_map.seek(position_number)
        file_map.readline()
        direct_cell = []    #Direct Cell
        for i in range (number_of_dimensions):
            direct_cell.append(file_map.readline().split()[0:number_of_dimensions])
#        print(direct_cell)
        direct_cell = np.array(direct_cell,dtype='double')

        file_map.seek(position_number)
        file_map.readline()

        reciprocal_cell = []    #Reciprocal cell
        for i in range (number_of_dimensions):
            reciprocal_cell.append(file_map.readline().split()[number_of_dimensions:number_of_dimensions*2])
        reciprocal_cell = np.array(reciprocal_cell,dtype='double')


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


def read_from_file_trajectory(file_name,structure):

#   Maximum number of structures that's gonna be read
    limit_number_structures = 99000
    last_points_taken = 10000

    with open(file_name, "r+") as f:
    # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)
        position_number=file_map.find('NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())
#        print('Number of atoms:',number_of_atoms)
#       check number of atoms

#       Read time step
        position_number=file_map.find('POTIM  =')
        file_map.seek(position_number+8)
        time_step = float(file_map.readline().split()[0])



################Change to check if multiple ############################
        if number_of_atoms != structure.get_number_of_cell_atoms():
            print('Warning: Number of atoms not matching, check VASP output files')
        structure.set_number_of_atoms(number_of_atoms)
######################################################################


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
#                print np.array(( file_map.readline().split()[0:3] ),dtype='double')
                read_coordinates.append(file_map.readline().split()[0:structure.get_number_of_dimensions()])
            position_number=file_map.find('energy(')
            file_map.seek(position_number)
            read_energy = file_map.readline().split()[2]
            trajectory.append(np.array(read_coordinates,dtype=float).flatten())
            energy.append(np.array(read_energy,dtype=float))
#            print(np.array(supercell,dtype=float).flatten())
            limit_number_structures -= 1

            if limit_number_structures < 0:
                break

        file_map.close()

#        print('fi')
 #       print(number_of_atoms)
#        trajectory = np.array(trajectory)
        number_of_dimensions = structure.get_number_of_dimensions()
 #       print(trajectory[0])

        trajectory = np.array([[[trajectory[i][j*number_of_dimensions+k] for k in range(number_of_dimensions) ] for j in range(number_of_atoms)] for i in range (len(trajectory))])
#        print (trajectory[0,1,:])

        trajectory = trajectory[-last_points_taken:,:,:]
        energy = energy[-last_points_taken:]

        print('No Points:',trajectory.shape[0])
        time = np.array([ i*time_step for i in range(trajectory.shape[0])],dtype=float)


        print('Trajectory file read')
        return dyn.Dynamics(structure = structure,
                            trajectory = np.array(trajectory),
                            energy = np.array(energy),
                            time=time)



def generate_test_trajectory(structure,eigenvectors,frequencies,q_vector_o):

    print('Making fake ideal data for testing')
    super_cell= structure.get_super_cell_matrix()

    q_vector_o = np.array ([0.2,0.1,0.4])
    number_of_atoms = structure.get_number_of_cell_atoms()
    number_of_frequencies = len(frequencies)
    total_time = 5
    time_step = 0.01
    amplitude = 0.5/len(np.arange(0,2,0.1))
#    print('Freq Num',number_of_frequencies)

    for i in range(structure.get_number_of_dimensions()):
        number_of_atoms *= super_cell[i]
#    print('At Num',number_of_atoms)

    atom_type = structure.get_atom_type_index(super_cell=super_cell)
#    print('At type',atom_type)


    print(structure.get_atomic_types(super_cell=super_cell))
    #Generate an xyz file for checking
    xyz_file = open('Data Files/test.xyz','w')

    trajectory = []

    for time in np.arange(total_time,step=time_step):
        print(time)
        xyz_file.write(str(number_of_atoms) + '\n\n')
        coordinates = []
        for i_atom in range(number_of_atoms):
            coordinate = map(complex,structure.get_positions(super_cell=super_cell)[i_atom])
            for i_freq in range(number_of_frequencies):
                for i_long in np.arange(0,2,0.01):
                    q_vector = np.array(q_vector_o) * i_long
                    coordinate += 1 / np.sqrt(structure.get_masses(super_cell=super_cell)[i_atom]) *\
                                  amplitude * eigenvectors[i_freq,atom_type[i_atom]]*\
                                  np.exp(np.complex(0,-1)*frequencies[i_freq]*time)*\
                                  np.exp(np.complex(0,1)*np.dot(q_vector,structure.get_positions(super_cell=super_cell)[i_atom]))

#            print('\t'.join([str(item) for item in coordinate]))

            xyz_file.write(structure.get_atomic_types(super_cell=super_cell)[i_atom]+'\t'+
                           '\t'.join([str(item) for item in coordinate.real]) + '\n')
            coordinates.append(coordinate)
        trajectory.append(coordinates)
    xyz_file.close()


    trajectory = np.array(trajectory)
    print(trajectory.shape[0])


    time = np.array([ i*time_step for i in range(trajectory.shape[0])],dtype=float)
    energy = np.array([ 0*i for i in range(trajectory.shape[0])],dtype=float)

###########################CAL CANVIAR EN ALGUN MOMENT#################
    structure.set_number_of_atoms(number_of_atoms)
##########################################################################

    return dyn.Dynamics(structure = structure,
                        trajectory = np.array(trajectory,dtype=complex),
                        energy = np.array(energy),
                        time=time)




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
#    structure.set_primitive_matrix([[1.0, 0.0],
#                                    [0.0, 1.0]])

    #Velocity reading section
    velocity = []
    while True:
        row = f_velocity.readline().replace('I','j').replace('*','').replace('^','E').split()
        if not row: break
        for i in range(len(row)): row[i] = complex('('+row[i]+')')
        velocity.append(row)
  #  velocity = velocity[:4000][:]  #Limitate the number of points (just for testing)

    time = np.array([velocity[i][0]  for i in range(len(velocity))]).real
    velocity = np.array([[[velocity[i][j*number_of_dimensions+k+1] for k in range( number_of_dimensions ) ] for j in range(number_of_atoms)] for i in range (len(velocity))])
    print('Velocity reading complete')


    #Trajectory reading
    trajectory = []
    while True:
        row = f_trajectory.readline().replace('I','j').replace('*','').replace('^','E').split()
        if not row: break
        for i in range(len(row)): row[i] = complex('('+row[i]+')')
        trajectory.append(row)

    trajectory = np.array([[[trajectory[i][j*number_of_dimensions+k+1] for k in range(number_of_dimensions) ] for j in range(number_of_atoms)] for i in range (len(trajectory))])

    print('Trajectory reading complete')

    return dyn.Dynamics(trajectory= trajectory,
        #                velocity=velocity,
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

#print (read_from_file_structure('../Data Files/NaCl/OUTCAR').get_number_of_dimensions())
#print (read_from_file_structure2('../Data Files/NaCl/OUTCAR'))
#print (read_from_file_trajectory('/home/abel/Desktop/Bi2O3_md/OUTCAR')[1,79,:])

