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

        #Reading atom type
        position_number = file_map.find('POSCAR =')
        file_map.seek(position_number+8)
        types = file_map.readline().split()

        #Fromating Atomic types
        atomic_types = []
        for i in range(atoms_per_type.shape[0]):
            for j in range(atoms_per_type[i]):
                atomic_types.append(types[i])

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
                              forces=None,
                              atomic_types=atomic_types)


def read_from_file_trajectory(file_name,structure):

    limit_number_structures = 100000

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


        if number_of_atoms != structure.number_of_atoms:
            print('Warning: Number of atoms not matching, check VASP output files')
            exit()

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

        time = np.array([ i*time_step for i in range(trajectory.shape[0])],dtype=float)

        print('Trajectory file read')
        return dyn.Dynamics(structure = structure,
                            trajectory = np.array(trajectory),
                            energy = np.array(energy),
                            time=time)


def read_from_file_test():

    #Condicions del test
    number_of_dimensions = 2

    f_coordinates = open('/home/abel/Dropbox/PycharmProjects/DynaPhoPy/Data Files/test.out', 'r')
    f_velocity = open('/home/abel/Dropbox/PycharmProjects/DynaPhoPy/Data Files/test2.out', 'r')
    f_trajectory = open('/home/abel/Dropbox/PycharmProjects/DynaPhoPy/Data Files/test3.out', 'r')


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
                                   cell=[[1,0],[0,1]],
                                   number_of_cell_atoms=2,
                                   masses=[1 for i in range(positions.shape[0])]) #all 1

    number_of_atoms = structure.number_of_atoms


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
                        velocity=velocity,
                        time=time,
                        structure=structure)



#print (read_from_file_structure('../Data Files/NaCl/OUTCAR').get_number_of_dimensions())
#print (read_from_file_structure2('../Data Files/NaCl/OUTCAR'))
#print (read_from_file_trajectory('/home/abel/Desktop/Bi2O3_md/OUTCAR')[1,79,:])



