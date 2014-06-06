import numpy as np
import Classes.dynamics as dyn
import Classes.atomstest as atomtest
import mmap
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

        #Reading dimensions
        position_number = file_map.find('ion  position')
        file_map.seek(position_number)
        file_map.readline()
        number_of_dimensions = len(file_map.readline().split())-1

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
        direct_cell = np.array(direct_cell,dtype='double')

        file_map.seek(position_number)
        file_map.readline()

        reciprocal_cell = []    #Reciprocal cell
        for i in range (number_of_dimensions):
            reciprocal_cell.append(file_map.readline().split()[number_of_dimensions:number_of_dimensions*2])
        reciprocal_cell = np.array(reciprocal_cell,dtype='double')

        #Normalized cell
        cell_normalized = direct_cell / np.linalg.norm(direct_cell, axis=-1)[:, np.newaxis]


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

    return atomtest.Structure(cell= cell_normalized,
                              positions=positions,
                              forces=None,
                              atomic_types=atomic_types)


def read_from_file_trajectory(file_name):

    limit_number_structures = 10000

    with open(file_name, "r+") as f:
    # memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)
        position_number=file_map.find('NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())
#        print('Number of atoms:',number_of_atoms)
        trajectory = []
        while True :
            position_number=file_map.find('POSITION')
            if position_number < 0 : break
            file_map.seek(position_number)
            file_map.readline()
            file_map.readline()
            supercell = []
            for i in range (number_of_atoms):
#                print np.array(( file_map.readline().split()[0:3] ),dtype='double')
                supercell.append(file_map.readline().split()[0:3])
            trajectory.append(supercell)
            limit_number_structures -= 1

            if limit_number_structures < 0:
                break

        file_map.close()
        trajectory = np.array(trajectory)
        return dyn.Dynamics(trajectory= trajectory)


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



