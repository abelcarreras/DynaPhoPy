import os
import numpy as np
import mmap
import dynaphopy.dynamics as dyn
import warnings


# VASP OUTCAR file parser
def read_vasp_trajectory(file_name, structure=None, time_step=None,
                         limit_number_steps=10000000,  # Maximum number of steps read (for security)
                         last_steps=None,
                         initial_cut=1,
                         end_cut=None,
                         memmap=False,
                         template=None):

    # warning
    warnings.warn('This parser will be deprecated, you can use XDATCAR instead', DeprecationWarning)

    # Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    # Check time step
    if time_step is not None:
        print('Warning! Time step flag has no effect reading from VASP OUTCAR file (time step will be read from file)')

    if memmap:
        print('Warning! Memory mapping is not implemented in VASP OUTCAR parser')

    # Starting reading
    print("Reading VASP trajectory")
    print("This may take long, please wait..")

    # Dimensionality of VASP calculation
    number_of_dimensions = 3

    with open(file_name, "r+") as f:

        #Memory-map the file
        file_map = mmap.mmap(f.fileno(), 0)
        position_number=file_map.find(b'NIONS =')
        file_map.seek(position_number+7)
        number_of_atoms = int(file_map.readline())

        #Read time step
        position_number=file_map.find(b'POTIM  =')
        file_map.seek(position_number+8)
        time_step = float(file_map.readline().split()[0])* 1E-3 # in picoseconds

        #Reading super cell
        position_number = file_map.find(b'direct lattice vectors')
        file_map.seek(position_number)
        file_map.readline()
        super_cell = []
        for i in range (number_of_dimensions):
            super_cell.append(file_map.readline().split()[0:number_of_dimensions])
        super_cell = np.array(super_cell, dtype='double')

        file_map.seek(position_number)
        file_map.readline()

    # Check if number of atoms is multiple of cell atoms
        if structure is not None:
            if number_of_atoms % structure.get_number_of_cell_atoms() != 0:
                print('Warning: Number of atoms not matching, check VASP output files')
    #        structure.set_number_of_atoms(number_of_atoms)

#       Read coordinates and energy
        trajectory = []
        energy = []
        counter = 0
        while True:

            counter +=1
            #Initial cut control
            if initial_cut > counter:
                continue

            position_number=file_map.find(b'POSITION')
            if position_number < 0 : break

            file_map.seek(position_number)
            file_map.readline()
            file_map.readline()

            read_coordinates = []
            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])

            read_coordinates = np.array(read_coordinates, dtype=float) # in angstrom
            if template is not None:
                indexing = np.argsort(template)
                read_coordinates = read_coordinates[indexing, :]

            position_number=file_map.find(b'energy(')
            file_map.seek(position_number)
            read_energy = file_map.readline().split()[2]
            trajectory.append(read_coordinates.flatten()) #in angstrom
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
                            supercell=super_cell,
                            memmap=memmap)


# LAMMPS custom dump file parser
def read_lammps_trajectory(file_name, structure=None, time_step=None,
                           limit_number_steps=10000000,
                           last_steps=None,
                           initial_cut=1,
                           end_cut=None,
                           memmap=False,
                           template=None):


    # Time in picoseconds
    # Coordinates in Angstroms

    # Read environtment variables
    try:
        temp_directory = os.environ["DYNAPHOPY_TEMPDIR"]
        if os.path.isdir(temp_directory):
            print('Set temporal directory: {0}'.format(temp_directory))
            temp_directory += '/'
        else:
            temp_directory = ''
    except KeyError:
        temp_directory = ''

    number_of_atoms = None
    bounds = None

    #Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    # Check time step
    if time_step is None:
        print('Warning! LAMMPS trajectory file does not contain time step information')
        print('Using default: 0.001 ps')
        time_step = 0.001

    # Starting reading
    print("Reading LAMMPS trajectory")
    print("This may take long, please wait..")

    # Dimension of LAMMP calculation
    if structure is None:
        number_of_dimensions = 3
    else:
        number_of_dimensions = structure.get_number_of_dimensions()

    time = []
    data = []
    counter = 0

    lammps_labels = False

    with open(file_name, "r+") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        while True:

            counter += 1

            #Read time steps
            position_number=file_map.find(b'TIMESTEP')
            if position_number < 0: break

            file_map.seek(position_number)
            file_map.readline()
            time.append(float(file_map.readline()))


            if number_of_atoms is None:
                #Read number of atoms
                file_map = mmap.mmap(f.fileno(), 0)
                position_number=file_map.find(b'NUMBER OF ATOMS')
                file_map.seek(position_number)
                file_map.readline()
                number_of_atoms = int(file_map.readline())

                # Check if number of atoms is multiple of cell atoms
                if structure is not None:
                    if number_of_atoms % structure.get_number_of_cell_atoms() != 0:
                        print('Warning: Number of atoms not matching, check LAMMPS output file')

            if bounds is None:
                #Read cell
                file_map = mmap.mmap(f.fileno(), 0)
                position_number=file_map.find(b'BOX BOUNDS')
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

                supercell = np.array([[xhi-xlo, xy,  xz],
                                       [0,  yhi-ylo,  yz],
                                       [0,   0,  zhi-zlo]]).T

                #for 2D
                supercell = supercell[:number_of_dimensions, :number_of_dimensions]

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

                # End testing cell

                # rotate lammps supercell to match unitcell orientation
                def unit_matrix(matrix):
                    return np.array([np.array(row)/np.linalg.norm(row) for row in matrix])

                unit_structure = unit_matrix(structure.get_cell())
                unit_supercell_lammps = unit_matrix(supercell)

                transformation_mat = np.dot(np.linalg.inv(unit_structure), unit_supercell_lammps).T

                supercell = np.dot(supercell, transformation_mat)

                if memmap:
                    if end_cut:
                        data = np.memmap(temp_directory+'trajectory.{0}'.format(os.getpid()), dtype='complex', mode='w+', shape=(end_cut - initial_cut+1, number_of_atoms, number_of_dimensions))
                    else:
                        print('Memory mapping requires to define reading range (use read_from/read_to option)')
                        exit()

            position_number = file_map.find(b'ITEM: ATOMS')

            file_map.seek(position_number)
            lammps_labels=file_map.readline()

            #Initial cut control
            if initial_cut > counter:
                time = []
                continue

            #Reading coordinates
            read_coordinates = []
            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])
            read_coordinates = np.array(read_coordinates, dtype=float)

            if template is not None:
                indexing = np.argsort(template)
                read_coordinates = read_coordinates[indexing, :]

            try:
                read_coordinates = np.dot(read_coordinates, transformation_mat)
                if memmap:
                    data[counter-initial_cut, :, :] = read_coordinates #in angstroms
                else:
                    data.append(read_coordinates) #in angstroms

            except ValueError:
                print("Error reading step {0}".format(counter))
                break
                # print(read_coordinates)

            #security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

    file_map.close()

    time = np.array(time) * time_step

    if not memmap:
        data = np.array(data, dtype=complex)

        if last_steps is not None:
            data = data[-last_steps:, :, :]
            time = time[-last_steps:]


    # Check position/velocity dump
    if b'vx vy' in lammps_labels:
        return dyn.Dynamics(structure=structure,
                            velocity=data,
                            time=time,
                            supercell=supercell,
                            memmap=memmap)

    if b'x y' in lammps_labels:
        return dyn.Dynamics(structure=structure,
                            trajectory=data,
                            time=time,
                            supercell=supercell,
                            memmap=memmap)

    print('LAMMPS parsing error. Data not recognized: {}'.format(lammps_labels))
    exit()


def read_VASP_XDATCAR(file_name, structure=None, time_step=None,
                      limit_number_steps=10000000,
                      last_steps=None,
                      initial_cut=1,
                      end_cut=None,
                      memmap=False,
                      template=None):

    # Time in picoseconds
    # Coordinates in Angstroms

    #Read environtment variables
    try:
        temp_directory = os.environ["DYNAPHOPY_TEMPDIR"]
        if os.path.isdir(temp_directory):
            print('Set temporal directory: {0}'.format(temp_directory))
            temp_directory += '/'
        else:
            temp_directory = ''
    except KeyError:
        temp_directory = ''

    number_of_atoms = None
    bounds = None

    #Check file exists
    if not os.path.isfile(file_name):
        print('Trajectory file does not exist!')
        exit()

    #Check time step
    if time_step is None:
        print('Warning! XDATCAR file does not contain time step information')
        print('Using default: 0.001 ps')
        time_step = 0.001

    #Starting reading
    print("Reading XDATCAR file")
    print("This may take long, please wait..")

    #Dimensionality of VASP calculation
    number_of_dimensions = 3

    time = []
    data = []
    counter = 0

    with open(file_name, "r+b") as f:

        file_map = mmap.mmap(f.fileno(), 0)

        #Read cell
        for i in range(2): file_map.readline()
        a = file_map.readline().split()
        b = file_map.readline().split()
        c = file_map.readline().split()
        super_cell = np.array([a, b, c], dtype='double')

        for i in range(1): file_map.readline()
        number_of_atoms = np.array(file_map.readline().split(), dtype=int).sum()

        while True:

            counter += 1
            #Read time steps
            position_number=file_map.find(b'Direct configuration')
            if position_number < 0: break

            file_map.seek(position_number)
            time.append(float(file_map.readline().split(b'=')[1]))

            if memmap:
                if end_cut:
                    data = np.memmap(temp_directory+'trajectory.{0}'.format(os.getpid()), dtype='complex', mode='w+', shape=(end_cut - initial_cut+1, number_of_atoms, number_of_dimensions))
                else:
                    print('Memory mapping requires to define reading range (use read_from/read_to option)')
                    exit()


            #Initial cut control
            if initial_cut > counter:
                continue

            #Reading coordinates
            read_coordinates = []
            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])

            read_coordinates = np.array(read_coordinates, dtype=float)  # in angstroms
            if template is not None:
                indexing = np.argsort(template)
                read_coordinates = read_coordinates[indexing, :]

            try:
                if memmap:
                    data[counter-initial_cut, :, :] = read_coordinates #in angstroms
                else:
                    data.append(read_coordinates) #in angstroms

            except ValueError:
                print("Error reading step {0}".format(counter))
                break
                # print(read_coordinates)

            #security routine to limit maximum of steps to read and put in memory
            if limit_number_steps+initial_cut < counter:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

            if end_cut is not None and end_cut <= counter:
                break

    file_map.close()

    time = np.array(time) * time_step

    if not memmap:
        data = np.array(data, dtype=complex)

        if last_steps is not None:
            data = data[-last_steps:, :, :]
            time = time[-last_steps:]


    return dyn.Dynamics(structure=structure,
                        scaled_trajectory=data,
                        time=time,
                        supercell=super_cell,
                        memmap=memmap)


if __name__ == "__main__":
    read_VASP_XDATCAR('/home/abel/VASP/MgO/MgO-FINAL/MgO_0.5_1600/No1/XDATCAR')
