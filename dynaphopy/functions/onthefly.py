import numpy as np
import mmap
import os
import gc

import dynaphopy.classes.dynamics as dyn
import dynaphopy.functions.projection as projection

def read_lammps_trajectory_direct(file_name, structure, time_step=None, reduced_q_vector=(0,0,0)):

    limit_number_steps = 0

    reduced_q_vector = np.array(reduced_q_vector)
    q_vector = np.dot(reduced_q_vector,2.0*np.pi*np.linalg.inv(structure.get_primitive_cell()))

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
    print("Reading LAMMPS trajectory (direct mode)")
    print("This could take really long, please wait..")

    #Dimensionality of LAMMP calculation
    number_of_dimensions = 3

    time = []
    trajectory = []
    vc = []

    with open(file_name, "r+") as f:


        file_map = mmap.mmap(f.fileno(), 0)

        while True:

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

                super_cell = np.array([[bounds[0, 1] - bounds[0, 0], 0,                           0],
                                       [bounds[0, 2],                bounds[1, 1] - bounds[1, 0], 0],
                                       [bounds[1, 2],                bounds[2, 2],                bounds[2, 1] - bounds[2, 0]]])


            position_number = file_map.find('ITEM: ATOMS')

            file_map.seek(position_number)
            file_map.readline()

            read_coordinates = []

            for i in range (number_of_atoms):
                read_coordinates.append(file_map.readline().split()[0:number_of_dimensions])

            trajectory.append(np.array(read_coordinates, dtype=float)) #in angstroms

            #On the fly calculation and projection
            last = 20

            if len(trajectory) > last:
                time_temp = time[-last:]
                trajectory_temp = trajectory[-last:]

                trajectory_temp = np.array(trajectory_temp, dtype=complex)
                time_temp = np.array(time_temp) * time_step

          #      print(trajectory_temp)
          #      print(time_temp)

                temp_dynamic = dyn.Dynamics(structure = structure,
                                trajectory=trajectory_temp,
                                time=time_temp,
                                super_cell=super_cell)

  #              print(temp_dynamic.get_velocity_mass_average())


#                vc.append(projection.project_onto_wave_vector(temp_dynamic, q_vector)[-10])
                projection.project_onto_wave_vector(temp_dynamic, q_vector)

            limit_number_steps += 1
            if limit_number_steps > 1000:
                print("Warning! maximum number of steps reached! No more steps will be read")
                break

    file_map.close()

    return np.array(vc), temp_dynamic



