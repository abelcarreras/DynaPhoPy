import numpy as np
import sys
from dynaphopy.displacements import atomic_displacement

def progress_bar(progress):
    bar_length = 30
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Progress error\r\n"
    if progress < 0:
        progress = 0
        status = "Halt ...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    text = "\rTrajectory: [{0}] {1:.2f}% {2}".format("#"*block + "-"*(bar_length-block),
                                                      progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

#print(disp.relative_trajectory(cell, traj ,pos))

def relativize_trajectory(dynamic):

    cell = dynamic.get_super_cell()
    number_of_atoms = dynamic.trajectory.shape[1]
    super_cell = dynamic.get_super_cell_matrix()
    position = dynamic.structure.get_positions(super_cell=super_cell)
#    normalized_trajectory = np.zeros_like(dynamic.trajectory.real)
    normalized_trajectory = dynamic.trajectory.copy()

    trajectory = dynamic.trajectory

#    progress_bar(0)

    for i in range(number_of_atoms):
        normalized_trajectory[:, i, :] = atomic_displacement(trajectory[:, i, :], position[i], cell)

   #     progress_bar(float(i+1)/number_of_atoms)
    return normalized_trajectory

def relativize_trajectory_py(dynamic):

    print('Using python rutine for calculating atomic displacements')
    cell = dynamic.get_super_cell()
    number_of_atoms = dynamic.trajectory.shape[1]
    super_cell = dynamic.get_super_cell_matrix()
    position = dynamic.structure.get_positions(super_cell=super_cell)
    normalized_trajectory = dynamic.trajectory.real.copy()

    progress_bar(0)

    for j in range(number_of_atoms):
        for i in range(0, normalized_trajectory.shape[0]):

            difference = normalized_trajectory[i, j, :] - position[j]

         #   difference_matrix = np.array(np.dot(np.linalg.inv(cell),(IniSep)),dtype=int)
            difference_matrix = np.around(np.dot(np.linalg.inv(cell), difference), decimals=0)
            normalized_trajectory[i, j, :] -= np.dot(difference_matrix, cell.T) + position[j]

        progress_bar(float(j+1)/number_of_atoms)

    return normalized_trajectory


def average_positions(dynamic, number_of_samples=8000):

    cell = dynamic.get_super_cell()
    number_of_atoms = dynamic.trajectory.shape[1]
    super_cell = dynamic.get_super_cell_matrix()
    position = dynamic.structure.get_positions(super_cell=super_cell)

    if dynamic.trajectory.shape[0] < number_of_samples:
        number_of_samples = dynamic.trajectory.shape[0]

    length = dynamic.trajectory.shape[0]
    positions = np.random.random_integers(length, size=(number_of_samples,))-1

    normalized_trajectory = dynamic.trajectory[positions, :]

    progress_bar(0)

    for j in range(number_of_atoms):
        for i in range(normalized_trajectory.shape[0]):

            difference = normalized_trajectory[i, j, :] - position[j]

            difference_matrix = np.around(np.dot(np.linalg.inv(cell), difference), decimals=0)
            normalized_trajectory[i, j, :] -= np.dot(difference_matrix, cell.T) + position[j]

        progress_bar(float(j+1)/number_of_atoms)

    reference = np.average(normalized_trajectory, axis=0)
    reference = reference+position

    for j in range(number_of_atoms):

        difference_matrix = np.around(np.dot(np.linalg.inv(cell), reference[j, :] - 0.5 * np.dot(np.ones((3)), cell.T)), decimals=0)
        print(difference_matrix)
        reference[j, :] -= np.dot(difference_matrix, cell.T)

    for i in reference:
        print '{0} {1} {2}'.format(*i.real)

    return normalized_trajectory


def trajectory_projection(dynamic, direction):

    direction = np.array(direction)/np.linalg.norm(direction)

    super_cell = dynamic.get_super_cell_matrix()
    trajectory = dynamic.get_relative_trajectory()

  #  print(trajectory)

    atom_type_index = dynamic.structure.get_atom_type_index(super_cell=super_cell)
    number_of_atom_types = dynamic.structure.get_number_of_atom_types()

    projections = []

    for j in range(number_of_atom_types):
        projection = np.array([])
        for i in range(0, trajectory.shape[1]):
            if atom_type_index[i] == j:
     #           print('atom:', i, 'type:', atom_type_index[i])
                projection = np.append(projection, np.dot(trajectory[:, i, :].real,
                                                          direction/np.linalg.norm(direction)))
        projections.append(projection)

    return np.array(projections)
