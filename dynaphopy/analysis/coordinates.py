import numpy as np
import sys

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

def relativize_trajectory(dynamic):

    cell = dynamic.structure.get_cell()
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

            normalized_trajectory[i, j, :] -= np.dot(difference_matrix, cell) + position[j]

        progress_bar(float(j+1)/number_of_atoms)

    return normalized_trajectory

def trajectory_projection(dynamic, direction):

    direction = np.array(direction)
    super_cell = dynamic.get_super_cell_matrix()
    trajectory = dynamic.get_relative_trajectory()

    atom_type_index = dynamic.structure.get_atom_type_index(super_cell=super_cell)
    number_of_atom_types = dynamic.structure.get_number_of_atom_types()

    projections = []

    for j in range(number_of_atom_types):
        projection = np.array([])
        for i in range(0, trajectory.shape[1]):
            if atom_type_index[i] == j:
                projection = np.append(projection, np.dot(trajectory[:, i, :], direction/np.linalg.norm(direction)))

        projections.append(projection)

    return np.array(projections)
