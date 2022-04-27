import numpy as np
import sys
from dynaphopy.displacements import atomic_displacements


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


# Not used (only for test)
def relativize_trajectory(dynamic, memmap=False):

    cell = dynamic.get_supercell()
    number_of_atoms = dynamic.trajectory.shape[1]
    supercell = dynamic.get_supercell_matrix()
    position = dynamic.structure.get_positions(supercell=supercell)
    # normalized_trajectory = np.zeros_like(dynamic.trajectory.real)
    normalized_trajectory = dynamic.trajectory.copy()

    trajectory = dynamic.trajectory

    if memmap:
        normalized_trajectory = np.memmap('r_trajectory.map', dtype='complex', mode='w+', shape=trajectory.shape)
    else:
        normalized_trajectory = dynamic.trajectory.copy()

    # progress_bar(0)

    for i in range(number_of_atoms):
        normalized_trajectory[:, i, :] = atomic_displacements(trajectory[:, i, :], position[i], cell)

   # progress_bar(float(i+1)/number_of_atoms)
    return normalized_trajectory

# Not used (only for test)
def relativize_trajectory_py(dynamic):

    print('Using python rutine for calculating atomic displacements')
    cell = dynamic.get_supercell()
    number_of_atoms = dynamic.trajectory.shape[1]
    supercell = dynamic.get_supercell_matrix()
    position = dynamic.structure.get_positions(supercell=supercell)
    normalized_trajectory = dynamic.trajectory.real.copy()

    progress_bar(0)

    for j in range(number_of_atoms):
        for i in range(0, normalized_trajectory.shape[0]):

            difference = normalized_trajectory[i, j, :] - position[j]

            # difference_matrix = np.array(np.dot(np.linalg.inv(cell),(IniSep)),dtype=int)
            difference_matrix = np.around(np.dot(np.linalg.inv(cell), difference), decimals=0)
            normalized_trajectory[i, j, :] -= np.dot(difference_matrix, cell) + position[j]

        progress_bar(float(j+1)/number_of_atoms)

    return normalized_trajectory


def trajectory_projection(dynamic, direction):

    direction = np.array(direction)/np.linalg.norm(direction)

    supercell = dynamic.get_supercell_matrix()
    trajectory = dynamic.get_relative_trajectory()
    atom_type_index = dynamic.structure.get_atom_type_index(supercell=supercell)
    number_of_atom_types = dynamic.structure.get_number_of_atom_types()

    projections = []

    for j in range(number_of_atom_types):
        projection = np.array([])
        for i in range(0, trajectory.shape[1]):
            if atom_type_index[i] == j:
                # print('atom:', i, 'type:', atom_type_index[i])
                projection = np.append(projection, np.dot(trajectory[:, i, :].real,
                                                          direction/np.linalg.norm(direction)))
        projections.append(projection)

    return np.array(projections)
