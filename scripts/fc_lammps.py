#!/usr/bin/env python

import numpy as np
import argparse

import dynaphopy.interface.iofile as reading
import dynaphopy.interface.phonopy_link as pho_interface
from lammps import lammps
from phonopy.file_IO import write_FORCE_CONSTANTS
from dynaphopy.interface.iofile import get_correct_arrangement
from dynaphopy.interface.phonopy_link import get_phonon, ForceConstants

parser = argparse.ArgumentParser(description='rfc_calc options')
parser.add_argument('input_file', metavar='data_file', type=str,
                    help='input file containing structure related data')

parser.add_argument('lammps_input_file', metavar='lammps_file', type=str,
                    help='input file containing lammps script')

parser.add_argument('-o', metavar='file', type=str, default='FORCE_CONSTANTS',
                    help='force constants output file [default: FORCE_CONSTANTS]')

parser.add_argument('-p', action='store_true',
                   help='plot phonon band structure')


args = parser.parse_args()


def get_cells_with_displacements(structure, displacement_distance=0.01):
    """
    generate the cells with displacements using phonopy

    :param structure:  dynaphopy type Structure object
    :param displacement_distance: displacement distance in Angstroms
    :return: data_sets phonopy dictionary (without forces) and list of numpy arrays containing
    the atoms positions of supecells with displacements with shape [Natoms x 3]
    """

    phonon = get_phonon(structure, setup_forces=False,
                        custom_supercell=structure.get_supercell_matrix())
    phonon.get_displacement_dataset()
    phonon.generate_displacements(distance=displacement_distance)
    cells_with_disp = phonon.get_supercells_with_displacements()

    cells_with_disp = [cell.get_positions() for cell in cells_with_disp]

    data_sets = phonon.get_displacement_dataset()

    return data_sets, cells_with_disp


def get_force_constants(structure, data_sets):
    """
    calculate force constants from structure and data_sets using phonopy (synaphopy interface)
    :param structure: dynaphopy Structure type object
    :param data_sets: phonopy data_sets dictionary (with forces)
    :return: force_constants matrix [3Natoms x 3Natoms]
    """

    phonon = get_phonon(structure, setup_forces=False,
                        custom_supercell=structure.get_supercell_matrix())

    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()
    force_constants = phonon.get_force_constants()

    return force_constants


def get_lammps_forces(structure, cell_with_disp, input_file):
    """
    Calculate the forces of a supercell using lammps
    :param structure: unit cell
    :param cell_with_disp: supercell from which determine the forces
    :param input_file: input file name of lammps script containg potential information
    :return: numpy array matrix with forces of atoms [Natoms x 3]
    """

    lmp = lammps(cmdargs=['-echo','none', '-log', 'none', '-screen', 'none'])

    supercell = np.diag(structure.get_supercell_matrix())

    lmp.file(input_file)
    lmp.command('replicate {} {} {}'.format(*supercell))
    lmp.command('run 0')

    xc = lmp.gather_atoms("x", 1, 3)
    reference = np.array(xc).reshape((-1, 3))
    template = get_correct_arrangement(reference, structure)
    indexing = np.argsort(template)

    na = lmp.get_natoms()
    for i in range(na):
        lmp.command('set atom {} x {} y {} z {}'.format(i+1,
                                                        cell_with_disp[template[i], 0],
                                                        cell_with_disp[template[i], 1],
                                                        cell_with_disp[template[i], 2]))

    lmp.command('run 0')

    forces = lmp.gather_atoms("f", 1, 3)

    forces = np.array(forces).reshape((-1,3))[indexing, :]

    lmp.close()

    return forces


def plot_phonon_dispersion_bands(structure, force_constants):
    """
    Plot phonon band structure using seekpath automatic k-path

    :param structure: Dynaphopy type Structure object
    :param force_constants:  Dynaphopy type ForceConstants Object
    :return:
    """

    import matplotlib.pyplot as plt

    bands_and_labels = structure.get_path_using_seek_path()
    _bands = pho_interface.obtain_phonon_dispersion_bands(structure,
                                                          bands_and_labels['ranges'],
                                                          force_constants=ForceConstants(force_constants,
                                                          supercell=structure.get_supercell_matrix()))

    for i, freq in enumerate(_bands[1]):
        plt.plot(_bands[1][i], _bands[2][i], color='r')

        # plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_xaxis().set_ticks([])

    plt.ylabel('Frequency [THz]')
    plt.xlabel('Wave vector')
    plt.xlim([0, _bands[1][-1][-1]])
    plt.axhline(y=0, color='k', ls='dashed')
    plt.suptitle('Phonon dispersion')

    if 'labels' in bands_and_labels:
        plt.rcParams.update({'mathtext.default': 'regular'})

        labels = bands_and_labels['labels']

        labels_e = []
        x_labels = []
        for i, freq in enumerate(_bands[1]):
            if labels[i][0] == labels[i - 1][1]:
                labels_e.append('$' + labels[i][0].replace('GAMMA', '\Gamma') + '$')
            else:
                labels_e.append(
                    '$' + labels[i - 1][1].replace('GAMMA', '\Gamma') + '/' + labels[i][0].replace('GAMMA',
                                                                                                   '\Gamma') + '$')
            x_labels.append(_bands[1][i][0])
        x_labels.append(_bands[1][-1][-1])
        labels_e.append('$' + labels[-1][1].replace('GAMMA', '\Gamma') + '$')
        labels_e[0] = '$' + labels[0][0].replace('GAMMA', '\Gamma') + '$'

        plt.xticks(x_labels, labels_e, rotation='horizontal')

    plt.show()


# Get data from input file & process parameters
input_parameters = reading.read_parameters_from_input_file(args.input_file)
if 'structure_file_name_outcar' in input_parameters:
    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name_outcar'])
else:
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])
structure.get_data_from_dict(input_parameters)

# Alternative use of supercell_matrix for convenience
structure.set_supercell_matrix(input_parameters['supercell_phonon'])

# Get cells with displacements
data_sets, disp_cells = get_cells_with_displacements(structure,
                                                     displacement_distance=0.01)

# Get forces from lammps
for i, cell in enumerate(disp_cells):
    print ('displacement {} / {}'.format(i+1, len(disp_cells)))
    force = get_lammps_forces(structure, cell, args.lammps_input_file)
    data_sets['first_atoms'][i]['forces'] = force

# Get force constants from phonopy
force_constants = get_force_constants(structure, data_sets)

# Plot data if requested
if args.p:
    plot_phonon_dispersion_bands(structure, force_constants)

# Write force constants in file
print('writing force constants')
write_FORCE_CONSTANTS(force_constants, filename=args.o)
