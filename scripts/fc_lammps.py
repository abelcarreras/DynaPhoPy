#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse

import dynaphopy.interface.phonopy_link as pho_interface
import dynaphopy.interface.iofile as reading
from lammps import lammps
from dynaphopy.interface.iofile import get_correct_arrangement
from phonopy.file_IO import write_FORCE_CONSTANTS


parser = argparse.ArgumentParser(description='rfc_calc options')
parser.add_argument('input_file', metavar='data_file', type=str,
                    help='input file containing structure related data')

parser.add_argument('lammps_input_file', metavar='lammps_file', type=str,
                    help='input file containing lammps script')

parser.add_argument('-o', metavar='file', type=str, default='FORCE_CONSTANTS',
                    help='force constants output file [default: FORCE_CONSTANTS]')

args = parser.parse_args()


def get_cells_with_displacements(structure, displacement_distance=0.01):

    from dynaphopy.interface.phonopy_link import get_phonon

    phonon = get_phonon(structure, setup_forces=False,
                        custom_supercell=structure.get_supercell_matrix())
    phonon.get_displacement_dataset()
    phonon.generate_displacements(distance=displacement_distance)
    cells_with_disp = phonon.get_supercells_with_displacements()

    cells_with_disp = [cell.get_positions() for cell in cells_with_disp]

    data_sets = phonon.get_displacement_dataset()

    return data_sets, cells_with_disp


def get_force_constants(structure, data_sets):

    print 'force_constants'
    from dynaphopy.interface.phonopy_link import get_phonon

    phonon = get_phonon(structure, setup_forces=False,
                        custom_supercell=structure.get_supercell_matrix())

    phonon.set_displacement_dataset(data_sets)
    phonon.produce_force_constants()
    force_constants = phonon.get_force_constants()

    return force_constants


def get_lammps_forces(structure, cell_with_disp, input_file):

    lmp = lammps(cmdargs=['-echo','none', '-log', 'none', '-screen', 'none'])

    supercell = np.diag(structure.get_supercell_matrix())

    lmp.file(input_file)
    lmp.command('replicate {} {} {}'.format(*supercell))
    lmp.command('run 0')

    xlo =lmp.extract_global("boxxlo", 1)
    xhi =lmp.extract_global("boxxhi", 1)
    ylo =lmp.extract_global("boxylo", 1)
    yhi =lmp.extract_global("boxyhi", 1)
    zlo =lmp.extract_global("boxzlo", 1)
    zhi =lmp.extract_global("boxzhi", 1)
    xy =lmp.extract_global("xy", 1)
    yz =lmp.extract_global("yz", 1)
    xz =lmp.extract_global("xz", 1)

    simulation_cell = np.array([[xhi-xlo, xy,  xz],
                           [0,  yhi-ylo,  yz],
                           [0,   0,  zhi-zlo]]).T

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

# Get data from input file & process parameters
input_parameters = reading.read_parameters_from_input_file(args.input_file)

if 'structure_file_name_outcar' in input_parameters:
    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name_outcar'])
else:
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])

structure.get_data_from_dict(input_parameters)
structure.set_supercell_matrix(input_parameters['supercell_phonon'])

data_sets, disp_cells = get_cells_with_displacements(structure,
                                          displacement_distance=0.01)

for i, cell in enumerate(disp_cells):
    print ('displacement {} / {}'.format(i+1, len(disp_cells)))
    force = get_lammps_forces(structure, cell, args.lammps_input_file)

    data_sets['first_atoms'][i]['forces'] = force

force_constants = get_force_constants(structure, data_sets)

write_FORCE_CONSTANTS(force_constants, filename=args.o)
