#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from phonopy import PhonopyQHA
import argparse
import dynaphopy.interface.iofile as reading


parser = argparse.ArgumentParser(description='qha_quasiparticles options')
parser.add_argument('input_file', metavar='data_file', type=str, nargs=1,
                    help='input file containing structure related data')

parser.add_argument('-cv_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant volume')

parser.add_argument('-ct_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant temperature')


parser.add_argument('-volumes', metavar='volumes', type=str, nargs='*', required=True,
                    help='list of volumes')

parser.add_argument('-temperatures', metavar='temperatures', type=str, nargs='*', required=True,
                    help='list of temperatures')

parser.add_argument('-temp_at_vol', metavar='temperatures', type=float,
                    help='temperature at the volumes')

parser.add_argument('-ev', metavar='data', type=str, required=True,
                    help='Energy volume file')

args = parser.parse_args()

# Get data from input file & process parameters
input_parameters = reading.read_parameters_from_input_file(args.input_file[0])

if 'structure_file_name_outcar' in input_parameters:
    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name_outcar'])
else:
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])

structure.get_data_from_dict(input_parameters)




# Apply QHA using phonopy
phonopy_qha = PhonopyQHA(np.array(volumes),
                         np.array(electronic_energies),
                         eos="vinet",
                         temperatures=np.array(temperatures),
                         free_energy=np.array(fe_phonon),
                         cv=np.array(cv),
                         entropy=np.array(entropy),
                         #t_max=target_temperature,
                         verbose=False)
