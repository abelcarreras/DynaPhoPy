#!/usr/bin/env python
import argparse
import numpy as np

import phonopy.file_IO as file_IO
import dynaphopy.functions.iofunctions as reading
import dynaphopy.classes.controller as controller
import dynaphopy.functions.interactive_ui as interactive_ui

#Define arguments
parser = argparse.ArgumentParser(description='DynaPhonoPy options')
parser.add_argument('input_file', metavar='data_file', type=str, nargs=1,
                   help='input file containing structure related data')

parser.add_argument('md_file', metavar='OUTCAR', type=str, nargs=1,
                   help='VASP Output file containing MD calculation (Super cell)')

parser.add_argument('-i', '--interactive', action='store_true',
                    help='enter interactive mode')

parser.add_argument('-q', metavar='N', type=float, nargs=3,
                   help='wave vector used to calculate projections (default: 0 0 0)')

parser.add_argument('-r', '--frequency_range',metavar='N', type=float, nargs=3,
                   help='Frequency range for correlation function calculation (min, max, number of points)')

parser.add_argument('-n', metavar='N', type=int, default=2000,
                   help='number of MD last steps to take (default: 2000)')

parser.add_argument('-pw', '--plot_wave_vector', action='store_true',
                    help='Plot projection into wave vector')

parser.add_argument('-pp', '--plot_phonon_mode', action='store_true',
                    help='plot projection into phonon modes')

parser.add_argument('-sw', '--save_wave_vector', metavar='file', type=str, nargs=1,
                    help='save projection into wave vector to file')

parser.add_argument('-sp', '--save_phonon_mode', metavar='file', type=str, nargs=1,
                    help='save projection into phonon modes to file')

parser.add_argument('-sv', '--save_velocity', metavar='file', type=str, nargs=1,
                    help='save velocity into hdf5 file')

parser.add_argument('-lv', '--load_velocity', metavar='file', type=str, nargs=1,
                    help='load velocity from hdf5 file')

#Under development
parser.add_argument('-psm', '--power_spectrum_algorithm',metavar='N', type=int, nargs=1,
                   help='Select power spectrum calculation algorithm (default 0)')

parser.add_argument('-cf', '--number_of_mem_coefficients',metavar='N', type=int, nargs=1,
                   help='Number of coefficients to use in MEM algorithm (default 300)')

parser.add_argument('-pa', '--coefficient_scan_analysis', action='store_true',
                    help='Request coefficient scan analysis (MEM only)')

parser.add_argument('-pi', '--peak_analysis', action='store_true',
                    help='Request a peak analysis (MEM only)')

parser.add_argument('-ds', '--display_spectrum', action='store_true',
                    help='display phonon dispersion spectrum data in screen (just for test: not recommended)')

args = parser.parse_args()


#Get data from input file
input_parameters = reading.read_parameters_from_input_file(args.input_file[0])

if 'structure_file_name_outcar' in input_parameters:
    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name_outcar'])
    structure_file = input_parameters['structure_file_name_outcar']
else:
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])
    structure_file = input_parameters['structure_file_name_poscar']

if 'force_constants_file_name' in input_parameters:
    structure.set_force_set( file_IO.parse_FORCE_SETS(filename=input_parameters['force_constants_file_name']))

if 'primitive_matrix' in input_parameters:
    structure.set_primitive_matrix(input_parameters['primitive_matrix'])

if 'super_cell_matrix' in input_parameters:
    structure.set_super_cell_phonon(input_parameters['super_cell_matrix'])

#trajectory = reading.read_from_file_trajectory(args.md_file[0],structure,last_steps=args.n)

if args.load_velocity:
    trajectory = reading.initialize_from_file(args.load_velocity[0],structure)
    #calculation.read_velocity(args.load_velocity[0])
else:
    trajectory = reading.read_from_file_trajectory(args.md_file[0],structure,last_steps=args.n)

calculation = controller.Calculation(trajectory)

#print(calculation.get_vc())

if 'bands' in input_parameters:
    calculation.set_band_ranges(input_parameters['bands'])

#Process properties arguments

if args.q:
    calculation.set_reduced_q_vector(np.array(args.q))

if args.power_spectrum_algorithm:
    calculation.select_power_spectra_algorithm(args.power_spectrum_algorithm[0])

if args.number_of_mem_coefficients:
    calculation.set_number_of_mem_coefficients (args.number_of_mem_coefficients[0])

#Process calculation arguments

if args.save_velocity:
    calculation.save_velocity(args.save_velocity[0])

if args.frequency_range:
    calculation.set_frequency_range(np.linspace(*args.frequency_range))

if args.save_wave_vector:
    calculation.write_correlation_wave_vector(args.save_wave_vector[0])

if args.save_phonon_mode:
    calculation.write_correlation_phonon(args.save_phonon_mode[0])

if args.plot_wave_vector:
    calculation.plot_correlation_wave_vector()

if args.plot_phonon_mode:
    calculation.plot_correlation_phonon()

if args.interactive:
    interactive_ui.interactive_interface(calculation, trajectory, args, structure_file)

#On development methods (not recommended)
if args.display_spectrum:
    calculation.print_phonon_dispersion_spectrum()

if args.coefficient_scan_analysis:
    calculation.phonon_width_scan_analysis()

if args.peak_analysis:
    calculation.phonon_width_individual_analysis()
