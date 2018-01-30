#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from phonopy import PhonopyQHA
import argparse
import dynaphopy.interface.iofile as reading
import yaml
from dynaphopy.interface.phonopy_link import get_force_sets_from_file, get_force_constants_from_file


parser = argparse.ArgumentParser(description='qha_quasiparticles options')
parser.add_argument('input_file', metavar='data_file', type=str,
                    help='input file containing structure related data')

parser.add_argument('-cv_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant volume')

parser.add_argument('-ct_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant temperature')

parser.add_argument('-temperatures', metavar='temperatures', type=float, nargs='*', required=True,
                    help='list of temperatures')

parser.add_argument('-temp_at_vol', metavar='temperatures', type=float, default=None,
                    help='temperature at the volumes')

parser.add_argument('-ev', metavar='data', type=str, required=True,
                    help='Energy volume file')

args = parser.parse_args()

# Get data from input file & process parameters
input_parameters = reading.read_parameters_from_input_file(args.input_file)

if 'structure_file_name_outcar' in input_parameters:
    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name_outcar'])
else:
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])

structure.get_data_from_dict(input_parameters)

if 'supercell_phonon' in input_parameters:
    supercell_phonon = input_parameters['supercell_phonon']
else:
    supercell_phonon = np.identity(3)

structure.set_force_constants(get_force_constants_from_file(file_name=input_parameters['force_constants_file_name'],
                                                            fc_supercell=supercell_phonon))

# Start script here ##########################################

# Harmonic
volumes_harmonic = args.ct_data[0]

with open(args.cv_data[0], 'r') as stream:
    data = yaml.load(stream)
print data

from dynaphopy.interface.phonopy_link import get_renormalized_force_constants, obtain_eigenvectors_and_frequencies, get_commensurate_points

com_points = [d['reduced_wave_vector'] for d in data]

com_ev = []
renormalized_frequencies = []

for qpoint in data:
    arranged_ev, frequencies = obtain_eigenvectors_and_frequencies(structure, qpoint['reduced_wave_vector'], print_data=True)
    com_ev.append(arranged_ev)
    renormalized_frequencies.append(qpoint['frequencies'])

print renormalized_frequencies

fc = get_renormalized_force_constants(renormalized_frequencies, com_ev, structure, supercell_phonon, symmetrize=False)


class Fc_fit():
    def __init__(self, files_volume, files_temperature, temperatures):
        self.files_volume = files_volume
        self.files_temperature = files_temperature

        self._shift_temperature = None
        self._temperatures = temperatures
        self._shift_matrix = None

    def get_ref_data(self):
        with open(args.cv_data[0], 'r') as stream:
            ref_data = yaml.load(stream)
        return ref_data

    def get_shift_matrix(self):

        if self._shift_matrix is None:
            ref_data = self.get_ref_data()

            shift_matrix = []
            list_t = self._temperatures
            for i, t in enumerate(list_t):
                print i, t
                with open(self.files_temperature[i], 'r') as stream:
                    data = yaml.load(stream)

                renormalized_frequencies = []
                reference_frequencies = []

                for qpoint, ref_qpoint in zip(data, ref_data):
                    renormalized_frequencies.append(qpoint['frequencies'])
                    reference_frequencies.append(ref_qpoint['frequencies'])

                shift_matrix.append(np.array(renormalized_frequencies) - np.array(reference_frequencies))

            self._shift_matrix = np.array(shift_matrix).swapaxes(0, 2)

        return self._shift_matrix

    def get_interpolated_shifts_temperature(self, temperature, kind='quadratic'):
        from scipy.interpolate import interp1d

        shift_matrix = self.get_shift_matrix()
        if self._shift_temperature is None:
            self._shift_temperature = interp1d(self._temperatures, shift_matrix, kind=kind)

        interpolated_shifts = self._shift_temperature(temperature).T

        return interpolated_shifts

    def plot_shifts(self, qpoint=0):
        import matplotlib.pyplot as plt

        shift_matrix = self.get_shift_matrix()
        print shift_matrix.shape

        chk_list = np.arange(self._temperatures[0], self._temperatures[-1], 10)
        chk_shift_matrix = np.array([self.get_interpolated_shifts_temperature(t) for t in chk_list]).T


        plt.plot(chk_list, chk_shift_matrix[:, qpoint].T, '-')
        plt.plot(self._temperatures, shift_matrix[:, qpoint].T, 'o')
        plt.show()


fc_fit = Fc_fit(files_temperature=args.ct_data,
                files_volume=args.cv_data,
                temperatures=args.temperatures)

fc_fit.plot_shifts(qpoint=0)


exit()



PRESS = -40
TEMP = 900.0



# At reference volume (P=0) (at T = 900)

# Harmonic data
structure_h, force_constants_h = get_data_from_workflow(wf, temperature=0, pressure=0)

list_t = wf.get_parameter('scan_temperatures')
shift_matrix = []
for t in list_t:
    structure_h, force_constants_r = get_data_from_workflow(wf, temperature=t, pressure=0)

    inline_params = {'structure': structure_h,
                     'phonopy_input': parameters['phonopy_input'],
                     'force_constants': force_constants_h,
                     'r_force_constants': force_constants_r}

    shifts = phonopy_commensurate_shifts_inline(**inline_params)
    shift_matrix.append(shifts['commensurate'].get_array('shifts'))
    qpoints = shifts['commensurate'].get_array('qpoints')

shift_matrix = np.array(shift_matrix).swapaxes(0,2)

from scipy.interpolate import interp1d
f_temperature = interp1d(list_t, shift_matrix, kind='quadratic')

# Here we set temperature (T=900)
interpolated_shifts = f_temperature(TEMP).T

shifts = ArrayData()
shifts.set_array('qpoints', qpoints)
shifts.set_array('shifts', interpolated_shifts)

shifts = {'commensurate' : shifts}
print interpolated_shifts.shape



# PHONON CALCULATIONS

# Estimated  (T=0, P=50) + temperature shifts (T=900)

structure_p, force_constants_p = get_data_from_workflow(wf, temperature=0, pressure=PRESS)

inline_params = {'structure': structure_p,
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': force_constants_p}

harmonic_p = phonopy_commensurate_inline(**inline_params)

inline_params = {'structure': structure_p,
                 'phonopy_input': parameters['phonopy_input'],
                 'harmonic': harmonic_p,
                 'renormalized': shifts}

estimate_force_constants = phonopy_merge(**inline_params)['final_results']


inline_params = {'structure': structure_p,
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': estimate_force_constants}

results_estimate = phonopy_calculation_inline(**inline_params)


print 'results_estimate done'

# Initial QHA

inline_params = {'structure': structure_p,
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': force_constants_p}

results_qha = phonopy_calculation_inline(**inline_params)

print 'results_qha done'



# Real total
# Workflow phonon (at given volume and temperature) (P=50) (at T = 900)
structure_t, force_constants_t = get_data_from_workflow(wf, temperature=900, pressure=PRESS)

inline_params = {'structure': structure_t,
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': force_constants_t}

results_total = phonopy_calculation_inline(**inline_params)

print 'results_total done'



# Reference harmonic
# Workflow phonon (at given volume and temperature) (P=0) (at T = 0)

inline_params = {'structure': structure_h,
                 'phonopy_input': parameters['phonopy_input'],
                 'force_constants': force_constants_h}

results_h = phonopy_calculation_inline(**inline_params)

print 'results_total done'


# Phonon Band structure plot

plt = plot_data(results_h['band_structure'], color='g')
#plt = plot_data(results_qha['band_structure'], color='g')
plt = plot_data(results_estimate['band_structure'], color='b')
plt = plot_data(results_total['band_structure'], color='r')
plt.legend()

plt = plot_dos(results_h['dos'], color='g')
#plt = plot_dos(results_qha['dos'], color='g')
plt = plot_dos(results_estimate['dos'], color='b')
plt = plot_dos(results_total['dos'], color='r')

plt.show()




# QHA (on development)



phonopy_qha = PhonopyQHA(np.array(volumes),
                         np.array(electronic_energies),
                         eos="vinet",
                         temperatures=np.array(temperatures),
                         free_energy=np.array(fe_phonon),
                         cv=np.array(cv),
                         entropy=np.array(entropy),
                         # t_max=options.t_max,
                         verbose=False)

# Get data
qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
helmholtz_volume = phonopy_qha.get_helmholtz_volume()
thermal_expansion = phonopy_qha.get_thermal_expansion()
volume_temperature = phonopy_qha.get_volume_temperature()
heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
volume_expansion = phonopy_qha.get_volume_expansion()
gibbs_temperature = phonopy_qha.get_gibbs_temperature()




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
