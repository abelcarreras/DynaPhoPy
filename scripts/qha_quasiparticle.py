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

parser.add_argument('-ref_vol', metavar='temperatures', type=float, default=None,
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

if '_mesh_phonopy' in input_parameters:
    mesh = input_parameters['_mesh_phonopy']
else:
    mesh =[20, 20, 20]

ev_data = np.loadtxt(args.ev)
volumes = ev_data[:, 0]
electronic_energies = ev_data[:, 1]

# Start main class

class ForceConstantsFitting():
    def __init__(self, structure, files_volume, files_temperature, temperatures, mesh=(20, 20, 20)):

        self.structure = structure
        self.files_volume = files_volume
        self.files_temperature = files_temperature
        self.supercell = structure.get_supercell_phonon()

        self._shift_temperature = None
        self._temperatures = temperatures
        self._shift_matrix = None

        self._eigenvectors = None
        self._mesh = mesh

    def get_temperature_range(self, step=10):
        return np.arange(300, 1700, step)


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

    def get_fitted_shifts_temperature(self, temperature, deg=2):

        shift_matrix = self.get_shift_matrix()

        if self._shift_temperature is None:
            p = []
            for i, row in enumerate(shift_matrix):
                p2 = []
                for j, r in enumerate(row):
                    p2.append(np.polyfit(self._temperatures, r, deg))
                p.append(p2)

            self._shift_temperature = p

        interpolated_shifts = []
        for p in self._shift_temperature:
            row = []
            for p2 in p:
                row.append(np.poly1d(p2)(temperature))
            interpolated_shifts.append(row)

        interpolated_shifts = np.array(interpolated_shifts).T

        return interpolated_shifts

    def plot_shifts(self, qpoint=0):
        import matplotlib.pyplot as plt

        shift_matrix = self.get_shift_matrix()
        print shift_matrix.shape

        chk_list = np.arange(self._temperatures[0], self._temperatures[-1], 10)
        chk_shift_matrix = np.array([self.get_fitted_shifts_temperature(t) for t in chk_list]).T
        #chk_shift_matrix = np.array([self.get_interpolated_shifts_temperature(t) for t in chk_list]).T

        plt.plot(chk_list, chk_shift_matrix[:, qpoint].T, '-')
        plt.plot(self._temperatures, shift_matrix[:, qpoint].T, 'o')
        plt.show()

    def get_eigenvectors(self):
        from dynaphopy.interface.phonopy_link import obtain_eigenvectors_and_frequencies

        if self._eigenvectors is None:
            data = self.get_data_temperature(0)
            com_ev = []
            for qpoint in data:
                arranged_ev, frequencies = obtain_eigenvectors_and_frequencies(self.structure, qpoint['reduced_wave_vector'],
                                                                               print_data=False)
                com_ev.append(arranged_ev)
            self._eigenvectors = com_ev
        return self._eigenvectors

    def get_data_temperature(self, index):
        with open(self.files_temperature[index], 'r') as stream:
            data = yaml.load(stream)
        return data

    def get_data_volume(self, index):
        with open(self.files_volume[index], 'r') as stream:
            data = yaml.load(stream)
        return data

    def _get_renormalized_force_constants(self, renormalized_frequencies):
        eigenvectors = self.get_eigenvectors()
        from dynaphopy.interface.phonopy_link import get_renormalized_force_constants
        fc = get_renormalized_force_constants(renormalized_frequencies,
                                              eigenvectors, self.structure,
                                              self.supercell,
                                              symmetrize=False)
        return fc

    def get_total_force_constants(self, temperature=300, volume_index=0):
        #temperature_shifts = self.get_interpolated_shifts_temperature(temperature)
        temperature_shifts = self.get_fitted_shifts_temperature(temperature)

        volume_frequencies = np.array([qpoint['frequencies'] for qpoint in self.get_data_volume(volume_index)])

        total_frequency = volume_frequencies + temperature_shifts

        return self._get_renormalized_force_constants(total_frequency)

    def get_thermal_properties(self, volume_index=0):
        from dynaphopy.interface.phonopy_link import obtain_phonopy_thermal_properties

        free_energy_list = []
        entropy_list = []
        cv_list = []
        for t in self.get_temperature_range():
            fc = self.get_total_force_constants(temperature=t, volume_index=volume_index)

            free_energy, entropy, cv = obtain_phonopy_thermal_properties(self.structure,
                                                                         temperature=t,
                                                                         mesh=self._mesh,
                                                                         force_constants=fc)
            print ('t: {} , {} {} {}'.format(t, free_energy, entropy, cv))
            free_energy_list.append(free_energy)
            entropy_list.append(entropy)
            cv_list.append(cv)

        return free_energy_list, entropy_list, cv_list

    def plot_density_of_states(self, volume_index=0, temperature=300):
        from dynaphopy.interface.phonopy_link import obtain_phonopy_dos
        import matplotlib.pyplot as plt
        fc = self.get_total_force_constants(temperature=temperature, volume_index=volume_index)
        dos = obtain_phonopy_dos(self.structure, mesh=self._mesh, force_constants=fc)
        plt.plot(dos[0], dos[1])
        plt.show()

    def plot_thermal_properties(self, volume_index=0):
        import matplotlib.pyplot as plt
        free_energy, entropy, cv = fc_fit.get_thermal_properties(volume_index=volume_index)
        temperature = self.get_temperature_range()
        plt.plot(temperature, free_energy, label='free energy')
        plt.plot(temperature, entropy, label='entropy')
        plt.plot(temperature, cv, label='Cv')
        plt.grid()
        plt.legend()
        plt.show()

fc_fit = ForceConstantsFitting(structure,
                               files_temperature=args.ct_data,
                               files_volume=args.cv_data,
                               temperatures=args.temperatures,
                               mesh=mesh)

fc_fit.plot_shifts(qpoint=1)
fc_fit.plot_density_of_states()
#print fc_fit.get_eigenvectors()
fc_fit.plot_thermal_properties(volume_index=3)


temperatures = fc_fit.get_temperature_range()

free_energy = []
entropy = []
cv = []

for i in range(len(volumes)):
    tp_data = fc_fit.get_thermal_properties(volume_index=i)
    free_energy.append(tp_data[0])
    entropy.append(tp_data[1])
    cv.append(tp_data[2])

free_energy = np.array(free_energy).T
entropy = np.array(entropy).T
cv = np.array(cv).T


phonopy_qha = PhonopyQHA(np.array(volumes),
                         np.array(electronic_energies),
                         eos="vinet",
                         temperatures=np.array(temperatures),
                         free_energy=np.array(free_energy),
                         cv=np.array(cv),
                         entropy=np.array(entropy),
                         t_max=fc_fit.get_temperature_range()[-1],
                         verbose=False)

phonopy_qha.plot_qha().show()
phonopy_qha.plot_bulk_modulus().show()
phonopy_qha.plot_gibbs_temperature().show()
phonopy_qha.plot_heat_capacity_P_numerical().show()
phonopy_qha.plot_helmholtz_volume().show()

# Get data
qha_temperatures = phonopy_qha._qha._temperatures[:phonopy_qha._qha._max_t_index]
helmholtz_volume = phonopy_qha.get_helmholtz_volume()
thermal_expansion = phonopy_qha.get_thermal_expansion()
volume_temperature = phonopy_qha.get_volume_temperature()
heat_capacity_P_numerical = phonopy_qha.get_heat_capacity_P_numerical()
volume_expansion = phonopy_qha.get_volume_expansion()
gibbs_temperature = phonopy_qha.get_gibbs_temperature()



