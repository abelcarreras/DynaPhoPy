#!/usr/bin/env python

import numpy as np
from phonopy import PhonopyQHA
import argparse
import dynaphopy.interface.iofile as reading
import yaml
from dynaphopy.interface.phonopy_link import get_force_constants_from_file


parser = argparse.ArgumentParser(description='qha_quasiparticles options')
parser.add_argument('input_file', metavar='data_file', type=str,
                    help='input file containing structure related data')

parser.add_argument('-cv_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant volume')

parser.add_argument('-ct_data', metavar='data_files', type=str, nargs='*', required=True,
                    help='quasiparticle data at constant temperature')

parser.add_argument('-temperatures', metavar='temperatures', type=float, nargs='*', required=True,
                    help='list of temperatures')

parser.add_argument('--ref_temperature', metavar='temperature', type=float, default=None,
                    help='Temperature at volume expansion')

parser.add_argument('--order', metavar='order', type=int, default=2,
                    help='fitting polynomial order for frequency shifts')

parser.add_argument('-ev', metavar='data', type=str, required=True,
                    help='Energy volume file')

args = parser.parse_args()


class ForceConstantsFitting():
    def __init__(self, structure, files_volume, files_temperature, temperatures,
                 mesh=(40, 40, 40), ref_index=0, fitting_order=2):

        self.structure = structure
        self.files_volume = files_volume
        self.files_temperature = files_temperature
        self.supercell = structure.get_supercell_phonon()

        self._shift_temperature = None
        self._temperatures = temperatures
        self._shift_matrix = None

        self._eigenvectors = None
        self._mesh = mesh
        self.ref_index = ref_index
        self.fitting_order = fitting_order

    def get_temperature_range(self, step=10):
        return np.arange(self._temperatures[0], self._temperatures[-1], step)

    def get_ref_data(self):
        with open(args.cv_data[self.ref_index], 'r') as stream:
            ref_data = yaml.load(stream)
        return ref_data

    def get_shift_matrix(self):

        if self._shift_matrix is None:
            ref_data = self.get_ref_data()

            shift_matrix = []
            list_t = self._temperatures
            for i, t in enumerate(list_t):
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

    def get_fitted_shifts_temperature(self, temperature):

        shift_matrix = self.get_shift_matrix()

        if self._shift_temperature is None:
            p = []
            for i, row in enumerate(shift_matrix):
                p2 = []
                for j, r in enumerate(row):
                    p2.append(np.polyfit(self._temperatures, r, self.fitting_order))
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

    def plot_shifts_vs_temperature(self, qpoint=0):
        import matplotlib.pyplot as plt

        shift_matrix = self.get_shift_matrix()

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
        print ('temperature   free energy(KJ/K/mol)  entropy(KJ/K/mol)   cv (J/K/mol)')
        for t in self.get_temperature_range():
            fc = self.get_total_force_constants(temperature=t, volume_index=volume_index)

            free_energy, entropy, cv = obtain_phonopy_thermal_properties(self.structure,
                                                                         temperature=t,
                                                                         mesh=self._mesh,
                                                                         force_constants=fc)
            print ('  {:.1f}        {:14.8f}        {:14.8f}    {:14.8f}'.format(t, free_energy, entropy, cv))
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
    mesh = [20, 20, 20]
    print ('mesh set to: {}'.format(mesh))

if args.ref_temperature is None:
    ref_index = 0
else:
    try:
        ref_index = args.temperatures.index(args.ref_temperature)
    except ValueError:
        print ('reference temperature does not exist')
        exit()

fc_fit = ForceConstantsFitting(structure,
                               files_temperature=args.ct_data,
                               files_volume=args.cv_data,
                               temperatures=args.temperatures,
                               mesh=mesh,
                               ref_index=ref_index,
                               fitting_order=args.order)

temperatures = fc_fit.get_temperature_range()

ev_data = np.loadtxt(args.ev)
volumes = ev_data[:, 0]
energy = ev_data[:, 1]

free_energy = []
entropy = []
cv = []
for i in range(len(volumes)):
    print ('Volume: {} Ang.      Energy(U): {} eV'.format(volumes[i], energy[i]))
    tp_data = fc_fit.get_thermal_properties(volume_index=i)
    free_energy.append(tp_data[0])
    entropy.append(tp_data[1])
    cv.append(tp_data[2])

free_energy = np.array(free_energy).T
entropy = np.array(entropy).T
cv = np.array(cv).T


phonopy_qha = PhonopyQHA(np.array(volumes),
                         np.array(energy),
                         eos="vinet",  # options: 'vinet', 'murnaghan' or 'birch_murnaghan'
                         temperatures=np.array(temperatures),
                         free_energy=np.array(free_energy),
                         cv=np.array(cv),
                         entropy=np.array(entropy),
                         t_max=fc_fit.get_temperature_range()[-1],
                         verbose=False)

phonopy_qha.plot_qha().show()

phonopy_qha.write_bulk_modulus_temperature()
phonopy_qha.write_gibbs_temperature()
phonopy_qha.write_heat_capacity_P_numerical()
phonopy_qha.write_gruneisen_temperature()
phonopy_qha.write_thermal_expansion()
phonopy_qha.write_helmholtz_volume()
phonopy_qha.write_volume_expansion()
phonopy_qha.write_volume_temperature()


