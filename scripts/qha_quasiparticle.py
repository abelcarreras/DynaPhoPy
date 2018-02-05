#!/usr/bin/env python

import numpy as np
import argparse
import dynaphopy.interface.iofile as reading

from phonopy import PhonopyQHA
from phonopy.file_IO import read_v_e, write_FORCE_CONSTANTS
from dynaphopy.interface.phonopy_link import get_force_constants_from_file

import yaml


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
                    help='fitting polynomial order')

parser.add_argument('-ev', metavar='data', type=str, required=True,
                    help='Unit cell volume vs energy file in ang.^3 and eV')

parser.add_argument('--pressure', metavar='p', type=float, required=False, default=0.0,
                    help='external pressure in GPa (default: 0 GPa)')


args = parser.parse_args()

class ForceConstantsFitting():
    def __init__(self, structure, files_volume, files_temperature, temperatures=None, volumes=None,
                 mesh=(40, 40, 40), ref_index=0, fitting_order=2, tmin=None, tmax=None):

        self.structure = structure
        self.files_volume = files_volume
        self.files_temperature = files_temperature
        self.supercell = structure.get_supercell_phonon()

        self._mesh = mesh
        self.ref_index = ref_index
        self.fitting_order = fitting_order

        self._temperatures = temperatures
        self._volumes = volumes

        self._tmin = tmin
        self._tmax = tmax

        self._shift_temperature = None
        self._shift_volume = None

        self._shift_matrix_temperature = None
        self._shift_matrix_volume = None

        self._eigenvectors = None

    # Properties
    @property
    def temperatures(self):
        if self._temperatures is None:
            return range(len(self.files_temperature))
        else:
            return self._temperatures

    @property
    def volumes(self):
        if self._volumes is None:
            return range(len(self.files_volume))
        else:
            return self._volumes

    def get_temperature_range(self, step=10):
        if self._tmin is None:
            self._tmin = self.temperatures[0]
        if self._tmax is None:
            self._tmax = self.temperatures[-1]
        return np.arange(self._tmin, self._tmax, step)

    def get_reference_temperature_data(self):
        with open(args.ct_data[self.ref_index], 'r') as stream:
            ref_data = yaml.load(stream)
        return ref_data

    def get_shift_matrix_temperature(self):

        if self._shift_matrix_temperature is None:
            h_frequencies, ev = self.get_eigenvectors()

            shift_matrix = []
            list_t = self.temperatures
            for i, t in enumerate(list_t):
                with open(self.files_temperature[i], 'r') as stream:
                    data = yaml.load(stream)

                renormalized_frequencies = []
                for j, qpoint in enumerate(data):
                    renormalized_frequencies.append(qpoint['frequencies'])

                shift_matrix.append(np.array(renormalized_frequencies) - np.array(h_frequencies))

            self._shift_matrix_temperature = np.array(shift_matrix).swapaxes(0, 2)

        return self._shift_matrix_temperature

    def get_shift_matrix_volume(self):

        if self._shift_matrix_volume is None:
            ref_data = self.get_reference_temperature_data()

            shift_matrix = []
            list_v = self.volumes
            for i, t in enumerate(list_v):
                with open(self.files_volume[i], 'r') as stream:
                    data = yaml.load(stream)

                renormalized_frequencies = []
                reference_frequencies = []

                for qpoint, ref_qpoint in zip(data, ref_data):
                    renormalized_frequencies.append(qpoint['frequencies'])
                    reference_frequencies.append(ref_qpoint['frequencies'])

                shift_matrix.append(np.array(renormalized_frequencies) - np.array(reference_frequencies))

            self._shift_matrix_volume = np.array(shift_matrix).swapaxes(0, 2)

        return self._shift_matrix_volume



    def get_interpolated_shifts_temperature(self, temperature, kind='quadratic'):
        from scipy.interpolate import interp1d

        shift_matrix = self.get_shift_matrix_temperature()
        if self._shift_temperature is None:
            self._shift_temperature = interp1d(self.temperatures, shift_matrix, kind=kind)

        interpolated_shifts = self._shift_temperature(temperature).T

        return interpolated_shifts

    def get_fitted_shifts_temperature(self, temperature):

        shift_matrix = self.get_shift_matrix_temperature()

        if self._shift_temperature is None:
            p = []
            for i, row in enumerate(shift_matrix):
                p2 = []
                for j, r in enumerate(row):
                    p2.append(np.polyfit(self.temperatures, r, self.fitting_order))
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

    def get_fitted_shifts_volume(self, volume):

        shift_matrix = self.get_shift_matrix_volume()

        if self._shift_volume is None:
            p = []
            for i, row in enumerate(shift_matrix):
                p2 = []
                for j, r in enumerate(row):
                    p2.append(np.polyfit(self.volumes, r, self.fitting_order))
                p.append(p2)

            self._shift_volume = p

        interpolated_shifts = []
        for p in self._shift_volume:
            row = []
            for p2 in p:
                row.append(np.poly1d(p2)(volume))
            interpolated_shifts.append(row)

        interpolated_shifts = np.array(interpolated_shifts).T

        return interpolated_shifts

    def get_eigenvectors(self):
        from dynaphopy.interface.phonopy_link import obtain_eigenvectors_and_frequencies

        if self._eigenvectors is None:
            data = self.get_data_temperature(0)
            com_ev = []
            com_freq = []
            for qpoint in data:
                arranged_ev, frequencies = obtain_eigenvectors_and_frequencies(self.structure,
                                                                               qpoint['reduced_wave_vector'],
                                                                               print_data=False)
                com_ev.append(arranged_ev)
                com_freq.append(frequencies)
            self._eigenvectors = com_ev
            self._h_frequencies = com_freq
        return self._h_frequencies, self._eigenvectors

    def get_data_temperature(self, index):
        with open(self.files_temperature[index], 'r') as stream:
            data = yaml.load(stream)
        return data

    def get_data_volume(self, index):
        with open(self.files_volume[index], 'r') as stream:
            data = yaml.load(stream)
        return data

    def _get_renormalized_force_constants(self, renormalized_frequencies):
        h_frequencies, eigenvectors = self.get_eigenvectors()
        from dynaphopy.interface.phonopy_link import get_renormalized_force_constants
        fc = get_renormalized_force_constants(renormalized_frequencies,
                                              eigenvectors, self.structure,
                                              self.supercell,
                                              symmetrize=False)
        return fc

    def get_total_force_constants(self, temperature=300, volume=None):
        if volume is None:
            volume = self.volumes[0]

        # temperature_shifts = self.get_interpolated_shifts_temperature(temperature)
        temperature_shifts = self.get_fitted_shifts_temperature(temperature)
        volume_shifts = self.get_fitted_shifts_volume(volume)
        h_frequencies, ev = self.get_eigenvectors()

        total_frequency = h_frequencies + volume_shifts + temperature_shifts

        return self._get_renormalized_force_constants(total_frequency)

    def get_thermal_properties(self, volume_index=0):
        from dynaphopy.interface.phonopy_link import obtain_phonopy_thermal_properties

        free_energy_list = []
        entropy_list = []
        cv_list = []
        print ('temperature   free energy(KJ/K/mol)  entropy(KJ/K/mol)   cv (J/K/mol)')
        for t in self.get_temperature_range():
            fc = self.get_total_force_constants(temperature=t, volume=self.volumes[volume_index])

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
        fc = self.get_total_force_constants(temperature=temperature, volume=self.volumes[volume_index])
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

    def qpoint_to_index(self, qpoint):
        qindex = None
        for i, qp in enumerate(self.get_data_temperature(0)):
            if np.allclose(qp['reduced_wave_vector'], qpoint):
                qindex = i
                break
        return qindex

    def plot_fitted_shifts_temperature(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift_matrix = self.get_shift_matrix_temperature()

        chk_list = np.arange(self.temperatures[0], self.temperatures[-1], 10)
        chk_shift_matrix = np.array([self.get_fitted_shifts_temperature(t) for t in chk_list]).T
        # chk_shift_matrix = np.array([self.get_interpolated_shifts_temperature(t) for t in chk_list]).T

        plt.title('Frequency shift at wave vector={} (relative to {} K)'.format(qpoint,
                                                                                 self.temperatures[self.ref_index]))
        plt.xlabel('Temperature [K]')
        plt.ylabel('Frequency shift [THz]')
        plt.plot(chk_list, chk_shift_matrix[:, qindex].T, '-')
        plt.plot(self.temperatures, shift_matrix[:, qindex].T, 'o')
        plt.show()

    def plot_fitted_shifts_volumes(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift_matrix = self.get_shift_matrix_volume()

        chk_list = np.arange(self.volumes[0], self.volumes[-1], 0.1)
        chk_shift_matrix = np.array([self.get_fitted_shifts_volume(v) for v in chk_list]).T
        # chk_shift_matrix = np.array([self.get_interpolated_shifts_temperature(t) for t in chk_list]).T

        plt.title('Frequencies at wave vector={} (relative to {} K)'.format(qpoint,
                                                                                self.temperatures[self.ref_index]))
        plt.xlabel('Volumes [K]')
        plt.ylabel('Frequency [THz]')
        plt.plot(chk_list, chk_shift_matrix[:, qindex].T, '-')
        plt.plot(self.volumes, shift_matrix[:, qindex].T, 'o')
        plt.show()

    def plot_linewidth_volume(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)
        if qindex is not None:
            linewidth = []
            for v in range(len(self.volumes)):
                data = self.get_data_volume(v)
                linewidth.append(data[qindex]['linewidths'])

            plt.title('Linewidths at wave vector={}'.format(qpoint))
            plt.xlabel('Volume [Angs.^3]')
            plt.ylabel('Linewidth [THz]')
            plt.plot(self.volumes, linewidth)
            plt.show()
        else:
            print ('qpoint not found!')
            return None

    def plot_linewidth_temperature(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)
        if qindex is not None:
            linewidth = []
            for t in range(len(self.temperatures)):
                data = self.get_data_temperature(t)
                linewidth.append(data[qindex]['linewidths'])

            plt.title('Linewidths at wave vector={}'.format(qpoint))
            plt.xlabel('Temperature [K]')
            plt.ylabel('Linewidth [THz]')
            plt.plot(self.temperatures, linewidth)
            plt.show()
        else:
            print ('qpoint not found!')
            return None


    def plot_shift_volume(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)
        if qindex is not None:
            linewidth = []
            for v in range(len(self.volumes)):
                data = self.get_data_volume(v)
                linewidth.append(data[qindex]['frequency_shifts'])

            plt.title('Frequency shift at wave vector={} (relative to harmonic)'.format(qpoint))
            plt.xlabel('Volume [Angs.^3]')
            plt.ylabel('Frequency shift [THz]')
            plt.plot(self.volumes, linewidth)
            plt.show()
        else:
            print ('qpoint not found!')
            return None

    def plot_shift_temperature(self, qpoint=(0, 0, 0)):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)
        if qindex is not None:
            linewidth = []
            for t in range(len(self.temperatures)):
                data = self.get_data_temperature(t)
                linewidth.append(data[qindex]['frequency_shifts'])

            plt.title('Frequency shift at wave vector={} (relative to harmonic)'.format(qpoint))
            plt.xlabel('Temperature [K]')
            plt.ylabel('Frequency shift [THz]')
            plt.plot(self.temperatures, linewidth)
            plt.show()
        else:
            print ('qpoint not found!')
            return None




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

volumes, energy = read_v_e(args.ev, factor=1.0, volume_factor=1.0, pressure=args.pressure)

fc_fit = ForceConstantsFitting(structure,
                               files_temperature=args.ct_data,
                               files_volume=args.cv_data,
                               temperatures=args.temperatures,
                               volumes=volumes,
                               mesh=mesh,
                               ref_index=ref_index,
                               fitting_order=args.order,
                               tmin=270)

fc_fit.plot_fitted_shifts_volumes(qpoint=[0.5, 0.5, 0.5])
fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5])

fc_fit.plot_shift_temperature(qpoint=[0.5, 0.5, 0.5])
fc_fit.plot_shift_volume(qpoint=[0.5, 0.5, 0.5])

fc_fit.plot_linewidth_temperature(qpoint=[0.5, 0.5, 0.5])
fc_fit.plot_linewidth_volume(qpoint=[0.5, 0.5, 0.5])


temperatures = fc_fit.get_temperature_range()

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

# get FC at constant pressure
volumes = phonopy_qha.get_volume_temperature()
for t, v in zip(temperatures, volumes):
    fc = fc_fit.get_total_force_constants(temperature=t, volume=v)
    write_FORCE_CONSTANTS(fc.get_array(), filename='FC_{}'.format(t))