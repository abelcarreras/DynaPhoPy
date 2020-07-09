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
    def __init__(self,
                 structure,
                 files_volume,
                 files_temperature,
                 temperatures=None,
                 volumes=None,
                 mesh=(40, 40, 40),
                 ref_temperature=None,
                 fitting_order=2,
                 tmin=None,
                 tmax=None,
                 use_NAC=False):

        self.structure = structure
        self.files_volume = files_volume
        self.files_temperature = files_temperature
        self.supercell = structure.get_supercell_phonon()

        self._mesh = mesh
        self.ref_temperature = ref_temperature
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
        self._nac = use_NAC # Use_Nac requires a BORN named file (phonopy style) in the work directory

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


    def get_shift_matrix_temperature(self):

        if self._shift_matrix_temperature is None:
            h_frequencies, ev = self.get_harmonic_frequencies_and_eigenvectors()

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
            #ref_data = self.get_reference_temperature_data()
            h_frequencies, ev = self.get_harmonic_frequencies_and_eigenvectors()
            ref_frequencies = self.get_fitted_shifts_temperature(temperature=self.ref_temperature) + h_frequencies

            shift_matrix = []
            list_v = self.volumes
            for i, t in enumerate(list_v):
                with open(self.files_volume[i], 'r') as stream:
                    data = yaml.load(stream)

                renormalized_frequencies = []

                for qpoint in data:
                    renormalized_frequencies.append(qpoint['frequencies'])

                shift_matrix.append(np.array(renormalized_frequencies) - ref_frequencies)

            self._shift_matrix_volume = np.array(shift_matrix).swapaxes(0, 2)

        return self._shift_matrix_volume

    def get_interpolated_shifts_temperature(self, temperature, kind='quadratic'):
        from scipy.interpolate import interp1d

        shift_matrix = self.get_shift_matrix_temperature()
        shift_temperature = interp1d(self.temperatures, shift_matrix, kind=kind)
        interpolated_shifts = shift_temperature(temperature).T

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

    def get_interpolated_shifts_volume(self, volume, kind='quadratic'):
        from scipy.interpolate import interp1d

        shift_matrix = self.get_shift_matrix_volume()
        shift_volume = interp1d(self.volumes, shift_matrix, kind=kind)

        interpolated_shifts = shift_volume(volume).T

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

    def get_harmonic_frequencies_and_eigenvectors(self):
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
        h_frequencies, eigenvectors = self.get_harmonic_frequencies_and_eigenvectors()
        from dynaphopy.interface.phonopy_link import get_renormalized_force_constants
        fc = get_renormalized_force_constants(renormalized_frequencies,
                                              eigenvectors, self.structure,
                                              self.supercell,
                                              symmetrize=False)
        return fc

    def get_total_shifts(self, volume, temperature, interpolate_temperature=False, interpolate_volume=False):
        if volume is None:
            volume = self.volumes[0]

        if interpolate_temperature:
            temperature_shifts = self.get_interpolated_shifts_temperature(temperature)
        else:
            temperature_shifts = self.get_fitted_shifts_temperature(temperature)

        if interpolate_volume:
            volume_shifts = self.get_fitted_shifts_volume(volume)
        else:
            volume_shifts = self.get_interpolated_shifts_volume(volume)


        return temperature_shifts + volume_shifts

    def get_total_force_constants(self, temperature=300, volume=None):

        total_shifts = self.get_total_shifts(volume, temperature)
        h_frequencies, ev = self.get_harmonic_frequencies_and_eigenvectors()

        total_frequency = h_frequencies + total_shifts

        return self._get_renormalized_force_constants(total_frequency)

    def get_thermal_properties(self, volume=None):
        from dynaphopy.interface.phonopy_link import obtain_phonopy_thermal_properties

        if volume is None:
            volume = self.volumes[0]
            print ('use volume: {}'.format(volume))

        free_energy_list = []
        entropy_list = []
        cv_list = []
        print ('temperature   free energy(KJ/K/mol)  entropy(KJ/K/mol)   cv (J/K/mol)')
        for t in self.get_temperature_range():
            fc = self.get_total_force_constants(temperature=t, volume=volume)

            free_energy, entropy, cv = obtain_phonopy_thermal_properties(self.structure,
                                                                         temperature=t,
                                                                         mesh=self._mesh,
                                                                         force_constants=fc, NAC=self._nac)
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
        free_energy, entropy, cv = self.get_thermal_properties(volume=self.volumes[volume_index])
        temperature = self.get_temperature_range()
        plt.plot(temperature, free_energy, label='free energy')
        plt.plot(temperature, entropy, label='entropy')
        plt.plot(temperature, cv, label='Cv')
        plt.grid()
        plt.legend()
        plt.show()

    def qpoint_to_index(self, qpoint, silent=False):

        for qindex, qp in enumerate(self.get_data_temperature(0)):
            if np.allclose(qp['reduced_wave_vector'], qpoint):
                return qindex
        if not silent:
            print ('{} is not a commensurate point'.format(qpoint))
            exit()
        return None

    def plot_fitted_shifts_temperature(self, qpoint=(0, 0, 0), branch=None, interpolated=False):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift_matrix = self.get_shift_matrix_temperature()

        chk_list = np.arange(self.temperatures[0], self.temperatures[-1], 10)

        if interpolated:
            chk_shift_matrix = np.array([self.get_interpolated_shifts_temperature(t) for t in chk_list]).T
        else:
            chk_shift_matrix = np.array([self.get_fitted_shifts_temperature(t) for t in chk_list]).T

        plt.suptitle('Frequency shift at wave vector={} (relative to harmonic)'.format(qpoint))
        plt.xlabel('Temperature [K]')
        plt.ylabel('Frequency shift [THz]')
        if branch is None:
            plt.plot(chk_list, chk_shift_matrix[:, qindex].T, '-')
            plt.plot(self.temperatures, shift_matrix[:, qindex].T, 'o')
        else:
            plt.title('Branch {}'.format(branch))
            plt.plot(chk_list, chk_shift_matrix[branch, qindex].T, '-')
            plt.plot(self.temperatures, shift_matrix[branch, qindex].T, 'o')
        plt.show()

    def plot_fitted_shifts_volumes(self, qpoint=(0, 0, 0), branch=None, interpolate=False):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift_matrix = self.get_shift_matrix_volume()

        chk_list = np.arange(self.volumes[0], self.volumes[-1], 0.1)

        if interpolate:
            chk_shift_matrix = np.array([self.get_interpolated_shifts_volume(v) for v in chk_list]).T
        else:
            chk_shift_matrix = np.array([self.get_fitted_shifts_volume(v) for v in chk_list]).T

        plt.suptitle('Shifts at wave vector={} (relative to {} K)'.format(qpoint, self.ref_temperature))
        plt.xlabel('Volumes [Ang.^3]')
        plt.ylabel('Frequency [THz]')
        if branch is None:
            plt.plot(chk_list, chk_shift_matrix[:, qindex].T, '-')
            plt.plot(self.volumes, shift_matrix[:, qindex].T, 'o')
        else:
            plt.title('Branch {}'.format(branch))

            plt.plot(chk_list, chk_shift_matrix[branch, qindex].T, '-')
            plt.plot(self.volumes, shift_matrix[branch, qindex].T, 'o')
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
            plt.plot(self.volumes, linewidth, linestyle='--', marker='o')
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
            plt.plot(self.temperatures, linewidth, linestyle='--', marker='o')
            plt.show()
        else:
            print ('qpoint not found!')
            return None


    def plot_shift_volume(self, qpoint=(0, 0, 0), branch=None):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift = []
        for v in range(len(self.volumes)):
            data = self.get_data_volume(v)
            shift.append(data[qindex]['frequency_shifts'])

        plt.suptitle('Frequency shift at wave vector={} (relative to {} K)'.format(qpoint, self.ref_temperature))
        plt.xlabel('Volume [Angs.^3]')
        plt.ylabel('Frequency shift [THz]')

        if branch is None:
            plt.plot(self.volumes, shift)
        else:
            plt.title('Branch {}'.format(branch))
            plt.plot(self.volumes, np.array(shift).T[branch].T)
        plt.show()

    def get_shift_temperature(self, qpoint=(0, 0, 0)):

        qindex = self.qpoint_to_index(qpoint)

        shift = []
        for t in range(len(self.temperatures)):
            data = self.get_data_temperature(t)
            shift.append(data[qindex]['frequency_shifts'])

        return np.array(shift).T

    def plot_shift_temperature(self, qpoint=(0, 0, 0), branch=None):
        import matplotlib.pyplot as plt

        qindex = self.qpoint_to_index(qpoint)

        shift = []
        for t in range(len(self.temperatures)):
            data = self.get_data_temperature(t)
            shift.append(data[qindex]['frequency_shifts'])

        plt.suptitle('Frequency shift at wave vector={} (relative to harmonic)'.format(qpoint))
        plt.xlabel('Temperature [K]')
        plt.ylabel('Frequency shift [THz]')

        if branch is None:
            plt.plot(self.temperatures, shift)
        else:
            plt.title('Branch {}'.format(branch))
            plt.plot(self.temperatures, np.array(shift).T[branch].T)
        plt.show()

    def get_band_structure(self, temperature, volume, band_ranges=None):
        from dynaphopy.interface.phonopy_link import obtain_phonon_dispersion_bands
        if band_ranges is None:
            band_ranges = self.structure.get_path_using_seek_path()['ranges']

        fc = self.get_total_force_constants(temperature=temperature, volume=volume)
        return obtain_phonon_dispersion_bands(self.structure, bands_ranges=band_ranges,
                                              force_constants=fc, NAC=self._nac,
                                              band_resolution=30, band_connection=False)

    def get_dos(self, temperature, volume):
        from dynaphopy.interface.phonopy_link import obtain_phonopy_dos
        fc = self.get_total_force_constants(temperature=temperature, volume=volume)
        return obtain_phonopy_dos(self.structure, mesh=self._mesh, force_constants=fc, NAC=self._nac)

    def plot_dos(self, temperature, volume):
        import matplotlib.pyplot as plt
        dos = self.get_dos(temperature, volume)
        plt.plot(dos[0], dos[1])
        plt.show()


class QuasiparticlesQHA():
    def __init__(self, args, load_data=False, verbose=False, tmin=None):

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
            mesh = [20, 20, 20] # Default
            print ('mesh set to: {}'.format(mesh))

        if 'bands' in input_parameters is None:
            self._bands =  structure.get_path_using_seek_path()
        else:
            self._bands = input_parameters['_band_ranges']

        volumes, energies = read_v_e(args.ev, factor=1.0, volume_factor=1.0, pressure=args.pressure)

        self._fc_fit = ForceConstantsFitting(structure,
                                       files_temperature=args.ct_data,
                                       files_volume=args.cv_data,
                                       temperatures=args.temperatures,
                                       volumes=volumes,
                                       mesh=mesh,
                                       ref_temperature=args.ref_temperature,
                                       fitting_order=args.order,
                                       tmin=tmin,
                                       use_NAC=True)

        if not load_data:
            temperatures = self._fc_fit.get_temperature_range()

            free_energy = []
            entropy = []
            cv = []
            for v, e in zip(volumes, energies):
                print ('Volume: {} Ang.      Energy(U): {} eV'.format(v, e))
                tp_data = self._fc_fit.get_thermal_properties(volume=v)
                free_energy.append(tp_data[0])
                entropy.append(tp_data[1])
                cv.append(tp_data[2])

            free_energy = np.array(free_energy).T
            entropy = np.array(entropy).T
            cv = np.array(cv).T

            np.save('free_energy.npy', free_energy)
            np.save('temperatures.npy', temperatures)
            np.save('cv.npy', cv)
            np.save('entropy.npy', entropy)
        else:
            free_energy = np.load('free_energy.npy')
            temperatures = np.load('temperatures.npy')
            cv = np.load('cv.npy')
            entropy = np.load('entropy.npy')

        self.phonopy_qha = PhonopyQHA(volumes,
                                 energies,
                                 eos="vinet",  # options: 'vinet', 'murnaghan' or 'birch_murnaghan'
                                 temperatures=temperatures,
                                 free_energy=free_energy,
                                 cv=cv,
                                 entropy=entropy,
                                 t_max=self.fc_fit.get_temperature_range()[-1],
                                 verbose=False)

        # Write data files to disk
        self.phonopy_qha.write_bulk_modulus_temperature()
        self.phonopy_qha.write_gibbs_temperature()
        self.phonopy_qha.write_heat_capacity_P_numerical()
        self.phonopy_qha.write_gruneisen_temperature()
        self.phonopy_qha.write_thermal_expansion()
        self.phonopy_qha.write_helmholtz_volume()
        self.phonopy_qha.write_volume_expansion()
        self.phonopy_qha.write_volume_temperature()

        if verbose:
            self.phonopy_qha.plot_qha().show()

    # Designed for test only
    def volume_shift(self, volume_range=np.arange(-2.0, 2.0, 0.1)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        volumes = self.phonopy_qha._qha._volumes
        energies = self.phonopy_qha._qha._electronic_energies

        free_energy = np.load('free_energy.npy')
        temperatures = np.load('temperatures.npy')
        cv = np.load('cv.npy')
        entropy = np.load('entropy.npy')

        for i in volume_range:
            volumesi = np.array(volumes) + i
            print(volumesi)

            phonopy_qha = PhonopyQHA(volumesi,
                                     energies,
                                     eos="vinet",  # options: 'vinet', 'murnaghan' or 'birch_murnaghan'
                                     temperatures=temperatures,
                                     free_energy=free_energy,
                                     cv=cv,
                                     entropy=entropy,
                                     t_max=self.fc_fit.get_temperature_range()[-1],
                                     verbose=False)

            cp = phonopy_qha.get_heat_capacity_P_numerical()
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors

            cNorm = colors.Normalize(vmin=volume_range[0], vmax=volume_range[-1])
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=plt.cm.get_cmap('plasma'))
            ax.plot(phonopy_qha._qha._temperatures[:-3], cp, label='{}'.format(i), color=scalarMap.to_rgba(i))

        import matplotlib as mpl

        ax2 = fig.add_axes([0.93, 0.1, 0.02, 0.8])

        mpl.colorbar.ColorbarBase(ax2, cmap=plt.cm.get_cmap('plasma'), norm=cNorm,
                                  spacing='proportional', ticks=volume_range,
                                  boundaries=None, format='%1i')
        plt.show()

    def plot_dos_gradient(self):

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.colorbar as colorbar

        volumes = self.phonopy_qha.get_volume_temperature()
        temperatures = self.fc_fit.get_temperature_range()

        fig, ax = plt.subplots(1,1)
        for t, v in zip(temperatures[::40], volumes[::20]):
            print ('temperature: {} K'.format(t))
            dos = self.fc_fit.get_dos(t, v)
            cNorm = colors.Normalize(vmin=temperatures[0], vmax=temperatures[-1])
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=plt.cm.get_cmap('plasma'))
            ax.plot(dos[0], dos[1], color=scalarMap.to_rgba(t))

        plt.suptitle('Phonon density of states')
        plt.xlabel('Frequency [THz]')

        ax2 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
        colorbar.ColorbarBase(ax2, cmap=plt.cm.get_cmap('plasma'), norm=cNorm,
                              spacing='proportional', ticks=temperatures[::40],
                              boundaries=None, format='%1i')

        plt.show()

    def plot_band_structure_gradient(self, tmin=300, tmax=1600, tstep=100):

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.colorbar as colorbar

        def replace_list(text_string):
            substitutions = {'GAMMA': u'$\Gamma$',
                             }

            for item in substitutions.iteritems():
                text_string = text_string.replace(item[0], item[1])
            return text_string


        volumes = self.phonopy_qha.get_volume_temperature()
        cNorm = colors.Normalize(vmin=tmin, vmax=tmax)

        fig, ax = plt.subplots(1,1)
        for t in np.arange(tmin, tmax, tstep):
            print ('temperature: {} K'.format(t))
            v = self.get_volume_at_temperature(t)
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=plt.cm.get_cmap('plasma'))
            band_data = self.fc_fit.get_band_structure(t, v, band_ranges=self._bands['ranges'])

            for i, freq in enumerate(band_data[1]):
                ax.plot(band_data[1][i], band_data[2][i], color=scalarMap.to_rgba(t))

                # plt.axes().get_xaxis().set_visible(False)


        #plt.axes().get_xaxis().set_ticks([])
        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')

        plt.xlim([0, band_data[1][-1][-1]])
        plt.axhline(y=0, color='k', ls='dashed')
        plt.suptitle('Phonon dispersion')

        if 'labels' in self._bands:
            plt.rcParams.update({'mathtext.default': 'regular'})
            labels = self._bands['labels']

            labels_e = []
            x_labels = []
            for i in range(len(band_data[1])):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(band_data[1][i][0])
            x_labels.append(band_data[1][-1][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])
            plt.xticks(x_labels, labels_e, rotation='horizontal')

        ax2 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
        colorbar.ColorbarBase(ax2, cmap=plt.cm.get_cmap('plasma'), norm=cNorm,
                              spacing='proportional', ticks=np.arange(tmin, tmax, tstep),
                              boundaries=None, format='%1i')

        plt.show()

    def get_volume_at_temperature(self, temperature):

        temperatures = self.get_qha_temperatures()
        volumes = self.phonopy_qha.get_volume_temperature()
        volume = np.interp(temperature, temperatures, volumes)

        return volume

    def plot_band_structure_constant_pressure(self, temperature=300, external_data=None):

        import matplotlib.pyplot as plt

        def replace_list(text_string):
            substitutions = {'GAMMA': u'$\Gamma$',
                             }

            for item in substitutions.iteritems():
                text_string = text_string.replace(item[0], item[1])
            return text_string

        volume = self.get_volume_at_temperature(temperature)
        fig, ax = plt.subplots(1,1)
        band_data = self.fc_fit.get_band_structure(temperature, volume, band_ranges=self._bands['ranges'])

        for i, freq in enumerate(band_data[1]):
            ax.plot(band_data[1][i], band_data[2][i], color='r')

        #plt.axes().get_xaxis().set_ticks([])
        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')

        plt.xlim([0, band_data[1][-1][-1]])
        plt.axhline(y=0, color='k', ls='dashed')
        plt.suptitle('Phonon dispersion')

        if 'labels' in self._bands:
            plt.rcParams.update({'mathtext.default': 'regular'})
            labels = self._bands['labels']

            labels_e = []
            x_labels = []
            for i in range(len(band_data[1])):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(band_data[1][i][0])
            x_labels.append(band_data[1][-1][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])
            plt.xticks(x_labels, labels_e, rotation='horizontal')

        ax.plot(external_data[:, 0], external_data[:, 1], 'o', color='b')
        plt.show()

    def get_qha_temperatures(self):
        max_t_index = self.phonopy_qha._qha._get_max_t_index(self.phonopy_qha._qha._temperatures)
        temperatures = self.phonopy_qha._qha._temperatures[:max_t_index]

        return temperatures

    def get_FC_constant_pressure(self):
        temperatures = self.get_qha_temperatures()
        volumes = self.phonopy_qha.get_volume_temperature()

        for t, v in zip(temperatures[::20], volumes[::20]):
            fc = self.fc_fit.get_total_force_constants(temperature=t, volume=v)
            write_FORCE_CONSTANTS(fc.get_array(), filename='FC_{}'.format(t))

    def get_total_shift_constant_pressure(self, qpoint=(0, 0, 0)):

        qindex = self.fc_fit.qpoint_to_index(qpoint)

        volumes = self.phonopy_qha.get_volume_temperature()
        temperatures = self.get_qha_temperatures()
        h_frequencies, ev = self.fc_fit.get_harmonic_frequencies_and_eigenvectors()

        chk_shift_matrix = []
        for v, t in zip(volumes, temperatures):
            total_shifts = self.fc_fit.get_total_shifts(volume=v, temperature=t)

            chk_shift_matrix.append(total_shifts)
        chk_shift_matrix = np.array(chk_shift_matrix).T

        return chk_shift_matrix[:, qindex]

    def plot_total_shift_constant_pressure(self, qpoint=(0, 0, 0), branch=None):

        import matplotlib.pyplot as plt

        temperatures = self.get_qha_temperatures()
        chk_shift_matrix = self.get_total_shift_constant_pressure(qpoint=qpoint)

        plt.suptitle('Total frequency shift at wave vector={} (relative to {} K)'.format(qpoint, self.fc_fit.ref_temperature))
        plt.xlabel('Temperature [K]')
        plt.ylabel('Frequency shift [THz]')
        if branch is None:
            plt.plot(temperatures, chk_shift_matrix.T, '-')
        else:
            plt.title('Branch {}'.format(branch))
            plt.plot(temperatures, chk_shift_matrix[branch].T, '-')
        plt.show()

    @property
    def fc_fit(self):
        return self._fc_fit


# FROM HERE DIRTY DEVELOPMENT

qp = QuasiparticlesQHA(args, load_data=False, verbose=True, tmin=200)

qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5])
qp.fc_fit.plot_fitted_shifts_volumes(qpoint=[0.5, 0.5, 0.5])


experiment_data = np.loadtxt('FC/exp_bands')
qp.plot_band_structure_constant_pressure(external_data=experiment_data)
#qp.plot_band_structure_gradient()

qp.fc_fit.plot_shift_volume(qpoint=[0.5, 0.5, 0.5], branch=0)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=0)

qp.plot_total_shift_constant_pressure(qpoint=[0.5, 0.5, 0.5], branch=0)
qp.plot_total_shift_constant_pressure(qpoint=[0.5, 0.5, 0.5], branch=2)

np.savetxt('shift_r', qp.get_total_shift_constant_pressure(qpoint=[0.5, 0.5, 0.5]).T)
np.savetxt('tshift_r', qp.fc_fit.get_shift_temperature(qpoint=[0.5, 0.5, 0.5]).T)
print(qp.fc_fit.temperatures)

print(qp.get_qha_temperatures())
exit()


qp.plot_total_shift_constant_pressure(qpoint=[0.0, 0.0, 0.0], branch=4)
qp.plot_total_shift_constant_pressure(qpoint=[0.0, 0.0, 0.0])

qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.0, 0.0, 0.0])


qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=0)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=1)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=2)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=3)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5], branch=4)
qp.fc_fit.plot_fitted_shifts_temperature(qpoint=[0.5, 0.5, 0.5])
qp.fc_fit.plot_fitted_shifts_volumes(qpoint=[0.5, 0.5, 0.5])

qp.fc_fit.plot_linewidth_temperature(qpoint=[0.5, 0.5, 0.5])
qp.fc_fit.plot_linewidth_volume(qpoint=[0.5, 0.5, 0.5])

qp.plot_dos_gradient()

qp.fc_fit.plot_linewidth_volume([0, 0, 0])