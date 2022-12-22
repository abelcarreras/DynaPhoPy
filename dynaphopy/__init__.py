__version__ = '1.17.14'
import dynaphopy.projection as projection
import dynaphopy.parameters as parameters
import dynaphopy.interface.phonopy_link as pho_interface
import dynaphopy.interface.iofile as reading
import dynaphopy.analysis.energy as energy
import dynaphopy.analysis.fitting as fitting
import dynaphopy.analysis.modes as modes
import dynaphopy.analysis.coordinates as trajdist
import dynaphopy.analysis.thermal_properties as thm
import numpy as np
import matplotlib.pyplot as plt


from dynaphopy.power_spectrum import power_spectrum_functions
from scipy import integrate


class Quasiparticle:
    def __init__(self,
                 dynamic,
                 last_steps=None,
                 vc=None):

        self._dynamic = dynamic
        self._vc = vc
        self._eigenvectors = None
        self._frequencies = None
        self._vq = None
        self._power_spectrum_phonon = None
        self._power_spectrum_wave_vector = None
        self._power_spectrum_direct = None
        self._power_spectrum_partials = None
        self._bands = None
        self._renormalized_bands = None
        self._renormalized_force_constants = None
        self._commensurate_points_data = None
        self._temperature = None
        self._force_constants_qha = None
        self._parameters = parameters.Parameters()
        self.crop_trajectory(last_steps)
        #  print('Using {0} time steps for calculation'.format(len(self.dynamic.velocity)))

    # Crop trajectory
    def crop_trajectory(self, last_steps):
        if self._vc is None:
            self._dynamic.crop_trajectory(last_steps)
            print("Using {0} steps".format(len(self._dynamic.velocity)))
        else:
            if last_steps is not None:
                self._vc = self._vc[-last_steps:, :, :]
            print("Using {0} steps".format(len(self._vc)))

    # Memory clear methods
    def full_clear(self):
        self._eigenvectors = None
        self._frequencies = None
        self._vc = None
        self._vq = None
        self._power_spectrum_direct = None
        self._power_spectrum_wave_vector = None
        self._power_spectrum_phonon = None

    def power_spectra_clear(self):
        self._power_spectrum_phonon = None
        self._power_spectrum_wave_vector = None
        self._power_spectrum_direct = None
        self.force_constants_clear()

    def force_constants_clear(self):
        self._renormalized_force_constants = None
        self._commensurate_points_data = None
        self.bands_clear()

    def bands_clear(self):
        self._bands = None
        self._renormalized_bands = None

    # Properties
    @property
    def dynamic(self):
        return self._dynamic

    @property
    def parameters(self):
        return self._parameters

    def set_NAC(self, NAC):
        """
        Turns on or off Non-Analitic corrections

        :param NAC: True or False
        :return: None
        """
        self._bands = None
        self.parameters.use_NAC = NAC

    def write_to_xfs_file(self, file_name):
        reading.write_xsf_file(file_name, self.dynamic.structure)

    def save_velocity_hdf5(self, file_name, save_trajectory=True):
        if save_trajectory:
            trajectory = self.dynamic.trajectory
        else:
            trajectory = None

        reading.save_data_hdf5(file_name,
                               self.dynamic.get_time(),
                               self.dynamic.get_supercell_matrix(),
                               velocity=self.dynamic.velocity,
                               trajectory=trajectory)

        print("Velocity saved in file " + file_name)

    def save_vc_hdf5(self, file_name):

        reading.save_data_hdf5(file_name,
                               self.dynamic.get_time(),
                               self.dynamic.get_supercell_matrix(),
                               vc=self.get_vc(),
                               reduced_q_vector=self.get_reduced_q_vector())

        print("Projected velocity saved in file " + file_name)

    def set_number_of_mem_coefficients(self, coefficients):
        self.power_spectra_clear()
        self.parameters.number_of_coefficients_mem = coefficients

    def set_projection_onto_atom_type(self, atom_type):
        if atom_type in range(self.dynamic.structure.get_number_of_primitive_atoms()):
            self.parameters.project_on_atom = atom_type
        else:
            print('Atom type {} does not exist'.format(atom_type))
            exit()

    def _set_frequency_range(self, frequency_range):
        if not np.array_equiv(np.array(frequency_range), np.array(self.parameters.frequency_range)):
            self.power_spectra_clear()
            self.parameters.frequency_range = frequency_range

    def set_spectra_resolution(self, resolution):
        limits = [self.get_frequency_range()[0], self.get_frequency_range()[-1]]
        self.parameters.spectrum_resolution = resolution
        self._set_frequency_range(np.arange(limits[0], limits[1] + resolution, resolution))

    def set_frequency_limits(self, limits):
        resolution = self.parameters.spectrum_resolution
        self._set_frequency_range(np.arange(limits[0], limits[1] + resolution, resolution))

    def get_frequency_range(self):
        return self.parameters.frequency_range

    # Wave vector related methods
    def set_reduced_q_vector(self, q_vector):
        if len(q_vector) == len(self.parameters.reduced_q_vector):
            if (np.array(q_vector) != self.parameters.reduced_q_vector).any():
                self.full_clear()

        self.parameters.reduced_q_vector = np.array(q_vector)

    def get_reduced_q_vector(self):
        return self.parameters.reduced_q_vector

    def get_q_vector(self):
        return np.dot(self.parameters.reduced_q_vector,
                      2.0 * np.pi * np.linalg.inv(self.dynamic.structure.get_primitive_cell()).T)

    # Phonopy harmonic calculation related methods
    def get_eigenvectors(self):
        if self._eigenvectors is None:
            # print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = (
                pho_interface.obtain_eigenvectors_and_frequencies(self.dynamic.structure,
                                                                  self.parameters.reduced_q_vector))
        return self._eigenvectors

    def get_frequencies(self):
        if self._frequencies is None:
            # print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = (
                pho_interface.obtain_eigenvectors_and_frequencies(self.dynamic.structure,
                                                                  self.parameters.reduced_q_vector))
        return self._frequencies

    def set_band_ranges(self, band_ranges):
        self.bands_clear()
        if isinstance(band_ranges, dict):
            self.parameters.band_ranges = band_ranges
        elif isinstance(band_ranges, list):
            self.parameters.band_ranges = {'ranges': band_ranges}
        else:
            raise Exception('Incorrect band ranges format')

    def get_band_ranges_and_labels(self):
        # return self.parameters.band_ranges
        if self.parameters.band_ranges is None:
            self.parameters.band_ranges = self.dynamic.structure.get_path_using_seek_path()

        return self.parameters.band_ranges

    def plot_phonon_dispersion_bands(self):
        bands_and_labels = self.get_band_ranges_and_labels()

        band_ranges = bands_and_labels['ranges']

        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                       band_ranges,
                                                                       NAC=self.parameters.use_NAC)

        for i, freq in enumerate(self._bands[1]):
            plt.plot(self._bands[1][i], self._bands[2][i], color='r')

            # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_xaxis().set_ticks([])

        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, self._bands[1][-1][-1]])
        plt.axhline(y=0, color='k', ls='dashed')
        plt.suptitle('Phonon dispersion')

        if 'labels' in bands_and_labels:
            plt.rcParams.update({'mathtext.default': 'regular'})

            labels = bands_and_labels['labels']

            labels_e = []
            x_labels = []
            for i, freq in enumerate(self._bands[1]):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(self._bands[1][i][0])
            x_labels.append(self._bands[1][-1][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])

            plt.xticks(x_labels, labels_e, rotation='horizontal')

        plt.show()

    def plot_renormalized_phonon_dispersion_bands(self, plot_linewidths=False, plot_harmonic=True):

        bands_full_data = self.get_renormalized_phonon_dispersion_bands(with_linewidths=plot_linewidths)

        plot_title = 'Renormalized phonon dispersion relations'
        for i, path in enumerate(bands_full_data):

            plt.plot(path['q_path_distances'], np.array(list(path['renormalized_frequencies'].values())).T, color='r',
                     label='Renormalized')

            if plot_harmonic:
                plt.plot(path['q_path_distances'], np.array(list(path['harmonic_frequencies'].values())).T, color='b',
                         label='Harmonic')

            if plot_linewidths:
                for freq, linewidth in zip(list(path['renormalized_frequencies'].values()),
                                           list(path['linewidth'].values())):
                    plt.fill_between(path['q_path_distances'], freq + np.array(linewidth) / 2,
                                     freq - np.array(linewidth) / 2, color='r', alpha=0.2, interpolate=True,
                                     linewidth=0)
                    plot_title = 'Renormalized phonon dispersion relations and linewidths'

        # plt.axes().get_xaxis().set_visible(False)
        plt.suptitle(plot_title)
        # plt.axes().get_xaxis().set_ticks([])
        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, bands_full_data[-1]['q_path_distances'][-1]])
        plt.axhline(y=0, color='k', ls='dashed')

        if plot_harmonic:
            try:  # Handle issues with old versions of matplotlib
                handles = plt.gca().get_legend_handles_labels()[0]
                plt.legend([handles[-1], handles[0]], ['Harmonic', 'Renormalized'])
            except IndexError:
                pass

        if 'labels' in bands_full_data[0]:
            plt.rcParams.update({'mathtext.default': 'regular'})

            labels = [[bands_full_data[i]['labels']['inf'],
                       bands_full_data[i]['labels']['sup']]
                      for i in range(len(bands_full_data))]

            labels_e = []
            x_labels = []
            for i, freq in enumerate(bands_full_data):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(bands_full_data[i]['q_path_distances'][0])
            x_labels.append(bands_full_data[-1]['q_path_distances'][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])

            plt.xticks(x_labels, labels_e, rotation='horizontal')

        plt.show()

    def plot_linewidths_and_shifts_bands(self):

        bands_full_data = self.get_renormalized_phonon_dispersion_bands(with_linewidths=True,
                                                                        band_connection=True,
                                                                        interconnect_bands=True)

        number_of_branches = len(bands_full_data[0]['linewidth'])
        # print('number_of branches', number_of_branches)

        for i, path in enumerate(bands_full_data):
            prop_cicle = plt.rcParams['axes.prop_cycle']
            colors = prop_cicle.by_key()['color']

            for j in range(number_of_branches):
                plt.figure(0)
                branch = path['linewidth']['branch_{}'.format(j)]
                plt.plot(path['q_path_distances'], branch, color=np.roll(colors, -j)[0], label='linewidth')

                plt.figure(1)
                branch = path['harmonic_frequencies']['branch_{}'.format(j)]
                plt.plot(path['q_path_distances'], branch, color=np.roll(colors, -j)[0], label='linewidth')

                plt.figure(2)
                branch = path['frequency_shifts']['branch_{}'.format(j)]
                plt.plot(path['q_path_distances'], branch, color=np.roll(colors, -j)[0], label='linewidth')

                plt.figure(3)
                branch = path['renormalized_frequencies']['branch_{}'.format(j)]
                plt.plot(path['q_path_distances'], branch, color=np.roll(colors, -j)[0], label='linewidth')


        plt.figure(0)
        plt.suptitle('Phonon linewidths')

        plt.figure(1)
        plt.suptitle('Harmonic phonon dispersion relations')

        plt.figure(2)
        plt.suptitle('Frequency shifts')

        plt.figure(3)
        plt.suptitle('Renormalized phonon dispersion relations')

        for ifig in [0, 1, 2, 3]:
            plt.figure(ifig)

            plt.ylabel('Frequency [THz]')
            plt.xlabel('Wave vector')
            plt.xlim([0, bands_full_data[-1]['q_path_distances'][-1]])
            plt.axhline(y=0, color='k', ls='dashed')

            if 'labels' in bands_full_data[0]:
                plt.rcParams.update({'mathtext.default': 'regular'})

                labels = [[bands_full_data[i]['labels']['inf'],
                           bands_full_data[i]['labels']['sup']]
                          for i in range(len(bands_full_data))]

                labels_e = []
                x_labels = []
                for i, freq in enumerate(bands_full_data):
                    if labels[i][0] == labels[i - 1][1]:
                        labels_e.append(replace_list(labels[i][0]))
                    else:
                        labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                    x_labels.append(bands_full_data[i]['q_path_distances'][0])
                x_labels.append(bands_full_data[-1]['q_path_distances'][-1])
                labels_e.append(replace_list(labels[-1][1]))
                labels_e[0] = replace_list(labels[0][0])
                plt.xticks(x_labels, labels_e, rotation='horizontal')

        plt.show()

    def plot_frequencies_vs_linewidths(self):

        qpoints, multiplicity, frequencies, linewidths = self.get_mesh_frequencies_and_linewidths()

        plt.ylabel('Linewidth [THz]')
        plt.xlabel('Frequency [THz]')

        plt.axhline(y=0, color='k', ls='dashed')
        plt.title('Frequency vs linewidths (from mesh: {})'.format(self.parameters.mesh_phonopy))
        for f, l in zip(np.array(frequencies).T, np.array(linewidths).T):
            plt.scatter(f, l)
        plt.show()

    def get_renormalized_phonon_dispersion_bands(self,
                                                 with_linewidths=False,
                                                 interconnect_bands=False,
                                                 band_connection=False):

        def reconnect_eigenvectors(bands):
            order = range(bands[2][0].shape[1])
            for i, ev_bands in enumerate(bands[3]):
                if i > 0:
                    ref = bands[3][i-1][-1]
                    metric = np.abs(np.dot(ref.conjugate().T, ev_bands[0]))
                    order = np.argmax(metric, axis=1)
                    bands[2][i] = bands[2][i].T[order].T
                    bands[3][i] = bands[3][i].T[order].T

        def reconnect_frequencies(bands):
            order = range(bands[2][0].shape[1])
            for i, f_bands in enumerate(bands[2]):
                if i > 0:
                    order = []
                    ref = np.array(bands[2][i-1][-1]).copy()
                    for j, test_val in enumerate(f_bands[0]):
                        ov = np.argmin(np.abs(ref - test_val))
                        order.append(ov)
                        ref[ov] = 1000

                    # print(order)
                    bands[2][i] = bands[2][i].T[order].T
                    bands[3][i] = bands[3][i].T[order].T

        def eigenvector_order(ev_bands, ev_renormalized):
            metric = np.zeros_like(np.abs(np.dot(ev_bands[0].conjugate().T, ev_renormalized[0])))
            for ev_ref, ev in zip(ev_bands, ev_renormalized):
                metric += np.abs(np.dot(ev_ref.conjugate().T, ev))
            order = np.argmax(metric, axis=1,)
            return order

        def set_order(bands, renormalized_bands):
            order_list = []
            for ev_bands, ev_renormalized in zip(bands[3], renormalized_bands[3]):
                order = eigenvector_order(ev_bands, ev_renormalized)
                order_list.append(order)

            freq = np.array(renormalized_bands[2]).copy()
            for i, o in enumerate(order_list):
                renormalized_bands[2][i] = freq[i].T[o].T

        renormalized_force_constants = self.get_renormalized_force_constants()
        bands = self.get_band_ranges_and_labels()
        band_ranges = bands['ranges']

        _bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                              band_ranges,
                                                              NAC=self.parameters.use_NAC,
                                                              band_connection=band_connection,
                                                              band_resolution=self.parameters.band_resolution)

        if interconnect_bands:
            # reconnect_frequencies(_bands)
            reconnect_eigenvectors(_bands)

        _renormalized_bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                           band_ranges,
                                                                           force_constants=renormalized_force_constants,
                                                                           NAC=self.parameters.use_NAC,
                                                                           band_connection=band_connection,
                                                                           band_resolution=self.parameters.band_resolution)
        if band_connection:
            set_order(_bands, _renormalized_bands)

        data = self.get_commensurate_points_data()
        renormalized_frequencies = data['frequencies']
        eigenvectors = data['eigenvectors']
        linewidths = data['linewidths']
        fc_supercell = data['fc_supercell']

        sup_lim = pho_interface.get_renormalized_force_constants(renormalized_frequencies + linewidths / 2,
                                                                 eigenvectors,
                                                                 self.dynamic.structure,
                                                                 fc_supercell,
                                                                 symmetrize=self.parameters.symmetrize)

        inf_lim = pho_interface.get_renormalized_force_constants(renormalized_frequencies - linewidths / 2,
                                                                 eigenvectors,
                                                                 self.dynamic.structure,
                                                                 fc_supercell,
                                                                 symmetrize=self.parameters.symmetrize)

        if with_linewidths:
            renormalized_bands_s = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                                band_ranges,
                                                                                force_constants=sup_lim,
                                                                                NAC=self.parameters.use_NAC,
                                                                                band_connection=band_connection,
                                                                                band_resolution=self.parameters.band_resolution)

            renormalized_bands_i = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                                band_ranges,
                                                                                force_constants=inf_lim,
                                                                                NAC=self.parameters.use_NAC,
                                                                                band_connection=band_connection,
                                                                                band_resolution=self.parameters.band_resolution)

            if band_connection:
                set_order(_bands, renormalized_bands_s)
                set_order(_bands, renormalized_bands_i)

        bands_full_data = []
        for i, q_path in enumerate(_bands[1]):

            band = {'q_path_distances': q_path.tolist(),
                    'q_bounds': {'inf': list(band_ranges[i][0]), 'sup': list(band_ranges[i][1])},
                    'harmonic_frequencies': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                             enumerate(_bands[2][i].T)},
                    'renormalized_frequencies': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                                 enumerate(_renormalized_bands[2][i].T)},
                    'frequency_shifts': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                         enumerate(_renormalized_bands[2][i].T - _bands[2][i].T)},
                    }

            if with_linewidths:
                band.update({'linewidth_minus': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                                 enumerate(renormalized_bands_i[2][i].T)},
                             'linewidth_plus': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                                enumerate(renormalized_bands_s[2][i].T)},
                             'linewidth': {'branch_{}'.format(key): value.tolist() for (key, value) in
                                           enumerate(renormalized_bands_s[2][i].T - renormalized_bands_i[2][i].T)}}
                            )

            if 'labels' in bands:
                labels = bands['labels']
                band.update({'labels': {'inf': labels[i][0], 'sup': labels[i][1]}})

            bands_full_data.append(band)

        return bands_full_data

    def get_mesh_frequencies_and_linewidths(self):

        data = self.get_commensurate_points_data()

        renormalized_frequencies = data['frequencies']
        eigenvectors = data['eigenvectors']
        linewidths = data['linewidths']
        fc_supercell = data['fc_supercell']

        linewidths_fc = pho_interface.get_renormalized_force_constants(linewidths,
                                                                   eigenvectors,
                                                                   self.dynamic.structure,
                                                                   fc_supercell,
                                                                   symmetrize=self.parameters.symmetrize)

        _, _, linewidths_mesh = pho_interface.obtain_phonopy_mesh_from_force_constants(self.dynamic.structure,
                                                                                       force_constants=linewidths_fc,
                                                                                       mesh=self.parameters.mesh_phonopy,
                                                                                       NAC=None)

        frequencies_fc = pho_interface.get_renormalized_force_constants(renormalized_frequencies,
                                                                        eigenvectors,
                                                                        self.dynamic.structure,
                                                                        fc_supercell,
                                                                        symmetrize=self.parameters.symmetrize)

        qpoints, multiplicity, frequencies_mesh = pho_interface.obtain_phonopy_mesh_from_force_constants(self.dynamic.structure,
                                                                                        force_constants=frequencies_fc,
                                                                                        mesh=self.parameters.mesh_phonopy,
                                                                                        NAC=None)

        return qpoints, multiplicity, frequencies_mesh, linewidths_mesh

    def write_renormalized_phonon_dispersion_bands(self, filename='bands_data.yaml'):
        bands_full_data = self.get_renormalized_phonon_dispersion_bands(with_linewidths=True, band_connection=True)
        reading.save_bands_data_to_file(bands_full_data, filename)

    def print_phonon_dispersion_bands(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                       self.get_band_ranges_and_labels(),
                                                                       NAC=self.parameters.use_NAC)
        np.set_printoptions(linewidth=200)
        for i, freq in enumerate(self._bands[1]):
            print(str(np.hstack([self._bands[1][i][None].T, self._bands[2][i]])).replace('[', '').replace(']', ''))

    def plot_eigenvectors(self):
        modes.plot_phonon_modes(self.dynamic.structure,
                                self.get_eigenvectors(),
                                self.get_q_vector(),
                                vectors_scale=self.parameters.modes_vectors_scale)

    def plot_dos_phonopy(self, force_constants=None):

        phonopy_dos = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                       mesh=self.parameters.mesh_phonopy,
                                                       projected_on_atom=self.parameters.project_on_atom,
                                                       NAC=self.parameters.use_NAC)

        plt.plot(phonopy_dos[0], phonopy_dos[1], 'b-', label='Harmonic')

        if force_constants is not None:
            phonopy_dos_r = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                             mesh=self.parameters.mesh_phonopy,
                                                             force_constants=force_constants,
                                                             projected_on_atom=self.parameters.project_on_atom,
                                                             NAC=self.parameters.use_NAC)

            plt.plot(phonopy_dos_r[0], phonopy_dos_r[1], 'g-', label='Renormalized')

        plt.title('Density of states (Normalized to unit cell)')
        plt.xlabel('Frequency [THz]')
        plt.ylabel('Density of states')
        plt.legend()
        plt.axhline(y=0, color='k', ls='dashed')
        plt.show()

    def check_commensurate(self, q_point, decimals=4):
        supercell = self.dynamic.get_supercell_matrix()

        commensurate = False
        primitive_matrix = self.dynamic.structure.get_primitive_matrix()

        transform = np.dot(q_point, np.linalg.inv(primitive_matrix))
        transform = np.multiply(transform, supercell)
        transform = np.around(transform, decimals=decimals)

        if np.all(np.equal(np.mod(transform, 1), 0)):
            commensurate = True

        return commensurate

    # Projections related methods
    def get_vc(self):
        if self._vc is None:
            print("Projecting into wave vector")
            # Check if commensurate point
            if not self.check_commensurate(self.get_reduced_q_vector()):
                print("warning! This wave vector is not a commensurate q-point in MD supercell")

            if self.parameters.project_on_atom > -1:
                element = self.dynamic.structure.get_atomic_elements(unique=True)[self.parameters.project_on_atom]
                print('Project on atom {} : {}'.format(self.parameters.project_on_atom, element))

            self._vc = projection.project_onto_wave_vector(self.dynamic,
                                                           self.get_q_vector(),
                                                           project_on_atom=self.parameters.project_on_atom)
        return self._vc

    def get_vq(self):
        if self._vq is None:
            print("Projecting into phonon mode")
            self._vq = projection.project_onto_phonon(self.get_vc(), self.get_eigenvectors())
        return self._vq

    def plot_vq(self, modes=None):
        if not modes:
            modes = [0]
        plt.suptitle('Phonon mode projection')
        plt.xlabel('Time [ps]')
        plt.ylabel('$u^{1/2}\AA/ps$')

        time = np.linspace(0, self.get_vc().shape[0] * self.dynamic.get_time_step_average(),
                           num=self.get_vc().shape[0])

        for mode in modes:
            plt.plot(time, self.get_vq()[:, mode].real, label='mode: ' + str(mode))
        plt.legend()
        plt.show()

    def plot_vc(self, atoms=None, coordinates=None):
        if not atoms:
            atoms = [0]
        if not coordinates:
            coordinates = [0]
        time = np.linspace(0, self.get_vc().shape[0] * self.dynamic.get_time_step_average(),
                           num=self.get_vc().shape[0])

        plt.suptitle('Wave vector projection')
        plt.xlabel('Time [ps]')
        plt.ylabel('$u^{1/2}\AA/ps$')

        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(time,
                         self.get_vc()[:, atom, coordinate].real,
                         label='atom: ' + str(atom) + ' coordinate:' + str(coordinate))
        plt.legend()
        plt.show()

    def save_vc(self, file_name, atom=1):
        print("Saving wave vector projection to file")
        if not 0 < atom <= self.get_vc().shape[1]:
            raise Exception('Atom number {} does not exist in primitive cell'.format(atom))
        # np.savetxt(file_name, self.get_vc()[:, atom, :].real)
        np.savetxt(file_name, self.get_vc()[:, atom-1, :], fmt='%+.8e%+.8ej    '*3)

    def save_vq(self, file_name):
        ndim = self.get_vq().shape[1]
        print("Saving phonon projection to file")
        # np.savetxt(file_name, self.get_vq())
        np.savetxt(file_name, self.get_vq(), fmt='%+.8e%+.8ej    '*ndim)

    # Power spectra related methods
    def select_power_spectra_algorithm(self, algorithm):
        if algorithm in power_spectrum_functions.keys():
            if algorithm != self.parameters.power_spectra_algorithm:
                self.power_spectra_clear()
                self.parameters.power_spectra_algorithm = algorithm
            print("Using {0} function".format(power_spectrum_functions[algorithm][1]))
        else:
            print("Power spectrum algorithm number not found!\nPlease select:")
            for i in power_spectrum_functions.keys():
                print('{0} : {1}'.format(i, power_spectrum_functions[i][1]))
            exit()

    def select_fitting_function(self, function):
        from dynaphopy.analysis.fitting.fitting_functions import fitting_functions
        if function in fitting_functions.keys():
            if function != self.parameters.fitting_function:
                self.force_constants_clear()
                self.parameters.fitting_function = function
        else:
            print("Fitting function number not found!\nPlease select:")
            for i in fitting_functions.keys():
                print('{0} : {1}'.format(i, fitting_functions[i]))
            exit()

    def get_power_spectrum_phonon(self):
        if self._power_spectrum_phonon is None:
            print("Calculating phonon projection power spectra")

            if self.parameters.use_symmetry:
                initial_reduced_q_point = self.get_reduced_q_vector()
                power_spectrum_phonon = []
                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(self.get_reduced_q_vector(),
                                                                                        self.dynamic.structure)
                #                print(q_points_equivalent)
                for q_point in q_points_equivalent:
                    self.set_reduced_q_vector(q_point)
                    power_spectrum_phonon.append(
                        (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vq(),
                                                                                               self.dynamic,
                                                                                               self.parameters))

                self.set_reduced_q_vector(initial_reduced_q_point)
                self._power_spectrum_phonon = np.average(power_spectrum_phonon, axis=0)
            else:
                self._power_spectrum_phonon = (
                    power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vq(),
                                                                                          self.dynamic,
                                                                                          self.parameters)

        return self._power_spectrum_phonon

    def get_power_spectrum_wave_vector(self):

        if self._power_spectrum_wave_vector is None:
            print('Calculating wave vector projection power spectrum')
            size = self.get_vc().shape[1] * self.get_vc().shape[2]
            if self.parameters.use_symmetry:
                initial_reduced_q_point = self.get_reduced_q_vector()
                power_spectrum_wave_vector = []
                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(self.get_reduced_q_vector(),
                                                                                        self.dynamic.structure)
                #  print(q_points_equivalent)
                for q_point in q_points_equivalent:
                    self.set_reduced_q_vector(q_point)

                    power_spectrum_wave_vector.append((power_spectrum_functions[
                                                           self.parameters.power_spectra_algorithm])[0](
                        self.get_vc().swapaxes(1, 2).reshape(-1, size),
                        self.dynamic,
                        self.parameters))
                power_spectrum_wave_vector = np.array(power_spectrum_wave_vector)
                self.set_reduced_q_vector(initial_reduced_q_point)

                self._power_spectrum_wave_vector = np.average(power_spectrum_wave_vector, axis=0)

            else:
                self._power_spectrum_wave_vector = (
                    power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                    self.get_vc().swapaxes(1, 2).reshape(-1, size),
                    self.dynamic,
                    self.parameters)

        return np.nansum(self._power_spectrum_wave_vector, axis=1)

    def get_power_spectrum_full(self, projection_on_coordinate=-1):

        # temporal interface
        number_of_dimensions = self.dynamic.structure.get_number_of_dimensions()
        projected_atom_type = self.parameters.project_on_atom

        if self._power_spectrum_direct is None:
            print("Calculation full power spectrum")

            velocity_mass_average = self.dynamic.get_velocity_mass_average()

            if projected_atom_type >= 0:
                print('Power spectrum projected onto atom type {0}'.format(projected_atom_type))
                supercell = self.dynamic.get_supercell_matrix()
                atom_types = np.array(self.dynamic.structure.get_atom_type_index(supercell=supercell))
                atom_indices = np.argwhere(atom_types == projected_atom_type).flatten()
                if len(atom_indices) == 0:
                    print('Atom type {0} does not exist'.format(projected_atom_type))
                    exit()

                # Only works if project on atom is requested!
                if projection_on_coordinate >= number_of_dimensions:
                    print('Projected coordinate should be smaller than {}'.format(number_of_dimensions))
                    exit()
                if projection_on_coordinate > -1:
                    print('Power spectrum projected onto coordinate {}'.format(projection_on_coordinate))
                    velocity_mass_average = velocity_mass_average[:, atom_indices, projection_on_coordinate, None]
                else:
                    velocity_mass_average = velocity_mass_average[:, atom_indices]

            size = velocity_mass_average.shape[1] * velocity_mass_average.shape[2]

            # Memory efficient algorithm
            if self.parameters.silent:
                self._power_spectrum_direct = np.zeros_like(self.parameters.frequency_range[None].T)
                for i in range(velocity_mass_average.shape[1]):
                    for j in range(velocity_mass_average.shape[2]):
                        self._power_spectrum_direct += \
                            (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                                velocity_mass_average[:, i, j][None].T,
                                self.dynamic,
                                self.parameters)

            else:
                self._power_spectrum_direct = (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                    velocity_mass_average.swapaxes(1, 2).reshape(-1, size),
                    self.dynamic,
                    self.parameters)

            self._power_spectrum_direct = np.sum(self._power_spectrum_direct, axis=1)
        return self._power_spectrum_direct

    def get_power_spectrum_partials(self, save_to_file=None):

        if self._power_spectrum_partials is None:
            print("Calculation power spectrum partials")

            velocity_mass_average = self.dynamic.get_velocity_mass_average()

            self._power_spectrum_partials = (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                velocity_mass_average[:, :, 0],
                self.dynamic,
                self.parameters)

            for i in [1, 2]:
                self._power_spectrum_partials += (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                    velocity_mass_average[:, :, i],
                    self.dynamic,
                    self.parameters)

        if save_to_file is not None:
            np.savetxt(save_to_file, np.hstack([self.get_frequency_range()[None].T, self._power_spectrum_partials]))

        return self._power_spectrum_partials

    def phonon_width_scan_analysis(self):
        from dynaphopy.power_spectrum import mem_coefficient_scan_analysis
        print("Phonon coefficient scan analysis(Maximum entropy method/Symmetric Lorentzian fit only)")
        mem_coefficient_scan_analysis(self.get_vq(), self.dynamic, self.parameters)

    def phonon_individual_analysis(self):
        print("Peak analysis analysis")

        fitting.phonon_fitting_analysis(self.get_power_spectrum_phonon(),
                                        self.parameters.frequency_range,
                                        harmonic_frequencies=self.get_frequencies(),
                                        thermal_expansion_shift=self.get_qha_shift(self.get_reduced_q_vector()),
                                        show_plots=not self.parameters.silent,
                                        fitting_function_type=self.parameters.fitting_function,
                                        use_degeneracy=self.parameters.use_symmetry,
                                        show_occupancy=self.parameters.project_on_atom < 0  # temporal interface
                                        )
        return

    def plot_power_spectrum_full(self):

        fig, ax1 = plt.subplots()

        ax1.plot(self.get_frequency_range(), self.get_power_spectrum_full(), 'r-', label='Power spectrum (MD)')
        ax1.set_xlabel('Frequency [THz]')
        ax1.set_ylabel('eV * ps')
        ax2 = ax1.twinx()

        if self.dynamic.structure.forces_available():
            phonopy_dos = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                           mesh=self.parameters.mesh_phonopy,
                                                           freq_min=self.get_frequency_range()[0],
                                                           freq_max=self.get_frequency_range()[-1],
                                                           projected_on_atom=self.parameters.project_on_atom,
                                                           NAC=self.parameters.use_NAC)

            ax2.plot(phonopy_dos[0], phonopy_dos[1], 'b-', label='DoS (Lattice dynamics)')
            ax2.set_ylabel('Density of states')

        if self._renormalized_force_constants is not None:
            phonopy_dos_r = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                             mesh=self.parameters.mesh_phonopy,
                                                             freq_min=self.get_frequency_range()[0],
                                                             freq_max=self.get_frequency_range()[-1],
                                                             force_constants=self._renormalized_force_constants,
                                                             projected_on_atom=self.parameters.project_on_atom,
                                                             NAC=self.parameters.use_NAC)

            ax2.plot(phonopy_dos_r[0], phonopy_dos_r[1], 'g-', label='Renormalized DoS')

        plt.suptitle('Full power spectrum')

        handles1, labels = ax1.get_legend_handles_labels()
        handles2, labels = ax2.get_legend_handles_labels()

        handles = handles1 + handles2
        plt.legend(handles, ['Power spectrum (MD)', 'DoS (Harmonic)', 'DoS (Renormalized)'])
        # plt.legend()
        plt.show()

        total_integral = integrate.simps(self.get_power_spectrum_full(), x=self.get_frequency_range())
        print("Total Area (Kinetic energy <K>): {0} eV".format(total_integral))

    def plot_power_spectrum_wave_vector(self):
        plt.suptitle('Projection onto wave vector')
        plt.plot(self.get_frequency_range(), self.get_power_spectrum_wave_vector(), 'r-')
        plt.xlabel('Frequency [THz]')
        plt.ylabel('eV * ps')
        plt.axhline(y=0, color='k', ls='dashed')
        plt.show()
        total_integral = integrate.simps(self.get_power_spectrum_wave_vector(), x=self.get_frequency_range())
        print("Total Area (Kinetic energy <K>): {0} eV".format(total_integral))

    def plot_power_spectrum_phonon(self):
        for i in range(self.get_power_spectrum_phonon().shape[1]):
            plt.figure(i)
            plt.suptitle('Projection onto phonon mode {0}'.format(i + 1))
            plt.plot(self.get_frequency_range(), self.get_power_spectrum_phonon()[:, i])
            plt.xlabel('Frequency [THz]')
            plt.ylabel('eV * ps')
            plt.axhline(y=0, color='k', ls='dashed')

        plt.show()

    # Plot dynamical properties related methods
    def plot_trajectory(self, atoms=None, coordinates=None):
        if atoms is None: atoms = [0]
        if coordinates is None: coordinates = [0]

        plt.suptitle('Trajectory')
        time = np.linspace(0, self.dynamic.trajectory.shape[0] * self.dynamic.get_time_step_average(),
                           num=self.dynamic.trajectory.shape[0])
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(time, self.dynamic.trajectory[:, atom, coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom, coordinate))

        plt.legend()
        plt.xlabel('Time [ps]')
        plt.ylabel('Angstrom')
        plt.show()

    def plot_velocity(self, atoms=None, coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Velocity')
        time = np.linspace(0, self.dynamic.velocity.shape[0] * self.dynamic.get_time_step_average(),
                           num=self.dynamic.velocity.shape[0])

        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(time, self.dynamic.velocity[:, atom, coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom, coordinate))

        plt.legend()
        plt.xlabel('Time [ps]')
        plt.ylabel('$\AA/ps$')
        plt.show()

    def plot_energy(self):
        plt.suptitle('Energy')
        plt.plot(self.dynamic.get_time().real,
                 self.dynamic.get_energy().real)
        plt.show()

    def plot_trajectory_distribution(self, direction):
        from dynaphopy.analysis.fitting.fitting_functions import Gaussian_function

        atomic_types = self.dynamic.structure.get_atomic_elements()
        atom_type_index_unique = np.unique(self.dynamic.structure.get_atom_type_index(), return_index=True)[1]
        atomic_types_unique = [atomic_types[i] for i in atom_type_index_unique]

        direction = np.array(direction)

        distributions, distance = self.get_atomic_displacements(direction)

        plt.figure()
        for atom in range(distributions.shape[0]):

            plt.figure(atom + 1)
            plt.title('Atomic displacements')
            plt.suptitle('Atom {0}, Element {1}'.format(atom, atomic_types_unique[atom]))

            width = (distance[1] - distance[0])
            center = (distance[:-1] + distance[1:] + width) / 2

            print('\nAtom {0}, Element {1}'.format(atom, atomic_types_unique[atom]))
            print('-----------------------------------------')
            try:

                distance_centers = distance[:-1] + width
                fitting_function = Gaussian_function(distance_centers,
                                                     distributions[atom],
                                                     guess_height=1,
                                                     guess_position=0)

                parameters = fitting_function.get_fitting()
                print('Mean               {0:15.6f} Angstrom'.format(parameters['peak_position']))
                print('Standard deviation {0:15.6f} Angstrom'.format(parameters['width']))
                print('Global fit error   {0:15.6f}'.format(parameters['global_error']))

                plt.plot(distance, fitting_function.get_curve(distance),
                         label=fitting_function.curve_name,
                         linewidth=3, color='g')
            except:
                print('Gaussian fitting failed')

            plt.bar(center, distributions[atom], align='center', width=width)
            plt.xlabel('Direction: ' + ' '.join(np.array(direction, dtype=str)) + ' [Angstrom]')
            plt.xlim([distance[0], distance[-1]])
            plt.ylim([0, None])
            plt.axhline(y=0, color='k', ls='dashed')
            plt.legend()
        plt.show()

    # Printing data to files
    def write_power_spectrum_full(self, file_name):
        reading.write_curve_to_file(self.get_frequency_range(),
                                    self.get_power_spectrum_full()[None].T,
                                    file_name)
        total_integral = integrate.simps(self.get_power_spectrum_full(), x=self.get_frequency_range())
        print("Total Area (Kinetic energy <K>): {0} eV".format(total_integral))

    def write_power_spectrum_wave_vector(self, file_name):
        reading.write_curve_to_file(self.get_frequency_range(),
                                    self.get_power_spectrum_wave_vector()[None].T,
                                    file_name)
        total_integral = integrate.simps(self.get_power_spectrum_wave_vector(), x=self.get_frequency_range())
        print("Total Area (Kinetic energy <K>): {0} eV".format(total_integral))

    def write_power_spectrum_phonon(self, file_name):
        reading.write_curve_to_file(self.get_frequency_range(),
                                    self.get_power_spectrum_phonon(),
                                    file_name)

    def get_atomic_displacements(self, direction):

        number_of_bins = self.parameters.number_of_bins_histogram
        direction = np.array(direction)

        projections = trajdist.trajectory_projection(self.dynamic, direction)

        min_val = np.amin(projections)
        max_val = np.amax(projections)

        bins = None
        distributions = []
        for atom in range(projections.shape[0]):
            distribution, bins = np.histogram(projections[atom],
                                              bins=number_of_bins,
                                              range=(min_val, max_val),
                                              density=True)

            distributions.append(distribution)

        distance = np.array([i_bin - (bins[1] - bins[0]) / 2 for i_bin in bins])

        return np.array(distributions), distance

    def write_atomic_displacements(self, direction, file_name):
        distributions, distance = self.get_atomic_displacements(direction)
        reading.write_curve_to_file(distance, distributions.T, file_name)

    # Molecular dynamics analysis related methods
    def show_boltzmann_distribution(self):
        energy.boltzmann_distribution(self.dynamic, self.parameters)

    def get_temperature(self):
        if not self._temperature:
            save_status = self.parameters.silent
            self.parameters.silent = True
            self._temperature = energy.boltzmann_distribution(self.dynamic, self.parameters)
            self.parameters.silent = save_status
        return self._temperature

    def set_temperature(self, temperature):
        self._temperature = temperature

    def get_algorithm_list(self):
        return power_spectrum_functions.values()

    def get_commensurate_points_data(self, auto_range=True):

        if self._commensurate_points_data is None:

            if auto_range:
                # Get range from harmonic DOS
                phonopy_dos = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                               mesh=self.parameters.mesh_phonopy)

                self.set_frequency_limits([0, np.max(phonopy_dos[0][-1]) * 1.2])
                print('set frequency range: {} - {}'.format(self.get_frequency_range()[0],
                                                            self.get_frequency_range()[-1]))

            # Decide the size of the supercell to use to calculate the renormalized force constants
            if self._parameters.use_MD_cell_commensurate:
                fc_supercell = np.diag(self.dynamic.get_supercell_matrix())
            else:
                fc_supercell = self.dynamic.structure.get_supercell_phonon()

            com_points = pho_interface.get_commensurate_points(self.dynamic.structure,
                                                               fc_supercell)

            initial_reduced_q_vector = self.get_reduced_q_vector()

            renormalized_frequencies = []
            frequency_shifts = []
            eigenvectors = []
            linewidths = []
            q_points_list = []

            for i, reduced_q_vector in enumerate(com_points):

                print("\nQ-point: {0} / {1} ".format(i + 1, len(com_points)) +
                      "    [{:8.5f} {:8.5f} {:8.5f} ]".format(*reduced_q_vector))

                self.set_reduced_q_vector(reduced_q_vector)
                eigenvectors.append(self.get_eigenvectors())

                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(reduced_q_vector,
                                                                                        self.dynamic.structure)
                q_index = _vector_in_list(q_points_equivalent, q_points_list)
                q_points_list.append(reduced_q_vector)

                if q_index != 0 and self.parameters.use_symmetry:
                    renormalized_frequencies.append(renormalized_frequencies[q_index])
                    linewidths.append(linewidths[q_index])
                    frequency_shifts.append(frequency_shifts[q_index])

                    print('Skipped, equivalent to {0}'.format(q_points_list[q_index]))
                    continue

                self.set_reduced_q_vector(reduced_q_vector)

                data = fitting.phonon_fitting_analysis(self.get_power_spectrum_phonon(),
                                                       self.parameters.frequency_range,
                                                       harmonic_frequencies=self.get_frequencies(),
                                                       thermal_expansion_shift=self.get_qha_shift(reduced_q_vector),
                                                       show_plots=False,
                                                       fitting_function_type=self.parameters.fitting_function,
                                                       use_degeneracy=self.parameters.use_symmetry)

                positions = data['positions']
                widths = data['widths']
                if (reduced_q_vector == [0, 0, 0]).all():
                    print('Fixing gamma point 0 frequencies')
                    positions[0] = 0.
                    positions[1] = 0.
                    positions[2] = 0.
                    widths[0] = 0.
                    widths[1] = 0.
                    widths[2] = 0.

                renormalized_frequencies.append(positions)
                linewidths.append(widths)
                frequency_shifts.append(np.array(positions) - self.get_frequencies())

            renormalized_frequencies = np.array(renormalized_frequencies)
            linewidths = np.array(linewidths)
            frequency_shifts = np.array(frequency_shifts)

            # To be deprecated
            if self.parameters.save_renormalized_frequencies:
                print("This option will be deprecated in the future. Please use save quasiparticle data option")
                np.savetxt('renormalized_frequencies', renormalized_frequencies)
            # np.savetxt('test_line', linewidths)

            self._commensurate_points_data = {'frequencies': renormalized_frequencies,
                                              'eigenvectors': eigenvectors,
                                              'linewidths': linewidths,
                                              'frequency_shifts': frequency_shifts,
                                              'q_points': q_points_list,
                                              'fc_supercell': fc_supercell}

            self.set_reduced_q_vector(initial_reduced_q_vector)

        return self._commensurate_points_data

    def get_renormalized_force_constants(self):
        data = self.get_commensurate_points_data()
        renormalized_frequencies = data['frequencies']
        eigenvectors = data['eigenvectors']
        fc_supercell = data['fc_supercell']

        if self._renormalized_force_constants is None:
            self._renormalized_force_constants = pho_interface.get_renormalized_force_constants(
                renormalized_frequencies,
                eigenvectors,
                self.dynamic.structure,
                fc_supercell,
                symmetrize=self.parameters.symmetrize)

        return self._renormalized_force_constants

    def get_commensurate_points_properties(self):

        #  Decide the size of the supercell to use to calculate the renormalized force constants
        if self._parameters.use_MD_cell_commensurate:
            fc_supercell = np.diag(self.dynamic.get_supercell_matrix())
        else:
            fc_supercell = self.dynamic.structure.get_supercell_phonon()

        com_points = pho_interface.get_commensurate_points(self.dynamic.structure,
                                                           fc_supercell)

        group_velocity_list = []
        q_points_list = []
        for i, reduced_q_vector in enumerate(com_points):

            gv = pho_interface.obtain_phonopy_group_velocity(self.dynamic.structure,
                                                             reduced_q_vector,
                                                             force_constants=self.get_renormalized_force_constants())

            group_velocity_list.append(gv)
            q_points_list.append(reduced_q_vector)

        return {'group_velocity': group_velocity_list,
                'q_points': q_points_list,
                'fc_supercell': fc_supercell}

    def write_renormalized_constants(self, filename="FORCE_CONSTANTS"):
        force_constants = self.get_renormalized_force_constants()
        pho_interface.save_force_constants_to_file(force_constants, filename)

    def write_quasiparticles_data(self, filename="quasiparticles_data.yaml", with_extra=False):
        quasiparticle_data = self.get_commensurate_points_data()
        if with_extra:
            quasiparticle_data.update(self.get_commensurate_points_properties())
        reading.save_quasiparticle_data_to_file(quasiparticle_data, filename)

    def write_mesh_data(self, file_name='mesh_data.yaml'):
        mesh_data = self.get_mesh_frequencies_and_linewidths()
        reading.save_mesh_data_to_yaml_file(mesh_data, file_name)

    def get_thermal_properties(self, force_constants=None):

        temperature = self.get_temperature()

        phonopy_dos = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                       mesh=self.parameters.mesh_phonopy,
                                                       freq_min=0.01,
                                                       freq_max=self.get_frequency_range()[-1],
                                                       force_constants=force_constants,
                                                       projected_on_atom=self.parameters.project_on_atom,
                                                       NAC=self.parameters.use_NAC)

        free_energy = thm.get_free_energy(temperature, phonopy_dos[0], phonopy_dos[1])
        entropy = thm.get_entropy(temperature, phonopy_dos[0], phonopy_dos[1])
        c_v = thm.get_cv(temperature, phonopy_dos[0], phonopy_dos[1])
        integration = integrate.simps(phonopy_dos[1], x=phonopy_dos[0]) / (
            self.dynamic.structure.get_number_of_atoms() *
            self.dynamic.structure.get_number_of_dimensions())
        total_energy = thm.get_total_energy(temperature, phonopy_dos[0], phonopy_dos[1])

        if force_constants is not None:
            # Free energy correction
            phonopy_dos_h = pho_interface.obtain_phonopy_dos(self.dynamic.structure,
                                                             mesh=self.parameters.mesh_phonopy,
                                                             freq_min=0.01,
                                                             freq_max=self.get_frequency_range()[-1],
                                                             projected_on_atom=self.parameters.project_on_atom,
                                                             NAC=self.parameters.use_NAC)

            free_energy += thm.get_free_energy_correction_dos(temperature, phonopy_dos_h[0], phonopy_dos_h[1],
                                                              phonopy_dos[1])
            total_energy += thm.get_free_energy_correction_dos(temperature, phonopy_dos_h[0], phonopy_dos_h[1],
                                                               phonopy_dos[1])

            # correction = thm.get_free_energy_correction_dos(temperature, phonopy_dos_h[0], phonopy_dos_h[1], phonopy_dos[1])
            # print('Free energy/total energy correction: {0:12.4f} KJ/mol'.format(correction))

        return [free_energy, entropy, c_v, total_energy, integration]

    def display_thermal_properties(self, from_power_spectrum=False, normalize_dos=False, print_phonopy=False):

        temperature = self.get_temperature()

        print('Using mesh: {0}'.format(self.parameters.mesh_phonopy))

        if print_phonopy:
            harmonic_properties = pho_interface.obtain_phonopy_thermal_properties(self.dynamic.structure,
                                                                                  temperature,
                                                                                  mesh=self.parameters.mesh_phonopy,
                                                                                  NAC=self.parameters.use_NAC)

            renormalized_properties = pho_interface.obtain_phonopy_thermal_properties(self.dynamic.structure,
                                                                                      temperature,
                                                                                      mesh=self.parameters.mesh_phonopy,
                                                                                      force_constants=self.get_renormalized_force_constants(),
                                                                                      NAC=self.parameters.use_NAC)

            print('\nThermal properties per unit cell ({0:.2f} K) [From phonopy (Reference)]\n'
                  '----------------------------------------------'.format(temperature))
            print('                               Harmonic    Quasiparticle\n')
            print('Free energy (not corrected):   {0:.4f}       {3:.4f}     KJ/mol\n'
                  'Entropy:                       {1:.4f}       {4:.4f}     J/K/mol\n'
                  'Cv:                            {2:.4f}       {5:.4f}     J/K/mol\n'.format(
                *(harmonic_properties + renormalized_properties)))

        harmonic_properties = self.get_thermal_properties()
        renormalized_properties = self.get_thermal_properties(force_constants=self.get_renormalized_force_constants())
        frequency_range = self.get_frequency_range()

        if from_power_spectrum:
            normalization = np.prod(self.dynamic.get_supercell_matrix())

            power_spectrum_dos = thm.get_dos(temperature, frequency_range, self.get_power_spectrum_full(),
                                             normalization)
            integration = integrate.simps(power_spectrum_dos, x=frequency_range) / (
                self.dynamic.structure.get_number_of_atoms() *
                self.dynamic.structure.get_number_of_dimensions())

            if normalize_dos:
                power_spectrum_dos /= integration
                integration = 1.0
                if self.parameters.project_on_atom > -1:
                    power_spectrum_dos /= self.dynamic.structure.get_number_of_primitive_atoms()
                    integration /= self.dynamic.structure.get_number_of_primitive_atoms()

            free_energy = thm.get_free_energy(temperature, frequency_range, power_spectrum_dos)
            entropy = thm.get_entropy(temperature, frequency_range, power_spectrum_dos)
            c_v = thm.get_cv(temperature, frequency_range, power_spectrum_dos)
            total_energy = thm.get_total_energy(temperature, frequency_range, power_spectrum_dos)

            power_spectrum_properties = [free_energy, entropy, c_v, total_energy, integration]
            print('\nThermal properties per unit cell ({0:.2f} K) [From DoS]\n'
                  '----------------------------------------------'.format(temperature))
            print('                             Harmonic   Quasiparticle   Power spectrum\n')
            print('Free energy   (KJ/mol): {0:12.4f}  {5:12.4f}  {10:12.4f}\n'
                  'Entropy      (J/K/mol): {1:12.4f}  {6:12.4f}  {11:12.4f}\n'
                  'Cv           (J/K/mol): {2:12.4f}  {7:12.4f}  {12:12.4f}\n'
                  'Total energy  (KJ/mol): {3:12.4f}  {8:12.4f}  {13:12.4f}\n'
                  'Integration:            {4:12.4f}  {9:12.4f}  {14:12.4f}\n'.format(*(harmonic_properties +
                                                                                        renormalized_properties +
                                                                                        power_spectrum_properties)))
            if not self.parameters.silent:
                plt.plot(frequency_range, power_spectrum_dos, 'r-', label='Molecular dynamics')
                plt.axhline(y=0, color='k', ls='dashed')

        else:
            print('\nThermal properties per unit cell ({0:.2f} K) [From DoS]\n'
                  '----------------------------------------------'.format(temperature))
            print('                            Harmonic    Quasiparticle\n')
            print('Free energy   (KJ/mol): {0:12.4f}  {5:12.4f}\n'
                  'Entropy      (J/K/mol): {1:12.4f}  {6:12.4f}\n'
                  'Cv           (J/K/mol): {2:12.4f}  {7:12.4f}\n'
                  'Total energy  (KJ/mol): {3:12.4f}  {8:12.4f}\n'
                  'Integration:            {4:12.4f}  {9:12.4f}\n'.format(
                *(harmonic_properties + renormalized_properties)))

        if not self.parameters.silent:
            self.plot_dos_phonopy(force_constants=self.get_renormalized_force_constants())

    def get_anisotropic_displacement_parameters(self, coordinate_type='uvrs', print_on_screen=True):

        elements = self.dynamic.structure.get_atomic_elements()

        atom_type = self.dynamic.structure.get_atom_type_index()
        atom_type_index_unique = np.unique(atom_type, return_index=True)[1]

        atom_equivalent = np.unique(atom_type, return_counts=True)[1]
        atomic_types_unique = [elements[i] for i in atom_type_index_unique]

        average_positions = self.dynamic.get_mean_displacement_matrix(use_average_positions=True)
        if print_on_screen:
            print('Anisotropic displacement parameters ({0}) [relative to average atomic positions]'.format(
                coordinate_type))
            print('          U11          U22          U33          U23          U13          U12')

        anisotropic_displacements = []
        for i, u_cart in enumerate(average_positions):

            cell = self.dynamic.structure.get_cell()
            cell_inv = np.linalg.inv(cell)  # Check this point
            n = np.array([[np.linalg.norm(cell_inv.T[0]), 0, 0],
                          [0, np.linalg.norm(cell_inv.T[1]), 0],
                          [0, 0, np.linalg.norm(cell_inv.T[2])]])

            u_crys = np.dot(np.dot(cell_inv.T, u_cart), cell_inv)
            u_uvrs = np.dot(np.dot(np.linalg.inv(n), u_crys), np.linalg.inv(n).T)

            u = {'cart': u_cart,
                 'crys': u_crys,
                 'uvrs': u_uvrs}

            if print_on_screen:
                for equivalent in range(atom_equivalent[i]):
                    print(
                        '{0:3} {1:12.8f} {2:12.8f} {3:12.8f} {4:12.8f} {5:12.8f} {6:12.8f}'.format(
                            atomic_types_unique[i],
                            u[coordinate_type][0, 0],
                            u[coordinate_type][1, 1],
                            u[coordinate_type][2, 2],
                            u[coordinate_type][1, 2],
                            u[coordinate_type][0, 2],
                            u[coordinate_type][0, 1]))

            anisotropic_displacements.append(u[coordinate_type])

        return anisotropic_displacements

    def get_average_atomic_positions(self, to_unit_cell=True):
        print('Average atomic positions')

        supercell = None
        if not to_unit_cell:
            supercell = self.dynamic.get_supercell_matrix()

        positions_average = self.dynamic.average_positions(to_unit_cell=to_unit_cell)
        elements = self.dynamic.structure.get_atomic_elements(supercell=supercell)

        for i, coordinate in enumerate(positions_average):
            print('{0:2} '.format(elements[i]) + '{0:15.8f} {1:15.8f} {2:15.8f}'.format(*coordinate.real))

    # QHA methods
    def set_qha_force_constants(self, fc_qha_file):
        self._force_constants_qha = pho_interface.get_force_constants_from_file(fc_qha_file,
                                                                                fc_supercell=self.dynamic.structure.get_supercell_phonon())

    def get_qha_shift(self, reduced_q_vector):
        if self._force_constants_qha is not None:
            import copy
            structure_qha = copy.copy(self.dynamic.structure)
            structure_qha.set_force_constants(self._force_constants_qha)
            qha_frequencies = pho_interface.obtain_eigenvectors_and_frequencies(structure_qha,
                                                                                reduced_q_vector,
                                                                                print_data=False)[1]
            return qha_frequencies - self.get_frequencies()
        else:
            return None


# Support functions
def _vector_in_list(vector_test_list, vector_full_list):
    for vector_test in vector_test_list:
        for i, vector_full in enumerate(vector_full_list):
            if (vector_full == vector_test).all():
                return i
    return 0


def replace_list(text_string):
    substitutions = {'GAMMA': u'$\Gamma$',
                     }

    for item in substitutions.items():
        text_string = text_string.replace(item[0], item[1])
    return text_string
