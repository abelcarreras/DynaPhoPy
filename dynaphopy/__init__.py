import numpy as np
import matplotlib.pyplot as plt

import dynaphopy.projection as projection
import dynaphopy.power_spectrum as power_spectrum
import dynaphopy.classes.parameters as parameters
import dynaphopy.interface.phonopy_link as pho_interface
import dynaphopy.interface.iofile as reading
import dynaphopy.analysis.energy as energy
import dynaphopy.analysis.fitting as fitting
import dynaphopy.analysis.modes as modes
import dynaphopy.analysis.coordinates as trajdist

power_spectrum_functions = {
    0: [power_spectrum.get_fourier_spectra_par_openmp, 'Fourier transform'],
    1: [power_spectrum.get_mem_spectra_par_openmp,     'Maximum entropy method'],
    2: [power_spectrum.get_fft_spectra,                'Fast Fourier transform (Numpy)'],
    3: [power_spectrum.get_fft_fftw_spectra,           'Fast Fourier transform (FFTW)']
}

class Calculation:

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
        self._bands = None
        self._renormalized_bands = None
        self._renormalized_force_constants = None

        self._parameters = parameters.Parameters()
        self.crop_trajectory(last_steps)
      #  print('Using {0} time steps for calculation'.format(len(self.dynamic.velocity)))

    #Crop trajectory
    def crop_trajectory(self, last_steps):
        if self._vc is None:
            self._dynamic.crop_trajectory(last_steps)
        else:
            if last_steps is not None:
                self._vc = self._vc[-last_steps:, :, :]

    #Memory clear methods
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
        self._renormalized_force_constants = None
        self._renormalized_bands = None

    #Properties
    @property
    def dynamic(self):
        return self._dynamic

    @property
    def parameters(self):
        return self._parameters


    def set_NAC(self, NAC):
        self._bands = None
        self.parameters.use_NAC = NAC

    def write_to_xfs_file(self,file_name):
        reading.write_xsf_file(file_name,self.dynamic.structure)

    def save_velocity_hdf5(self, file_name, save_trajectory=True):
        if save_trajectory:
            trajectory = self.dynamic.trajectory
        else:
            trajectory = ()

        reading.save_data_hdf5(file_name,
                               self.dynamic.get_time(),
                               self.dynamic.get_super_cell_matrix(),
                               velocity=self.dynamic.velocity,
                               trajectory=trajectory)

        print("Velocity saved in file " + file_name)


    def save_vc_hdf5(self, file_name):

        reading.save_data_hdf5(file_name,
                               self.dynamic.get_time(),
                               self.dynamic.get_super_cell_matrix(),
                               vc=self.get_vc(),
                               reduced_q_vector=self.get_reduced_q_vector())

        print("Projected velocity saved in file " + file_name)

    def set_number_of_mem_coefficients(self,coefficients):
        self.power_spectra_clear()
        self.parameters.number_of_coefficients_mem = coefficients

    #Frequency ranges related methods  (To be deprecated)

    def set_frequency_range(self,frequency_range):
        self.power_spectra_clear()
        self.parameters.frequency_range = frequency_range

    def get_frequency_range(self):
         return self.parameters.frequency_range

    #Wave vector related methods
    def set_reduced_q_vector(self,q_vector):
        self.full_clear()
        self.parameters.reduced_q_vector = np.array(q_vector)

    def get_reduced_q_vector(self):
        return self.parameters.reduced_q_vector

    def get_q_vector(self):
        return np.dot(self.parameters.reduced_q_vector,
                      2.0*np.pi*np.linalg.inv(self.dynamic.structure.get_primitive_cell()))


    #Phonopy harmonic calculation related methods
    def get_eigenvectors(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = (
                pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,
                                                               self.parameters.reduced_q_vector,
                                                               NAC=self.parameters.use_NAC))
        return self._eigenvectors

    def get_frequencies(self):
        if self._frequencies is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = (
                pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,
                                                               self.parameters.reduced_q_vector,
                                                               NAC=self.parameters.use_NAC))
        return self._frequencies

    def set_band_ranges(self,band_ranges):
        self.power_spectra_clear()
        self.parameters.band_ranges = band_ranges

    def get_band_ranges(self):
        return self.parameters.band_ranges

    def get_phonon_dispersion_bands(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                       self.parameters.band_ranges,
                                                                       NAC=self.parameters.use_NAC)

        for i,freq in enumerate(self._bands[1]):
            plt.plot(self._bands[1][i],self._bands[2][i],color ='r')
       # plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_xaxis().set_ticks([])

        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, self._bands[1][-1][-1]])
        plt.suptitle('Phonon dispersion')

        plt.show()

    def get_renormalized_phonon_dispersion_bands(self):

        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                       self.parameters.band_ranges,
                                                                       NAC=self.parameters.use_NAC)

        if self._renormalized_bands is None:
            self._renormalized_bands = pho_interface.obtain_renormalized_phonon_dispersion_bands(self.dynamic.structure,
                                                                                                 self.parameters.band_ranges,
                                                                                                 self.get_renormalized_constants(),
                                                                                                 NAC=self.parameters.use_NAC)


        for i,freq in enumerate(self._renormalized_bands[1]):
            plt.plot(self._bands[1][i],self._bands[2][i],color ='b', label='Harmonic (0K)')
            plt.plot(self._renormalized_bands[1][i],self._renormalized_bands[2][i],color ='r', label='Renormalized')
   #     plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_xaxis().set_ticks([])
        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, self._bands[1][-1][-1]])
        plt.axhline(y=0, color='k', ls='dashed')
        plt.suptitle('Renormalized phonon dispersion')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend([handles[0], handles[-1]], ['Harmonic','Renormalized'])
        plt.show()


    def print_phonon_dispersion_bands(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_bands(self.dynamic.structure,
                                                                       self.parameters.band_ranges,
                                                                       NAC=self.parameters.use_NAC)
        np.set_printoptions(linewidth=200)
        for i,freq in enumerate(self._bands[1]):
            print(str(np.hstack([self._bands[1][i][None].T,self._bands[2][i]])).replace('[','').replace(']',''))

    def plot_eigenvectors(self):
        modes.plot_phonon_modes(self.dynamic.structure,
                                self.get_eigenvectors(),
                                self.get_q_vector(),
                                vectors_scale=self.parameters.modes_vectors_scale)

    def check_commensurate(self, q_vector):
        super_cell= self.dynamic.get_super_cell_matrix()

        commensurate = False
        primitive_matrix = self.dynamic.structure.get_primitive_matrix()
        q_point_unit_cell = np.dot(q_vector, np.linalg.inv(primitive_matrix))
        q_point_unit_cell = np.multiply(q_point_unit_cell, super_cell)*2

        if np.all(np.equal(np.mod(q_point_unit_cell, 1), 0)):
            commensurate = True

        return commensurate

    #Projections related methods
    def get_vc(self):
        if self._vc is None:
            print("Projecting into wave vector")
            #Check if commensurate point
            if not self.check_commensurate(self.get_reduced_q_vector()):
                print("warning! Defined wave vector is not a commensurate q-point in this cell")
            self._vc = projection.project_onto_wave_vector(self.dynamic, self.get_q_vector())
        return self._vc

    def get_vq(self):
        if self._vq is None:
            print("Projecting into phonon")
            self._vq =  projection.project_onto_phonon(self.get_vc(),self.get_eigenvectors())
        return self._vq

    def plot_vq(self, modes=None):
        if not modes: modes = [0]
        plt.suptitle('Phonon mode projection')
        plt.xlabel('Time [ps]')
        plt.ylabel('$u^{1/2}\AA/ps$')

        time = np.linspace(0, self.get_vc().shape[0]*self.dynamic.get_time_step_average(),
                   num=self.get_vc().shape[0])

        for mode in modes:
            plt.plot(time,self.get_vq()[:, mode].real, label='mode: '+str(mode))
        plt.legend()
        plt.show()

    def plot_vc(self,atoms=None,coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]
        time = np.linspace(0, self.get_vc().shape[0]*self.dynamic.get_time_step_average(),
                           num=self.get_vc().shape[0])

        plt.suptitle('Wave vector projection')
        plt.xlabel('Time [ps]')
        plt.ylabel('$u^{1/2}\AA/ps$')

        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(time,
                         self.get_vc()[:, atom,coordinate].real,
                         label='atom: ' + str(atom) + ' coordinate:' + str(coordinate))
        plt.legend()
        plt.show()

    def save_vc(self,file_name):
        print("Saving wave vector projection to file")
        np.savetxt(file_name, self.get_vc()[:, 0, :].real)

    def save_vq(self,file_name):
        print("Saving phonon projection to file")
        np.savetxt(file_name, self.get_vq().real)

    #Power spectra related methods
    def select_power_spectra_algorithm(self,algorithm):
        if algorithm in power_spectrum_functions.keys():
            if algorithm != self.parameters.power_spectra_algorithm:
                self.power_spectra_clear()
                self.parameters.power_spectra_algorithm = algorithm
            print("Using {0} function".format(power_spectrum_functions[algorithm][1]))
        else:
            print("Power spectrum algorithm number not found!\nPlease select:")
            for i in power_spectrum_functions.keys():
                print('{0} : {1}'.format(i,power_spectrum_functions[i][1]))
            exit()

    def get_power_spectrum_phonon(self):
        if self._power_spectrum_phonon is None:
            print("Calculating phonon projection power spectra")

            if self.parameters.use_symmetry:
                initial_reduced_q_point = self.get_reduced_q_vector()
                power_spectrum_phonon = []
                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(self.get_reduced_q_vector(), self.dynamic.structure)
#                print(q_points_equivalent)
                for q_point in q_points_equivalent:
                    self.set_reduced_q_vector(q_point)
                    power_spectrum_phonon.append((
                        power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vq(),
                                                                               self.dynamic,
                                                                               self.parameters))

                self._power_spectrum_phonon = np.average(power_spectrum_phonon, axis=0)
                self.parameters.reduced_q_vector = initial_reduced_q_point
            else:
                self._power_spectrum_phonon = (
                    power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vq(),
                                                                                   self.dynamic,
                                                                                   self.parameters)

        return self._power_spectrum_phonon

    def get_power_spectrum_wave_vector(self):

        if self._power_spectrum_wave_vector is None:
            print('Calculating wave vector projection power spectrum')
            size = self.get_vc().shape[1]*self.get_vc().shape[2]
            if self.parameters.use_symmetry:
                initial_reduced_q_point = self.get_reduced_q_vector()
                power_spectrum_wave_vector = []
                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(self.get_reduced_q_vector(), self.dynamic.structure)
                print(q_points_equivalent)
                for q_point in q_points_equivalent:
                    self.set_reduced_q_vector(q_point)
                    power_spectrum_wave_vector.append((
                        power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vc().swapaxes(1, 2).reshape(-1, size),
                                                                                           self.dynamic,
                                                                                           self.parameters))

                power_spectrum_wave_vector = np.array(power_spectrum_wave_vector)
                self._power_spectrum_wave_vector = np.average(power_spectrum_wave_vector, axis=0)
                self.parameters.reduced_q_vector = initial_reduced_q_point
            else:
                self._power_spectrum_wave_vector = (
                        power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](self.get_vc().swapaxes(1, 2).reshape(-1, size),
                                                                                           self.dynamic,
                                                                                           self.parameters)

        return np.sum(self._power_spectrum_wave_vector,axis=1)

    def get_power_spectrum_direct(self):
        if self._power_spectrum_direct is None:
            print("Calculation full power spectrum")
            size = self.dynamic.get_velocity_mass_average().shape[1]*self.dynamic.get_velocity_mass_average().shape[2]

            self._power_spectrum_direct = (power_spectrum_functions[self.parameters.power_spectra_algorithm])[0](
                self.dynamic.get_velocity_mass_average().swapaxes(1, 2).reshape(-1, size),
                self.dynamic,
                self.parameters)

            self._power_spectrum_direct = np.sum(self._power_spectrum_direct, axis=1)
        return self._power_spectrum_direct

    def phonon_width_scan_analysis(self):
        print("Phonon coefficient scan analysis(Maximum entropy method/Symmetric Lorentzian fit only)")
        power_spectrum.mem_coefficient_scan_analysis(self.get_vq(), self.dynamic, self.parameters)

    def phonon_individual_analysis(self):
        print("Peak analysis analysis")
        fitting.phonon_fitting_analysis(self.get_power_spectrum_phonon(),
                                        self.parameters.frequency_range,
                                        harmonic_frequencies=self.get_frequencies(),
                                        show_plots=not self.parameters.silent,
                                        asymmetric_peaks=self.parameters.use_asymmetric_peaks)
        return

    def plot_power_spectrum_full(self):
        plt.suptitle('Full power spectrum (two sided)')
        plt.plot(self.get_frequency_range(), self.get_power_spectrum_direct(), 'r-')
        plt.xlabel('Frequency [THz]')
        plt.ylabel('eV * ps')
        plt.show()

        total_integral = np.trapz(self.get_power_spectrum_direct(), x=self.get_frequency_range())/(2 * np.pi)
        print ("Total Area (1/2 Kinetic energy <K>): {0} eV".format(total_integral))

    def plot_power_spectrum_wave_vector(self):
        plt.suptitle('Projection onto wave vector (two sided)')
        plt.plot(self.get_frequency_range(),self.get_power_spectrum_wave_vector(), 'r-')
        plt.xlabel('Frequency [THz]')
        plt.ylabel('eV * ps')
        plt.show()
        total_integral = np.trapz(self.get_power_spectrum_wave_vector(), x=self.get_frequency_range())/(2 * np.pi)
        print ("Total Area (1/2 Kinetic energy <K>): {0} eV".format(total_integral))


    def plot_power_spectrum_phonon(self):
        for i in range(self.get_power_spectrum_phonon().shape[1]):
            plt.figure(i)
            plt.suptitle('Projection onto phonon {0} (two sided)'.format(i+1))
            plt.plot(self.get_frequency_range(), self.get_power_spectrum_phonon()[:, i])
            plt.xlabel('Frequency [THz]')
            plt.ylabel('eV * ps')

        plt.show()

    #Plot dynamical properties related methods
    def plot_trajectory(self, atoms=None, coordinates=None):
        if atoms is None : atoms = [0]
        if coordinates is None : coordinates = [0]


        plt.suptitle('Trajectory')
        time = np.linspace(0, self.dynamic.trajectory.shape[0]*self.dynamic.get_time_step_average(),
                           num=self.dynamic.trajectory.shape[0])
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(time, self.dynamic.trajectory[:, atom, coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom,coordinate))

        plt.legend()
        plt.xlabel('Time [ps]')
        plt.ylabel('Angstrom')
        plt.show()

    def plot_velocity(self, atoms=None, coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Velocity')
        time = np.linspace(0, self.dynamic.velocity.shape[0]*self.dynamic.get_time_step_average(),
                           num=self.dynamic.velocity.shape[0])

        for atom in atoms:
            for coordinate in coordinates:
                 plt.plot(time, self.dynamic.velocity[:, atom, coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom,coordinate))

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

        atomic_types = self.dynamic.structure.get_atomic_types()
        atom_type_index_unique = np.unique(self.dynamic.structure.get_atom_type_index(), return_index=True)[1]
        atomic_types_unique = [atomic_types[i] for i in atom_type_index_unique]

        direction = np.array(direction)

        distributions, distance = self.get_atomic_displacements(direction)

        plt.figure()
        for atom in range(distributions.shape[0]):
            width = (distance[1] - distance[0])
            center = (distance[:-1] + distance[1:]) / 2

            plt.figure(atom+1)
            plt.title('Atomic displacements')
            plt.suptitle('Atom {0}, Element {1}'.format(atom, atomic_types_unique[atom]))
            plt.bar(center, distributions[atom], align='center', width=width)
            plt.xlabel('Direction: ' + ' '.join(np.array(direction, dtype=str)) + ' [Angstrom]')
            plt.xlim([distance[0], distance[-1]])

        plt.show()


    #Printing data to files
    def write_power_spectrum_full(self, file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
                                          self.get_power_spectrum_direct()[None].T,
                                          file_name)

    def write_power_spectrum_wave_vector(self, file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
                                          self.get_power_spectrum_wave_vector()[None].T,
                                          file_name)

    def write_power_spectrum_phonon(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
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
                                              normed=True)

            distributions.append(distribution)

        distance = np.array([ i_bin - (bins[1]-bins[0])/2 for i_bin in bins ])

        return np.array(distributions), distance

    def write_atomic_displacements(self, direction, file_name):
        distributions, distance = self.get_atomic_displacements(direction)
        reading.write_correlation_to_file(distance, distributions.T, file_name)


    #Molecular dynamics analysis related methods
    def show_boltzmann_distribution(self):
        energy.boltzmann_distribution(self.dynamic, self.parameters)

    #Other
    def get_algorithm_list(self):
        return power_spectrum_functions.values()


    def get_renormalized_constants(self):

        if self._renormalized_force_constants is None:
            com_points = pho_interface.get_commensurate_points(self.dynamic.structure)

            initial_reduced_q_point = self.get_reduced_q_vector()

            renormalized_frequencies = []
            linewidths = []
            q_points_list = []

            for i, reduced_q_point in enumerate(com_points):

                print ("\nQpoint: {0} / {1}      {2}".format(i+1, len(com_points), reduced_q_point))

                q_points_equivalent = pho_interface.get_equivalent_q_points_by_symmetry(reduced_q_point, self.dynamic.structure)
                q_index = vector_in_list(q_points_equivalent, q_points_list)
                q_points_list.append(reduced_q_point)

                if q_index != 0 and self.parameters.use_symmetry:
                    renormalized_frequencies.append(renormalized_frequencies[q_index])
                    linewidths.append(linewidths[q_index])
                    print('Skipped, equivalent to {0}'.format(q_points_list[q_index]))
                    continue



                self.set_reduced_q_vector(reduced_q_point)
                positions, widths = fitting.phonon_fitting_analysis(self.get_power_spectrum_phonon(),
                                    self.parameters.frequency_range,
                                    harmonic_frequencies=self.get_frequencies(),
                                    show_plots=False,
                                    asymmetric_peaks=self.parameters.use_asymmetric_peaks)

                if (reduced_q_point == [0, 0, 0]).all():
                    print('Fixing gamma point frequencies')
                    positions[0] = 0
                    positions[1] = 0
                    positions[2] = 0

                renormalized_frequencies.append(positions)
                linewidths.append(widths)

            renormalized_frequencies = np.array(renormalized_frequencies)
#            np.savetxt('test_freq', renormalized_frequencies)
#            np.savetxt('test_line', linewidths)


            self._renormalized_force_constants = pho_interface.get_renormalized_force_constants(renormalized_frequencies,
                                                                                                com_points,
                                                                                                self.dynamic.structure,
                                                                                                symmetrize=self.parameters.symmetrize,
                                                                                                degenerate=self.parameters.degenerate)
            self.set_reduced_q_vector(initial_reduced_q_point)


        return self._renormalized_force_constants

    def write_renormalized_constants(self, filename="FORCE_CONSTANTS"):
        force_constants = self.get_renormalized_constants()
        pho_interface.save_force_constants_to_file(force_constants, filename)


def vector_in_list(vector_test_list, vector_full_list):

    for vector_test in vector_test_list:
        for i, vector_full in enumerate(vector_full_list):
            if (vector_full == vector_test).all():
                return i
    return 0