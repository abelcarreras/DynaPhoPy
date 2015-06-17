import numpy as np

import matplotlib.pyplot as plt

import dynaphopy.functions.projection as projection
import dynaphopy.methods.correlate as correlate
import dynaphopy.methods.maximum_entropy as mem
import dynaphopy.classes.parameters as parameters
import dynaphopy.functions.phonopy_link as pho_interface
import dynaphopy.functions.iofile as reading
import dynaphopy.analysis.energy as energy
import dynaphopy.analysis.fitting as fitting
import dynaphopy.analysis.modes as modes

power_spectrum_functions = {
    0: correlate.get_correlation_spectra_par_python,
    1: mem.get_mem_spectra_par_python,
    2: correlate.get_correlation_spectra_serial,
    3: correlate.get_correlation_spectra_par_openmp,
    4: mem.get_mem_spectra_par_openmp
}


class Calculation:

    def __init__(self,
                 dynamic,
                 last_steps=None,
                 save_hfd5=None):

        self._dynamic = dynamic
        self._eigenvectors = None
        self._frequencies = None
        self._reduced_q_vector = None
        self._vc = None
        self._vq = None
        self._correlation_phonon = None
        self._correlation_wave_vector = None
        self._correlation_direct = None
        self._bands = None
        self._anharmonic_bands = None

        self._parameters = parameters.Parameters()

        if save_hfd5:
            self.save_velocity(save_hfd5)

        if last_steps:
            self._parameters.last_steps = last_steps

        self._dynamic.crop_trajectory(self._parameters.last_steps)
        print("Using {0} steps".format(dynamic.velocity.shape[0]))

    #Memory clear methods
    def full_clear(self):
        self._eigenvectors = None
        self._frequencies = None
        self._vc = None
        self._vq = None
        self._correlation_direct = None
        self._correlation_wave_vector = None
        self._correlation_phonon = None

    def correlation_clear(self):
        self._correlation_phonon = None
        self._correlation_wave_vector = None
        self._correlation_direct = None

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

    def save_velocity(self, file_name, save_trajectory=True):
        if save_trajectory:
            trajectory = self.dynamic.trajectory
        else:
            trajectory = ()

        reading.save_data_hdf5(file_name,
                               self.dynamic.velocity,
                               self.dynamic.get_time(),
                               self.dynamic.get_super_cell_matrix(),
                               trajectory=trajectory)

        print("Velocity saved in file " + file_name)


    def set_number_of_mem_coefficients(self,coefficients):
        self.correlation_clear()
        self.parameters.number_of_coefficients_mem = coefficients

       #Frequency ranges related methods  (To be deprecated)

    def set_frequency_range(self,frequency_range):
        self.correlation_clear()
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
            print("Frequencies obtained:")
            print(self._frequencies)
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
        self.correlation_clear()
        self.parameters.band_ranges = band_ranges

    def get_band_ranges(self):
        return self.parameters.band_ranges

    def get_phonon_dispersion_spectra(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,
                                                                         self.parameters.band_ranges,
                                                                         NAC=self.parameters.use_NAC)

        for i,freq in enumerate(self._bands[1]):
            plt.plot(self._bands[1][i],self._bands[2][i],color ='r')
        plt.axes().get_xaxis().set_visible(False)
        plt.ylabel('Frequency (THz)')
        plt.suptitle('Phonon dispersion spectra')
        plt.show()

    def print_phonon_dispersion_spectrum(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,
                                                                         self.parameters.band_ranges,
                                                                         NAC=self.parameters.use_NAC)
        np.set_printoptions(linewidth=200)
        for i,freq in enumerate(self._bands[1]):
            print(str(np.hstack([self._bands[1][i][None].T,self._bands[2][i]])).replace('[','').replace(']',''))

    def plot_eigenvectors(self):
        modes.plot_phonon_modes(self.dynamic.structure, self.get_eigenvectors(), self.get_q_vector())

    #Projections related methods
    def get_vc(self):
        if self._vc is None:
            print("Projecting into wave vector")
            self._vc = projection.project_onto_wave_vector(self._dynamic,self.get_q_vector())
        return self._vc

    def get_vq(self):
        if self._vq is None:
            print("Projecting into phonon")
            self._vq =  projection.project_onto_phonon(self.get_vc(),self.get_eigenvectors())
        return self._vq

    def plot_vq(self,modes=None):
        if not modes: modes = [0]
        plt.suptitle('Phonon mode projection')
        for mode in modes:
            plt.plot(self.dynamic.get_time().real,self.get_vq()[:,mode].real,label='mode: '+str(mode))
        plt.legend()
        plt.show()

    def plot_vc(self,atoms=None,coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Wave vector projection')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,
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
                self.correlation_clear()
                self.parameters.power_spectra_algorithm = algorithm
            print("Using {0} function".format(power_spectrum_functions[algorithm]))
        else:
            print("Algorithm function number not found!\nPlease select:")
            for i in power_spectrum_functions.keys():
                print(i,power_spectrum_functions[i])
            exit()

    def get_correlation_phonon(self):
        if self._correlation_phonon is None:
            print("Calculating phonon projection power spectrum")
            self._correlation_phonon = (
                power_spectrum_functions[self.parameters.power_spectra_algorithm])(self.get_vq(),
                                                                                   self.dynamic,
                                                                                   self.parameters)

        return self._correlation_phonon

    def get_correlation_wave_vector(self):
        if self._correlation_wave_vector is None:
            print("Calculating wave vector projection power spectrum")
            self._correlation_wave_vector = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.get_vc().shape[1]):
                self._correlation_wave_vector += (
                    power_spectrum_functions[self.parameters.power_spectra_algorithm])(self.get_vc()[:,i,:],
                                                                                       self.dynamic,
                                                                                       self.parameters)

        return np.sum(self._correlation_wave_vector,axis=1)

    def get_correlation_direct(self):
        if self._correlation_direct is None:
            print("Calculation direct power spectrum")
            self._correlation_direct = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.dynamic.get_velocity_mass_average().shape[1]):
                self._correlation_direct += (power_spectrum_functions[self.parameters.power_spectra_algorithm])(
                    self.dynamic.get_velocity_mass_average()[:, i, :],
                    self.dynamic,
                    self.parameters)
        self._correlation_direct = np.sum(self._correlation_direct, axis=1)
        return self._correlation_direct

    def phonon_width_scan_analysis(self):
        print("Phonon coefficient scan analysis(Maximum Entropy Method Only)")
        self._correlation_phonon =  mem.mem_coefficient_scan_analysis(self.get_vq(),
                                                                   self.dynamic,
                                                                   self.parameters)

    def phonon_width_individual_analysis(self):
        print("Phonon width analysis")
        fitting.phonon_fitting_analysis(self.get_correlation_phonon(),
                                        self.parameters.frequency_range,
                                        harmonic_frequencies=self.get_frequencies(),
                                        show_plots=not self.parameters.silent)
        return

    def plot_correlation_direct(self):
        plt.suptitle('Direct correlation')
        plt.plot(self.get_frequency_range(), self.get_correlation_direct(), 'r-')
        plt.show()

    def plot_correlation_wave_vector(self):
        plt.suptitle('Projection onto wave vector')
        plt.plot(self.get_frequency_range(),self.get_correlation_wave_vector(), 'r-')
        plt.show()

    def plot_correlation_phonon(self):
        for i in range(self.get_correlation_phonon().shape[1]):
            plt.figure(i)
            plt.suptitle('Projection onto Phonon {0}'.format(i+1))
            plt.plot(self.get_frequency_range(), self.get_correlation_phonon()[:, i])
        plt.show()

    def get_anharmonic_dispersion_spectra(self, band_resolution=30):
        if self._anharmonic_bands is None:
            anharmonic_bands = []
            for q_start, q_end in self.parameters.band_ranges:
                print(q_start)
                for i in range(band_resolution+1):
                    q_vector = np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i

                    self.set_reduced_q_vector(q_vector)
                    phonon_frequencies = fitting.phonon_fitting_analysis(self.get_correlation_phonon(),
                                                                     self.parameters.frequency_range,
                                                                     show_plots=False)[0]
                    anharmonic_bands.append(phonon_frequencies)
            self._anharmonic_bands = np.array(anharmonic_bands)
        return self._anharmonic_bands

    #Plot dynamical properties related methods
    def plot_trajectory(self, atoms=None, coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Trajectory')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,
                         self.dynamic.trajectory[:,atom,coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom,coordinate))
        plt.legend()
        plt.show()

    def plot_velocity(self, atoms=None, coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Velocity')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,
                         self.dynamic.velocity[:, atom, coordinate].real,
                         label='atom: {0}  coordinate: {1}'.format(atom,coordinate))
        plt.legend()
        plt.show()

    def plot_energy(self):
        plt.suptitle('Energy')
        plt.plot(self.dynamic.get_time().real,
                 self.dynamic.get_energy().real)
        plt.show()

    #Printing data to files
    def write_correlation_direct(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
                                          self.get_correlation_direct()[None].T,
                                          file_name)

    def write_correlation_wave_vector(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
                                          self.get_correlation_wave_vector()[None].T,
                                          file_name)

    def write_correlation_phonon(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),
                                          self.get_correlation_phonon(),
                                          file_name)

    #Molecular dynamics analysis related methods
    def show_boltzmann_distribution(self):
        energy.boltzmann_distribution(self.dynamic)

    #Other
    def get_algorithm_list(self):
        return power_spectrum_functions.values()