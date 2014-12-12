import numpy as np

import matplotlib.pyplot as plt

import dynaphopy.functions.projection as projection
import dynaphopy.functions.correlate as correlate
import dynaphopy.functions.entropy as mem
import dynaphopy.classes.parameters as parameters
import dynaphopy.functions.phonopy_interface as pho_interface
import dynaphopy.functions.iofunctions as reading
import dynaphopy.functions.energy as energy
import dynaphopy.functions.fitting as fitting

power_spectrum_functions = {
    0: correlate.get_correlation_spectra_par,
    1: mem.get_mem_spectra_par
}

class Calculation:

    def __init__(self,dynamic):

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

        self._parameters = parameters.Parameters()


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

    def set_NAC(self, NAC):
        self._parameters.use_NAC = NAC

    def write_to_xfs_file(self,file_name):
        reading.write_xsf_file(file_name,self.dynamic.structure)

    def save_velocity(self, file_name):
        reading.save_data_hdf5(file_name,
                               self.dynamic.velocity,
                               self.dynamic.get_time(),
                               self.dynamic.get_super_cell_matrix())

        print("Velocity saved in file", file_name)

    def read_velocity(self, file_name):
        print("Loading velocity from file", file_name)
        self.dynamic.velocity = reading.read_data_hdf5(file_name)

    def set_number_of_mem_coefficients(self,coefficients):
        self._parameters.number_of_coefficients_mem = coefficients

       #Frequency ranges related methods
    def set_frequency_range(self,frequency_range):
        self._parameters.frequency_range = frequency_range

    def get_frequency_range(self):
         return self._parameters.frequency_range

    #Wave vector related methods
    def set_reduced_q_vector(self,q_vector):
        self.full_clear()
        self._parameters.reduced_q_vector = np.array(q_vector)

    def get_reduced_q_vector(self):
        return self._parameters.reduced_q_vector


    def get_q_vector(self):
        return np.dot(self._parameters.reduced_q_vector,
                      2.0*np.pi*np.linalg.inv(self.dynamic.structure.get_primitive_cell()))

    #Phonopy harmonic calculation related methods
    def get_eigenvectors(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,
                                                                                                   self._parameters.reduced_q_vector,
                                                                                                   NAC=self._parameters.use_NAC)
            print("Frequencies obtained:")
            print(self._frequencies)
        return self._eigenvectors

    def get_frequencies(self):
        if self._frequencies is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,
                                                                                                   self._parameters.reduced_q_vector,
                                                                                                   NAC=self._parameters.use_NAC)
        return self._frequencies

    def set_band_ranges(self,band_ranges):
        self._parameters.band_ranges = band_ranges

    def get_band_ranges(self):
        return self._parameters.band_ranges

    def get_phonon_dispersion_spectra(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,
                                                                         self._parameters.band_ranges,
                                                                         NAC=self._parameters.use_NAC)

        for i,freq in enumerate(self._bands[1]):
            plt.plot(self._bands[1][i],self._bands[2][i],color ='r')
        plt.axes().get_xaxis().set_visible(False)
        plt.ylabel('Frequency (THz)')
        plt.suptitle('Phonon dispersion spectra')
        plt.show()

    def print_phonon_dispersion_spectrum(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,
                                                                         self._parameters.band_ranges,
                                                                         NAC=self._parameters.use_NAC)
        np.set_printoptions(linewidth=200)
        for i,freq in enumerate(self._bands[1]):
            print(str(np.hstack([self._bands[1][i][None].T,self._bands[2][i]])).replace('[','').replace(']',''))

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
    def select_power_spectra_algorithm(self,algorith):
        if algorith in power_spectrum_functions.keys():
            self._parameters.power_spectra_algorithm = algorith
            print("Using ", power_spectrum_functions[algorith], "Function")
        else:
            print("Algorithm function number not found!\nPlease select:")
            for i in power_spectrum_functions.keys():
                print(i,power_spectrum_functions[i])
            exit()

    def get_correlation_phonon(self):
        if self._correlation_phonon is None:
            print("Calculating phonon projection power spectrum")
            self._correlation_phonon = (power_spectrum_functions[self._parameters.power_spectra_algorithm])(self.get_vq(),
                                                                                                            self.dynamic,
                                                                                                            self._parameters)

        return self._correlation_phonon

    def get_correlation_wave_vector(self):
        if self._correlation_wave_vector is None:
            print("Calculating wave vector projection power spectrum")
            self._correlation_wave_vector = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.get_vc().shape[1]):
                self._correlation_wave_vector += (power_spectrum_functions[self._parameters.power_spectra_algorithm])(self.get_vc()[:,i,:],
                                                                                                                      self._dynamic,
                                                                                                                      self._parameters)

        return np.sum(self._correlation_wave_vector,axis=1)

    def get_correlation_direct(self):
        if self._correlation_direct is None:
            print("Calculation direct power spectrum")
            self._correlation_direct = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.dynamic.get_velocity_mass_average().shape[1]):
                self._correlation_direct += (power_spectrum_functions[self._parameters.power_spectra_algorithm])(self.dynamic.get_velocity_mass_average()[:, i, :],
                                                                                                                 self._dynamic,
                                                                                                                 self._parameters)
        self._correlation_direct = np.sum(self._correlation_direct, axis=1)
        return self._correlation_direct

    def phonon_width_scan_analysis(self):
        print("Phonon coefficient scan analysis")
        self._correlation_phonon =  mem.phonon_width_scan_analysis(self.get_vq(),
                                                                   self.dynamic,
                                                                   self._parameters)

    def phonon_width_individual_analysis(self):
        print("Phonon width analysis (Maximum Entropy Method Only)")
        fitting.phonon_fitting_analysis(self.get_correlation_phonon(),self._parameters)
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
            plt.suptitle('Projection onto Phonon ' + str(i+1))
            plt.plot(self.get_frequency_range(), self.get_correlation_phonon()[:, i])
        plt.show()

    #Analysis of dynamical properties related methods
    def plot_trajectory(self, atoms=None, coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Trajectory')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,
                         self.dynamic.get_trajectory()[:,atom,coordinate].real,
                         label='atom: ' + str(atom) + ' coordinate:' + str(coordinate))
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
                         label='atom: '+str(atom) + ' coordinate:' + str(coordinate))
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
        energy.bolzmann_distribution(self.dynamic)
