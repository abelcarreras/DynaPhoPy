import numpy as np
import Functions.projection as projection
import Functions.correlate as correlate
import matplotlib.pyplot as plt
import Functions.phonopy_interface as pho_interface
import Functions.iofunctions as reading
import Functions.energy as enerfunc


class Calculation:

    def __init__(self,dynamic):
        self._dynamic = dynamic
        self._eigenvectors = None
        self._frequencies = None
        self._frequency_range = None
        self._reduced_q_vector = None
        self._vc = None
        self._vq = None
        self._correlation_phonon = None
        self._correlation_wave_vector = None
        self._correlation_direct = None
        self._band_ranges = None
        self._bands = None
        self._phonopy_NAC = False



    #Memory clear methods
    def full_clear(self):
        self._eigenvectors = None
        self._frequencies = None
        self._reduced_q_vector =None
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

    def set_NAC(self,NAC):
        self._phonopy_NAC = NAC

    def write_to_xfs_file(self,file_name):
        reading.write_xsf_file(file_name,self.dynamic.structure)

    #Wave vector related methods
    def set_reduced_q_vector(self,q_vector):
      #  if not(np.allclose(np.array(q_vector), self._reduced_q_vector)):
        self.full_clear()
        self._reduced_q_vector = np.array(q_vector)


    def get_reduced_q_vector(self):
        if self._reduced_q_vector is None:
            print("Warning: No wave vector especified.\nUsing gamma point wave vector [0 0 0]")
            self._reduced_q_vector = np.array([0,0,0])
        return self._reduced_q_vector


    def get_q_vector(self):
        return np.dot(self.get_reduced_q_vector(), 2.0*np.pi*np.linalg.inv(self.dynamic.structure.get_primitive_cell().T))


    #Phonopy harmonic calculation related methods
    def get_eigenvectors(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,self.get_reduced_q_vector(),NAC=self._phonopy_NAC)
            print("Frequencies obtained:")
            print(self._frequencies)
        return self._eigenvectors


    def get_frequencies(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from Phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,self.get_reduced_q_vector(),NAC=self._phonopy_NAC)
        return self._frequencies


    def set_band_ranges(self,band_ranges):
        self._band_ranges = band_ranges


    def get_band_ranges(self):
        if self._band_ranges is None:
            self._band_ranges = [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.5] ]]
        return self._band_ranges


    def get_phonon_dispersion_spectra(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,self.get_band_ranges(),NAC=self._phonopy_NAC)

        for i,freq in enumerate(self._bands[1]):
            plt.plot(self._bands[1][i],self._bands[2][i],color ='r')
        plt.axes().get_xaxis().set_visible(False)
        plt.ylabel('Frequency (THz)')
        plt.suptitle('Phonon dispersion spectra')
        plt.show()

    def print_phonon_dispersion_spectrum(self):
        if self._bands is None:
            self._bands = pho_interface.obtain_phonon_dispersion_spectra(self.dynamic.structure,self.get_band_ranges(),NAC=self._phonopy_NAC)
        np.set_printoptions(linewidth=200)
        for i,freq in enumerate(self._bands[1]):
            print(str(np.hstack([self._bands[1][i][None].T,self._bands[2][i]])).replace('[','').replace(']',''))
 #           np.savetxt('spectrum.out', np.hstack([self._bands[1][i][None].T,self._bands[2][i]]))

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
                plt.plot(self.dynamic.get_time().real,self.get_vc()[:,atom,coordinate].real,label='atom: '+str(atom)+' coordinate:'+str(coordinate))
        plt.legend()
        plt.show()


    #Frequency ranges related methods
    def set_frequency_range(self,frequency_range):
        if np.array(frequency_range != self._frequency_range).all():

            print("Setting new frequency range")
            self.correlation_clear()
            self._frequency_range = frequency_range


    def get_frequency_range(self):
        if self._frequency_range is None:
            print("Not frequency range specified: using default (0-25THz)")
            self._frequency_range = np.array([0.05*i + 0.1 for i in range (500)])
        return self._frequency_range


    #Correlation related methods
    def get_correlation_phonon(self):
        if self._correlation_phonon is None:
            print("Calculating phonon projection autocorrelation function")
            self._correlation_phonon =  correlate.get_correlation_spectra_par(self.get_vq(),self.dynamic,self.get_frequency_range())
        return self._correlation_phonon


    def get_correlation_wave_vector(self):
        if self._correlation_wave_vector is None:
            print("Calculating wave vector projection autocorrelation function")
            self._correlation_wave_vector = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.get_vc().shape[1]):
                self._correlation_wave_vector += correlate.get_correlation_spectra_par(self.get_vc()[:,i,:],self._dynamic,self.get_frequency_range())
        return np.sum(self._correlation_wave_vector,axis=1)


    def get_correlation_direct(self):
        if self._correlation_direct is None:
            print("Calculation direct autocorrelation function")
            self._correlation_direct = np.zeros((len(self.get_frequency_range()),self.get_vc().shape[2]))
            for i in range(self.dynamic.get_velocity_mass_average().shape[1]):
                self._correlation_direct =+ correlate.get_correlation_spectra_par(self.dynamic.get_velocity_mass_average()[:,i,:],self._dynamic,self.get_frequency_range())
        self._correlation_direct = np.sum(self._correlation_direct,axis=1)
        return self._correlation_direct


    def plot_correlation_direct(self):
        plt.suptitle('Direct correlation')
        plt.plot(self.get_frequency_range(),self.get_correlation_direct(),'r-')
        plt.show()


    def plot_correlation_wave_vector(self):
        plt.suptitle('Projection onto wave vector')
        plt.plot(self.get_frequency_range(),self.get_correlation_wave_vector(),'r-')
        plt.show()


    def plot_correlation_phonon(self):
        for i in range(self.get_correlation_phonon().shape[1]):
            plt.figure(i)
            plt.suptitle('Projection onto Phonon '+str(i+1))
            plt.plot(self.get_frequency_range(),self.get_correlation_phonon()[:,i])
        plt.show()


    #Analysis of dynamical properties related methods
    def plot_trajectory(self,atoms=None,coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Trajectory')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,self.dynamic.get_trajectory()[:,atom,coordinate].real,label='atom: '+str(atom)+' coordinate:'+str(coordinate))
        plt.legend()
        plt.show()


    def plot_velocity(self,atoms=None,coordinates=None):
        if not atoms: atoms = [0]
        if not coordinates: coordinates = [0]

        plt.suptitle('Velocity')
        for atom in atoms:
            for coordinate in coordinates:
                plt.plot(self.dynamic.get_time().real,self.dynamic.velocity[:,atom,coordinate].real,label='atom: '+str(atom)+' coordinate:'+str(coordinate))
        plt.legend()
        plt.show()


    def plot_energy(self):
        plt.suptitle('Energy')
        plt.plot(self.dynamic.get_time().real,self.dynamic.get_energy().real)
        plt.show()


    #Printing data to files
    def write_correlation_direct(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),self.get_correlation_direct()[None].T,file_name)


    def write_correlation_wave_vector(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),self.get_correlation_wave_vector()[None].T,file_name)


    def write_correlation_phonon(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),self.get_correlation_phonon(),file_name)


    #Molecular dynamics analysis related methods
    def show_boltzmann_distribution(self):
        enerfunc.bolzmann_distribution(self.dynamic)

