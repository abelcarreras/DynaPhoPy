import numpy as np
import Functions.projection as projection
import Functions.correlate as correlate
import matplotlib.pyplot as plt
import Functions.phonopy_interface as pho_interface
import Functions.reading as reading


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

    ####################### Methods & Properties #####################

    #Objects inside
    @property
    def dynamic(self):
        return self._dynamic

    #Wave vector
    def set_reduced_q_vector(self,q_vector):
        self._reduced_q_vector = np.array(q_vector)

    def get_reduced_q_vector(self):
        if self._reduced_q_vector is None:
            self._reduced_q_vector = np.array([0,0,0])
        return self._reduced_q_vector

    def get_q_vector(self):
        print("Getting wave vector")
        return np.dot(self.get_reduced_q_vector(), 2*np.pi*np.linalg.inv(self.dynamic.structure.get_primitive_cell()).T)

    #Phonopy harmonic calculation
    def get_eigenvectors(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,self.get_reduced_q_vector())
        return self._eigenvectors

    def get_frequencies(self):
        if self._eigenvectors is None:
            print("Getting frequencies & eigenvectors from phonopy")
            self._eigenvectors, self._frequencies = pho_interface.obtain_eigenvectors_from_phonopy(self.dynamic.structure,self.get_reduced_q_vector())
        return self._frequencies

    #Projections
    def get_vc(self):
        if self._vc is None:
            print("Projecting into wave vector")
            self._vc = projection.project_onto_unit_cell(self._dynamic,self.get_q_vector())
        return self._vc

    def get_vq(self):
        if self._vq is None:
            print("Projecting into phonon")
            self._vq =  projection.project_onto_phonon(self.get_vc(),self.get_eigenvectors())
        return self._vq

    #Frequency ranges
    def set_frequency_range(self,frequency_range):
        self._frequency_range = frequency_range

    def get_frequency_range(self):
        if self._frequency_range is None:
            self._frequency_range = np.array([0.05*i + 0.1 for i in range (400)])
        return self._frequency_range


    #Correlation section
    def get_correlation_phonon(self):
        if self._correlation_phonon is None:
            print("Getting correlation from phonon projection")
            self._correlation_phonon =  correlate.get_correlation_spectrum_par(self.get_vq(),self.dynamic,self.get_frequency_range())
        return self._correlation_phonon

    def get_correlation_wave_vector(self):
        print("Under test")
        if self._correlation_wave_vector is None:
            self._correlation_wave_vector = np.sum(correlate.get_correlation_spectrum_par(np.sum(self.get_vc(),axis=2),self._dynamic,self.get_frequency_range()),axis=1)
        return self._correlation_wave_vector

    def get_correlation_direct(self):
        print("Not yet well implemented")
        if self._correlation_direct is None:
            self._correlation_direct = np.sum(correlate.get_correlation_spectrum_par(np.sum(self.dynamic.get_velocity_mass_average(),axis=1),self._dynamic,self.get_frequency_range()),axis=1)
        return self._correlation_direct

    def plot_correlation_wave_vector(self):
        plt.suptitle('Projection onto wave vector')
        plt.plot(self.get_frequency_range(),self.get_correlation_wave_vector(),'r-')
        plt.show()

    def plot_correlation_phonon(self):
        for i in range(self.get_correlation_phonon().shape[1]):
            plt.suptitle('Projection onto Phonon '+str(i))
            plt.plot(self.get_frequency_range(),self.get_correlation_phonon()[:,i])
            plt.show()


    #Printing data to files
    def write_correlation_wave_vector(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),np.array(self.get_correlation_wave_vector()[None]).T,file_name)

    def write_correlation_phonon(self,file_name):
        reading.write_correlation_to_file(self.get_frequency_range(),self.get_correlation_phonon(),file_name)