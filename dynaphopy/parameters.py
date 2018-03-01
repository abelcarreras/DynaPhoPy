import numpy as np

# This class contains all the default parameters for DynaPhoPy

class Parameters:

    def __init__(self,
                 # General
                 silent=False,

                 # Projections
                 reduced_q_vector=(0, 0, 0),  # default reduced wave vector

                 # Maximum Entropy Method
                 number_of_coefficients_mem=1000,
                 mem_scan_range=np.array(np.linspace(40, 1000, 100), dtype=int),

                 # Fourier transform Method
                 correlation_function_step=10,
                 integration_method=1,  # 0: Trapezoid  1:Rectangles

                 # Fast Fourier tranform Method
                 zero_padding=0,

                 # Power spectra
                    # 0: Correlation functions parallel (OpenMP) [Recommended]
                    # 1: Maximum Entropy Method parallel (OpenMP) [Recommended]
                    # 2: FFT via numpy
                    # 3: FFT via FFTW
                 power_spectra_algorithm=1,
                 spectrum_resolution=0.05,
                 frequency_range=np.arange(0, 40.05, 0.05),
                 # Fitting function
                    # 0: Correlation functions parallel (OpenMP) [Recommended]
                    # 1: Maximum Entropy Method parallel (OpenMP) [Recommended]
                    # 2: FFT via numpy
                    # 3: FFT via FFTW
                 fitting_function=0, # Default fitting function

                 # Phonon dispersion diagram
                 use_NAC=False,
                 band_ranges=None,
                 band_resolution=100,
                 number_of_bins_histogram=30,

                 # Force constants
                 symmetrize=False,
                 use_symmetry=True,

                # Renormalized frequencies
                 save_renormalized_frequencies=False,

                 # Modes (eigenvectors) display
                 modes_vectors_scale=10,

                 #Density of states mesh (phonopy)
                 mesh_phonopy=(40, 40, 40),

                 #Use supercell
                 use_MD_cell_commensurate=False,

                 #(On development (temporal))
                 project_on_atom=-1
                 ):

        self._silent = silent
        self._number_of_coefficients_mem = number_of_coefficients_mem
        self._mem_scan_range = mem_scan_range
        self._correlation_function_step = correlation_function_step
        self._integration_method = integration_method
        self._power_spectra_algorithm = power_spectra_algorithm
        self._fitting_function = fitting_function
        self._zero_padding = zero_padding
        self._frequency_range = frequency_range
        self._spectrum_resolution = spectrum_resolution
        self._reduced_q_vector = reduced_q_vector
        self._use_NAC = use_NAC
        self._band_ranges = band_ranges
        self._number_of_bins_histogram = number_of_bins_histogram
        self._band_resolution = band_resolution

        self._symmetrize = symmetrize
        self._use_symmetry = use_symmetry
        self._save_renormalized_frequencies = save_renormalized_frequencies

        self._modes_vectors_scale = modes_vectors_scale
        self._mesh_phonopy = mesh_phonopy
        self._use_MD_cell_commensurate = use_MD_cell_commensurate

        self._project_on_atom = project_on_atom

    def get_data_from_dict(self, data_dictionary):
        for data in self.__dict__:
            try:
                self.__dict__[data] = data_dictionary[data]
            except KeyError:
                continue


    #Properties
    @property
    def silent(self):
        return self._silent

    @silent.setter
    def silent(self, silent):
        self._silent = silent

    @property
    def reduced_q_vector(self):
        return self._reduced_q_vector

    @reduced_q_vector.setter
    def reduced_q_vector(self,reduced_q_vector):
        self._reduced_q_vector = reduced_q_vector

    @property
    def number_of_coefficients_mem(self):
        return self._number_of_coefficients_mem

    @number_of_coefficients_mem.setter
    def number_of_coefficients_mem(self,number_of_coefficients_mem):
        self._number_of_coefficients_mem = number_of_coefficients_mem

    @property
    def mem_scan_range(self):
        return self._mem_scan_range

    @mem_scan_range.setter
    def mem_scan_range(self,mem_scan_range):
        self._mem_scan_range = mem_scan_range

    @property
    def correlation_function_step(self):
        return self._correlation_function_step
    
    @correlation_function_step.setter
    def correlation_function_step(self,correlation_function_step):
        self._correlation_function_step = correlation_function_step

    @property
    def integration_method(self):
        return self._integration_method
    
    @integration_method.setter
    def integration_method(self,integration_method):
        self._integration_method = integration_method

    @property
    def frequency_range(self):
        return self._frequency_range

    @frequency_range.setter
    def frequency_range(self, frequency_range):
        self._frequency_range = frequency_range

    @property
    def spectrum_resolution(self):
        return self._spectrum_resolution

    @spectrum_resolution.setter
    def spectrum_resolution(self, spectrum_resolution):
        self._spectrum_resolution = spectrum_resolution

    @property
    def power_spectra_algorithm(self):
        return self._power_spectra_algorithm

    @power_spectra_algorithm.setter
    def power_spectra_algorithm(self,power_spectra_algorithm):
        self._power_spectra_algorithm = power_spectra_algorithm

    @property
    def use_NAC(self):
        return self._use_NAC

    @use_NAC.setter
    def use_NAC(self,use_NAC):
        self._use_NAC = use_NAC

    @property
    def band_ranges(self):
        return self._band_ranges
    
    @band_ranges.setter
    def band_ranges(self,band_ranges):
        self._band_ranges = band_ranges

    @property
    def number_of_bins_histogram(self):
        return self._number_of_bins_histogram

    @number_of_bins_histogram.setter
    def number_of_bins_histogram(self, number_of_bins_histogram):
        self._number_of_bins_histogram = number_of_bins_histogram

    @property
    def band_resolution(self):
        return self._band_resolution

    @band_resolution.setter
    def band_resolution(self, band_resolution):
        self._band_resolution = band_resolution

    @property
    def modes_vectors_scale(self):
        return self._modes_vectors_scale

    @modes_vectors_scale.setter
    def modes_vectors_scale(self, modes_vectors_scale):
        self._modes_vectors_scale = modes_vectors_scale

    @property
    def fitting_function(self):
        return self._fitting_function

    @fitting_function.setter
    def fitting_function(self, fitting_function):
        self._fitting_function = fitting_function

    @property
    def zero_padding(self):
        return self._zero_padding

    @zero_padding.setter
    def zero_padding(self, zero_padding):
        self._zero_padding = zero_padding

    @property
    def use_symmetry(self):
        return self._use_symmetry

    @use_symmetry.setter
    def use_symmetry(self, use_symmetry):
        self._use_symmetry = use_symmetry

    @property
    def symmetrize(self):
        return self._symmetrize

    @symmetrize.setter
    def symmetrize(self, symmetrize):
        self._symmetrize = symmetrize

    @property
    def save_renormalized_frequencies(self):
        return self._save_renormalized_frequencies

    @save_renormalized_frequencies.setter
    def save_renormalized_frequencies(self, save_renormalized_frequencies):
        self._save_renormalized_frequenciese = save_renormalized_frequencies

    @property
    def mesh_phonopy(self):
        return self._mesh_phonopy

    @mesh_phonopy.setter
    def mesh_phonopy(self, mesh_phonopy):
        self._mesh_phonopy = mesh_phonopy

    @property
    def use_MD_cell_commensurate(self):
        return self._use_MD_cell_commensurate

    @use_MD_cell_commensurate.setter
    def use_MD_cell_commensurate(self, use_MD_cell_commensurate):
        self._use_MD_cell_commensurate = use_MD_cell_commensurate

# On development ---------
    @property
    def project_on_atom(self):
        return self._project_on_atom

    @project_on_atom.setter
    def project_on_atom(self, project_on_atom):
        self._project_on_atom = project_on_atom
# ----------------------
