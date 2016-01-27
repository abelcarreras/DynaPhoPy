import numpy as np

#This class contains all the default parameters for DynaPhoPy

class Parameters:

    def __init__(self,
                 # General
                 silent=False,

                 # Projections
                 reduced_q_vector=(0, 0, 0),  # default reduced wave vector

                 # Maximum Entropy Method
                 number_of_coefficients_mem=1000,
                 mem_scan_range=np.array(np.linspace(40, 2000, 100), dtype=int),

                 # Fourier transform Method
                 correlation_function_step=10,
                 integration_method = 1,  # 0: Trapezoid  1:Rectangles

                 # Fast Fourier tranform Method
                 zero_padding=0,

                 # Power spectra
                    # 0: Correlation functions parallel (OpenMP) [Recommended]
                    # 1: Maximum Entropy Method parallel (OpenMP) [Recommended]
                    # 2: FFT via numpy
                    # 3: FFT via FFTW
                 power_spectra_algorithm=1,
                 use_asymmetric_peaks=False,
                 frequency_range=np.linspace(0, 40, 500),

                 # Phonon dispersion diagram
                 use_NAC = False,
                 band_ranges=([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]]),
                 number_of_bins_histogram = 30,

                 # Force constants
                 symmetrize = False,
                 degenerate = True,

                 # Modes (eigenvectors) display
                 modes_vectors_scale=10
                 ):

        self._silent = silent
        self._number_of_coefficients_mem=number_of_coefficients_mem
        self._mem_scan_range=mem_scan_range
        self._correlation_function_step = correlation_function_step
        self._integration_method = integration_method
        self._power_spectra_algorithm = power_spectra_algorithm
        self._use_asymmetric_peaks = use_asymmetric_peaks
        self._zero_padding = zero_padding
        self._frequency_range = frequency_range
        self._reduced_q_vector = reduced_q_vector
        self._use_NAC = use_NAC
        self._band_ranges = band_ranges
        self._number_of_bins_histogram = number_of_bins_histogram

        self._symmetrize = symmetrize
        self._degenerate = degenerate

        self._modes_vectors_scale = modes_vectors_scale

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
    def frequency_range(self,frequency_range):
        self._frequency_range = frequency_range

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
    def modes_vectors_scale(self):
        return self._modes_vectors_scale

    @modes_vectors_scale.setter
    def modes_vectors_scale(self, modes_vectors_scale):
        self._modes_vectors_scale = modes_vectors_scale

    @property
    def use_asymmetric_peaks(self):
        return self._use_asymmetric_peaks

    @use_asymmetric_peaks.setter
    def use_asymmetric_peaks(self, use_asymmetric_peaks):
        self._use_asymmetric_peaks = use_asymmetric_peaks

    @property
    def zero_padding(self):
        return self._zero_padding

    @zero_padding.setter
    def zero_padding(self, zero_padding):
        self._zero_padding = zero_padding

    @property
    def degenerate(self):
        return self._symmetrize

    @degenerate.setter
    def degenerate(self, degenerate):
        self._degenerate = degenerate

    @property
    def symmetrize(self):
        return self._symmetrize

    @symmetrize.setter
    def symmetrize(self, symmetrize):
        self._symmetrize = symmetrize
