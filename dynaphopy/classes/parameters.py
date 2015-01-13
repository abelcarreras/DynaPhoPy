import numpy as np

#This class contains all the default parameters for DynaPhoPy

class Parameters:

    def __init__(self,
                 #Cutting
                 last_steps=2000,  # default number of last steps used to perform the calculations

                 #Projections
                 reduced_q_vector=(0, 0, 0),  # default reduced wave vector

                 #Maximum Entropy Method
                 number_of_coefficients_mem=300,
                 mem_scan_range=np.array(np.linspace(40, 1000, 100),dtype=int),

                 #Correlation Method
                 correlation_function_step=10,
                 integration_method = 1,  # 0: Trapezoid  1:Rectangles

                 #Power spectra
                    # 0: Correlation functions parallel(python)
                    # 1: Maximum Entropy Method
                    # 2: Correlation functions serial
                    # 3: Correlation functions parallel (OpenMP)
                 power_spectra_algorithm=3,
                 frequency_range=np.linspace(0, 40, 500),

                 #Phonon dispersion diagram
                 use_NAC = False,
                 band_ranges=((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
                 ):

        self._last_steps = last_steps
        self._number_of_coefficients_mem=number_of_coefficients_mem
        self._mem_scan_range=mem_scan_range
        self._correlation_function_step = correlation_function_step
        self._integration_method = integration_method
        self._power_spectra_algorithm = power_spectra_algorithm
        self._frequency_range = frequency_range
        self._reduced_q_vector = reduced_q_vector
        self._use_NAC = use_NAC
        self._band_ranges = band_ranges


    #Properties
    @property
    def last_steps(self):
        return self._last_steps

    @last_steps.setter
    def last_steps(self, last_steps):
        self._last_steps = last_steps

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




