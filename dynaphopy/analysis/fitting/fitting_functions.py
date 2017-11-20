import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

h_planck = 4.135667662e-3  # eV/ps
h_planck_bar = 6.58211951e-4  # eV/ps
kb_boltzmann = 8.6173324e-5  # eV/K


def get_standard_errors_from_covariance(covariance):
   # return np.linalg.eigvals(covariance)
    return np.sqrt(np.diag(covariance))
    #return np.sqrt(np.trace(covariance))


class Lorentzian:
    def __init__(self,
                 test_frequencies_range,
                 power_spectrum,
                 guess_position=None,
                 guess_height=None):

        self.test_frequencies_range = test_frequencies_range
        self.power_spectrum = power_spectrum
        self.guess_pos = guess_position
        self.guess_height = guess_height

        self._fit_params = None
        self._fit_covariances = None

        self.curve_name = 'Lorentzian'

    def _function(self, x, a, b, c, d):
        """Lorentzian function
        x: frequency coordinate
        a: peak position
        b: half width
        c: area proportional parameter
        d: base line
        """
        return c/(np.pi*b*(1.0+((x - a)/b)**2))+d

    def get_fitting_parameters(self):

        if self._fit_params is None:
            if self.guess_pos is None or self.guess_height is None:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum)
            else:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum,
                                                        p0=[self.guess_pos, 0.1, self.guess_height, 0.0])
            self._fit_covariances = fit_covariances
            self._fit_params = fit_params

        return self._fit_params, self._fit_covariances

    def get_fitting(self):
        from scipy.integrate import quad

        try:

            fit_params, fit_covariances = self.get_fitting_parameters()

            maximum = fit_params[2]/(fit_params[1]*np.pi)
            width = 2.0*fit_params[1]
            frequency = fit_params[0]
            area = fit_params[2]

            standard_errors = get_standard_errors_from_covariance(fit_covariances)

            global_error = np.average(standard_errors[:2])/np.sqrt(area)
            if np.isnan(global_error):
                raise RuntimeError

            #error = get_error_from_covariance(fit_covariances)
            base_line = fit_params[3]


            return {'maximum': maximum,
                    'width': width,
                    'peak_position': frequency,
                    'standard_errors': standard_errors,
                    'global_error': global_error,
                    'area': area,
                    'base_line': base_line,
                    'all_good': True}

        except RuntimeError:
            return {'all_good': False}

    def get_curve(self, frequency_range):
        return self._function(frequency_range, *self.get_fitting_parameters()[0])


class Lorentzian_asymmetric:
    def __init__(self,
                 test_frequencies_range,
                 power_spectrum,
                 guess_position=None,
                 guess_height=None):

        self.test_frequencies_range = test_frequencies_range
        self.power_spectrum = power_spectrum
        self.guess_pos = guess_position
        self.guess_height = guess_height

        self._fit_params = None
        self._fit_covariances = None

        self.curve_name = 'Assym. Lorentzian'

    def _g_a (self, x, a, b, s):
        """Asymmetric width term
        x: frequency coordinate
        a: peak position
        b: half width
        s: asymmetry parameter
        """
        return 2*b/(1.0+np.exp(s*(x-a)))

    def _function(self, x, a, b, c, d, s):
        """Lorentzian asymmetric function
        x: frequency coordinate
        a: peak position
        b: half width
        c: area proportional parameter
        d: base line
        s: asymmetry parameter
        """
        return c/(np.pi*self._g_a(x, a, b, s)*(1.0+((x-a)/(self._g_a(x, a, b, s)))**2))+d

    def get_fitting_parameters(self):

        if self._fit_params is None:
            if self.guess_pos is None or self.guess_height is None:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum)
            else:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum,
                                                        p0=[self.guess_pos, 0.1, self.guess_height, 0.0, 0.0])
            self._fit_covariances = fit_covariances
            self._fit_params = fit_params

        return self._fit_params, self._fit_covariances


    def get_fitting(self):
        from scipy.integrate import quad

        try:

            fit_params, fit_covariances = self.get_fitting_parameters()

            peak_pos = minimize_scalar(lambda x: -self._function(x, *fit_params), fit_params[0],
                                       bounds=[self.test_frequencies_range[0], self.test_frequencies_range[-1]],
                                       method='bounded')

            frequency = peak_pos["x"]
            maximum = -peak_pos["fun"]
            width = 2.0 * self._g_a(frequency, fit_params[0], fit_params[1], fit_params[4])
            asymmetry = fit_params[4]

            area, error_integration = quad(self._function, 0, self.test_frequencies_range[-1],
                                           args=tuple(fit_params),
                                           epsabs=1e-8)
        #    area = fit_params[2]

            standard_errors = get_standard_errors_from_covariance(fit_covariances)
            global_error = np.average(standard_errors[:2])/np.sqrt(area)
            if np.isnan(global_error):
                raise RuntimeError

            #error = get_error_from_covariance(fit_covariances)
            base_line = fit_params[3]

            return {'maximum': maximum,
                    'width': width,
                    'peak_position': frequency,
                    'global_error': global_error,
                    'area': area,
                    'base_line': base_line,
                    'asymmetry': asymmetry,
                    'all_good': True}

        except RuntimeError:
            return {'all_good': False}

    def get_curve(self, frequency_range):
        return self._function(frequency_range, *self.get_fitting_parameters()[0])


class Damped_harmonic:
    def __init__(self,
                 test_frequencies_range,
                 power_spectrum,
                 guess_position=None,
                 guess_height=None):

        self.test_frequencies_range = test_frequencies_range
        self.power_spectrum = power_spectrum
        self.guess_pos = guess_position
        self.guess_height = guess_height

        self._fit_params = None
        self._fit_covariances = None

        self.curve_name = 'Damped Harm. Osc.'

    def _function(self, x, a, b, c, d):
        """Damped harmonic oscillator PS function
        x: frequency coordinate
        a: peak position
        b: half width
        c: area proportional parameter
        d: base line
        """
        return c/((a**2-x**2)**2 + (b*x)**2)+d

    def get_fitting_parameters(self):

        if self._fit_params is None:
            if self.guess_pos is None or self.guess_height is None:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum)
            else:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum,
                                                        p0=[self.guess_pos, 0.1, self.guess_height, 0.0])
            self._fit_covariances = fit_covariances
            self._fit_params = fit_params

        return self._fit_params, self._fit_covariances

    def get_fitting(self):
        from scipy.integrate import quad

        try:

            fit_params, fit_covariances = self.get_fitting_parameters()

            self._fit_params = fit_params

            width = abs(fit_params[1])

            maximum = fit_params[2]/(width*np.pi)
            frequency = fit_params[0]
            maximum = self.get_curve(frequency)


            area, error_integration = quad(self._function, 0, self.test_frequencies_range[-1],
                                           args=tuple(fit_params),
                                           epsabs=1e-8)
#            area = fit_params[2]*np.pi/(fit_params[0]**3*width)

            standard_errors = get_standard_errors_from_covariance(fit_covariances)

            global_error = np.average(standard_errors[:2])/np.sqrt(area)
            if np.isnan(global_error):
                raise RuntimeError

            base_line = fit_params[3]

            return {'maximum': maximum,
                    'width': width,
                    'peak_position': frequency,
                    'global_error': global_error,
                    'area': area,
                    'base_line': base_line,
                    'all_good': True}

        except RuntimeError:
            return {'all_good': False}

    def get_curve(self, frequency_range):
        return self._function(frequency_range, *self.get_fitting_parameters()[0])


class Gaussian_function:
    def __init__(self,
                 test_frequencies_range,
                 power_spectrum,
                 guess_position=None,
                 guess_height=None):

        self.test_frequencies_range = test_frequencies_range
        self.power_spectrum = power_spectrum
        self.guess_pos = guess_position
        self.guess_height = guess_height

        self._fit_params = None
        self._fit_covariances = None

        self.curve_name = 'Gaussian dist.'

    def _function(self, x, a, b, c, d):
        """Gaussian PDF function
        x: coordinate
        a: peak position
        b: deviation (sigma)
        c: area proportional parameter
        d: base line
        """
        return c/b*np.sqrt(2*np.pi)*np.exp(-(x-a)**2/(2*b**2))+d

    def get_fitting_parameters(self):

        if self._fit_params is None:
            if self.guess_pos is None or self.guess_height is None:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum)
            else:
                fit_params, fit_covariances = curve_fit(self._function,
                                                        self.test_frequencies_range,
                                                        self.power_spectrum,
                                                        p0=[self.guess_pos, 0.1, self.guess_height, 0.0])
            self._fit_covariances = fit_covariances
            self._fit_params = fit_params

        return self._fit_params, self._fit_covariances

    def get_fitting(self):
        from scipy.integrate import quad

        try:

            fit_params, fit_covariances = self.get_fitting_parameters()

            self._fit_params = fit_params

            width = abs(fit_params[1])

            frequency = fit_params[0]
            maximum = self.get_curve(frequency)

            area, error_integration = quad(self._function, 0, self.test_frequencies_range[-1],
                                           args=tuple(fit_params),
                                           epsabs=1e-8)
#            area = fit_params[2]*np.pi/(fit_params[0]**3*width)

            standard_errors = get_standard_errors_from_covariance(fit_covariances)

            global_error = np.average(standard_errors[:2])/np.sqrt(area)
            if np.isnan(global_error):
                raise RuntimeError

            base_line = fit_params[3]

            return {'maximum': maximum,
                    'width': width,
                    'peak_position': frequency,
                    'global_error': global_error,
                    'area': area,
                    'base_line': base_line,
                    'all_good': True}

        except RuntimeError:
            return {'all_good': False}

    def get_curve(self, frequency_range):
        return self._function(frequency_range, *self.get_fitting_parameters()[0])



fitting_functions = {
    0: Lorentzian,
    1: Lorentzian_asymmetric,
    2: Damped_harmonic,
}

# Test for automatic detection (order can change)
# import sys, inspect
# list_fitting = inspect.getmembers(sys.modules[__name__], inspect.isclass)
# Fitting_functions = {}
# for i, p in enumerate(list_fitting):
#    Fitting_functions[i] = p[1]
