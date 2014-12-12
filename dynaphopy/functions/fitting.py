import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lorentzian(x, a, b, c, d):
    return c/(np.pi*b*(1.0+((x-a)/b)**2))+d


def get_error_from_covariance(covariance):
    return abs(np.average(covariance))


def phonon_fitting_analysis(original, parameters):

    number_of_coefficients = parameters.number_of_coefficients_mem
    test_frequencies_range = parameters.frequency_range

    for i in range(original.shape[1]):

        power_spectrum = original[:, i]

        height = np.max(power_spectrum)
        position = test_frequencies_range[np.argmax(power_spectrum)]

        try:
            fit_params, fit_covariances = curve_fit(lorentzian,
                                                    test_frequencies_range,
                                                    power_spectrum,
                                                    p0=[position, 0.1, height, 0.0])
        except:
            print('Warning: Fitting error, skipping point!', number_of_coefficients)
            continue

        error = get_error_from_covariance(fit_covariances)
        width = 2.0*fit_params[1]

        print '\nPeak #', i+1
        print('------------------------------------')
        print 'Width(FWHM):', width, 'THz'

        print 'Position:', fit_params[0], 'THz'
        print 'Coefficients:', number_of_coefficients
        print 'Fitting Error:', error

        plt.xlabel('Frequency [THz]')
        plt.title('Curve fitting')

        plt.figure(i)
        plt.suptitle('Phonon '+str(i+1))
        plt.text(fit_params[0], 10, 'Width: ' + str(width), fontsize=12)
        plt.plot(test_frequencies_range, power_spectrum, label='Power spectrum')
        plt.plot(test_frequencies_range, lorentzian(test_frequencies_range, *fit_params), label='Lorentzian fit')
        plt.legend()

    plt.show()
