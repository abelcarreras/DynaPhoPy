import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

h_planck = 39.90310 # A^2 * u / ps
kb_bolzman = 0.831446 # u * A^2 / ( ps^2 * K )

def lorentzian(x, a, b, c, d):
    return c/(np.pi*b*(1.0+((x-a)/b)**2))+d


def get_error_from_covariance(covariance):
    return np.sqrt(np.sum(np.linalg.eigvals(covariance)**2))
  #  return np.sqrt(np.trace(covariance))


def phonon_fitting_analysis(original, test_frequencies_range, harmonic_frequencies=None, show_plots=True):

#    number_of_coefficients = parameters.number_of_coefficients_mem
#    test_frequencies_range = parameters.frequency_range

    widths = []
    positions = []
    shifts = []

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
            print('Warning: Fitting error in phonon {0}. Try increasing the spectrum point density'.format(i))
            continue

        maximum = fit_params[2]/(fit_params[1]*np.pi)
        error = get_error_from_covariance(fit_covariances)
        width = 2.0*fit_params[1]
        area = fit_params[2]
        frequency = fit_params[0]

        total_integral = np.trapz(power_spectrum, x=test_frequencies_range)

        #Calculated properties
        dt_Q2_lor = 2 * 2 * area
        dt_Q2_tot = 2 * 2 * total_integral

        Q2_lor = dt_Q2_lor / pow(frequency * 2 * np.pi, 2)
        Q2_tot = dt_Q2_tot / pow(frequency * 2 * np.pi,2)

 #       occupancy_lor = Q2_lor * frequency / h_planck * pow(2 * np.pi, 2) - 0.5
        occupancy_lor = dt_Q2_lor / (frequency * h_planck) - 0.5
        occupancy_tot = dt_Q2_tot / (frequency * h_planck) - 0.5


        #Print section
        print '\nPeak #', i+1
        print('------------------------------------')
        print 'Width (FWHM):           ', width, 'THz'
        print 'Position:               ', frequency, 'THz'
        print 'Area (1/2<K>) (lor):    ', area, 'u * Angstrom^2 / ps^2'             # Kinetic energy
        print 'Area (1/2<K>) (tot):    ', total_integral, 'u * Angstrom^2 / ps^2'   # Kinetic energy
        print '<|dQ/dt|^2> (lor):      ', dt_Q2_lor, 'u * Angstrom^2 / ps^2'        # Total energy
        print '<|dQ/dt|^2> (tot):      ', dt_Q2_tot, 'u * Angstrom^2 / ps^2'        # Total energy
 #       print '<|Q|^2> (lor):          ', Q2_lor, 'u * Angstrom^2'
 #       print '<|Q|^2> (tot):          ', Q2_tot, 'u * Angstrom^2'
        print 'Occupation number(lor): ', occupancy_lor
        print 'Occupation number(tot): ', occupancy_tot
        print 'Fit temperature (lor)   ', dt_Q2_lor / kb_bolzman, 'K'
        print 'Fit temperature (tot)   ', dt_Q2_tot / kb_bolzman, 'K'

        print 'Maximum height:         ', maximum, 'u * Angstrom^2 * 2 pi / ps'
        if harmonic_frequencies is not None:
            print 'Frequency shift:        ', frequency - harmonic_frequencies[i], 'THz'
        print 'Fit Error/Max. (RMS):   ', error/maximum

        positions.append(frequency)
        widths.append(width)

        if show_plots:
            plt.figure(i+1)

            plt.xlabel('Frequency [THz]')
            plt.ylabel('$u * \AA^2 \pi/ ps$')

            plt.title('Curve fitting')

            plt.suptitle('Phonon '+str(i+1))
            plt.text(fit_params[0], height/2, 'Width: ' + "{:10.4f}".format(width),
                     fontsize=12)

            plt.plot(test_frequencies_range, power_spectrum,
                     label='Power spectrum')
            plt.plot(test_frequencies_range, lorentzian(test_frequencies_range, *fit_params),
                     label='Lorentzian fit',
                     linewidth=3)

            plt.legend()


    if show_plots:
        plt.show()

    return positions, widths