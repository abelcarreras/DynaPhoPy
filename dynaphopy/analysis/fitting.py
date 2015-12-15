import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

h_planck = 4.135667662E-3  # eV/ps
kb_bolzman = 8.6173324E-5  # eV/K


def lorentzian(x, a, b, c, d):
    return c/(np.pi*b*(1.0+((x-a)/b)**2))+d


def get_error_from_covariance(covariance):
  #  return np.sqrt(np.sum(np.linalg.eigvals(covariance)**2))
    return np.sqrt(np.trace(covariance))


def phonon_fitting_analysis(original, test_frequencies_range, harmonic_frequencies=None, show_plots=True):

    widths = []
    positions = []

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
            positions.append(0)
            widths.append(0)
            continue

        maximum = fit_params[2]/(fit_params[1]*np.pi)
        error = get_error_from_covariance(fit_covariances)
        width = 2.0*fit_params[1]
        area = fit_params[2] / ( 2 * np.pi)
        frequency = fit_params[0]

        total_integral = np.trapz(power_spectrum, x=test_frequencies_range)/ (2 * np.pi)

        #Calculated properties
        dt_Q2_lor = 2 * 2 * area
        dt_Q2_tot = 2 * 2 * total_integral

        #Only in harmonic approximation
        Q2_lor = dt_Q2_lor / pow(frequency * 2 * np.pi, 2)
        Q2_tot = dt_Q2_tot / pow(frequency * 2 * np.pi,2)

        occupancy_lor = dt_Q2_lor / (frequency * h_planck) - 0.5
        occupancy_tot = dt_Q2_tot / (frequency * h_planck) - 0.5

        #Print section
        print '\nPeak #', i+1
        print('------------------------------------')
        print 'Width (FWHM):             ', width, 'THz'
        print 'Position:                 ', frequency, 'THz'
        print 'Area (1/2<K>) (Loretzian):', area, 'eV'             # 1/2 Kinetic energy
        print 'Area (1/2<K>) (Total):    ', total_integral, 'eV'   # 1/2 Kinetic energy
        print '<|dQ/dt|^2>      :        ', dt_Q2_lor, 'eV'        # Kinetic energy
 #       print '<|dQ/dt|^2> (tot):        ', dt_Q2_tot, 'eV'        # Kinetic energy
 #       print '<|Q|^2> (lor):          ', Q2_lor, 'u * Angstrom^2'
 #       print '<|Q|^2> (tot):          ', Q2_tot, 'u * Angstrom^2'
        print 'Occupation number:        ', occupancy_lor
 #       print 'Occupation number(tot): ', occupancy_tot
        print 'Fit temperature           ', dt_Q2_lor / kb_bolzman, 'K'
 #       print 'Fit temperature (tot)   ', dt_Q2_tot / kb_bolzman, 'K'

        print 'Maximum height:           ', maximum, 'eV * ps'
        if harmonic_frequencies is not None:
            print 'Frequency shift:          ', frequency - harmonic_frequencies[i], 'THz'
        print 'Fit Error(RMSD)/ Max. :   ', error/maximum

        positions.append(frequency)
        widths.append(width)

        if show_plots:
            plt.figure(i+1)

            plt.xlabel('Frequency [THz]')
            plt.ylabel('eV * ps')

            plt.title('Curve fitting')

            plt.suptitle('Phonon {0}'.format(i+1))
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