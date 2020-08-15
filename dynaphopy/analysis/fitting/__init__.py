import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from dynaphopy.analysis.fitting import fitting_functions

h_planck = 4.135667662e-3  # eV/ps
h_planck_bar = h_planck/(2.0*np.pi)  # eV/ps
kb_boltzmann = 8.6173324e-5  # eV/K


def degenerate_sets(freqs, cutoff=1e-4):
    indices = []
    done = []
    for i in range(len(freqs)):
        if i in done:
            continue
        else:
            f_set = [i]
            done.append(i)
        for j in range(i + 1, len(freqs)):
            if (np.abs(freqs[f_set] - freqs[j]) < cutoff).any():
                f_set.append(j)
                done.append(j)
        indices.append(f_set[:])

    return indices


def average_phonon(index, data, degeneracy):
    for i, set in enumerate(degeneracy):
        if index in set:
            return np.average([data[:, j] for j in degeneracy[i]], axis=0)


def phonon_fitting_analysis(original, ps_frequencies,
                            harmonic_frequencies=None,
                            thermal_expansion_shift=None,
                            fitting_function_type=0,
                            show_plots=True,
                            use_degeneracy=True,
                            show_occupancy=True):

    widths = []
    positions = []
    errors = []
    dt_Q2_s = []

    for i in range(original.shape[1]):

        if use_degeneracy:
            degeneracy = degenerate_sets(harmonic_frequencies)
            power_spectrum = average_phonon(i, original, degeneracy)
        else:
            power_spectrum = original[:, i]

        guess_height = np.max(power_spectrum)
        guess_position = ps_frequencies[np.argmax(power_spectrum)]

        Fitting_function_class = fitting_functions.fitting_functions[fitting_function_type]
        fitting_function = Fitting_function_class(ps_frequencies,
                                                  power_spectrum,
                                                  guess_height=guess_height,
                                                  guess_position=guess_position)

        fitting_parameters = fitting_function.get_fitting()

        if not fitting_parameters['all_good']:
            positions.append(0)
            widths.append(0)
            errors.append(0)
            dt_Q2_s.append(0)
            print ('Warning: Fitting not successful in peak #{0}'.format(i+1))
            continue

        position = fitting_parameters['peak_position']
        area = fitting_parameters['area']
        width = fitting_parameters['width']
        base_line = fitting_parameters['base_line']
        maximum = fitting_parameters['maximum']
        error = fitting_parameters['global_error']

        total_integral = simps(power_spectrum, x=ps_frequencies)

        # Calculated properties
        dt_Q2_lor = 2 * area
        dt_Q2_tot = 2 * total_integral

        try:
            # Only within harmonic approximation
            # Q2_lor = dt_Q2_lor / pow(position * 2 * np.pi, 2)
            # Q2_tot = dt_Q2_tot / pow(position * 2 * np.pi,2)

            occupancy_lor = dt_Q2_lor / (position * h_planck_bar) - 0.5
            occupancy_tot = dt_Q2_tot / (position * h_planck_bar) - 0.5

            # fit_temperature = dt_Q2_lor / kb_boltzmann  # High temperature limit
            fit_temperature = h_planck_bar * position / (kb_boltzmann * np.log((1.0 / occupancy_lor + 1.0)))
            fit_temperature_tot = h_planck_bar * position / (kb_boltzmann * np.log((1.0 / occupancy_tot + 1.0)))

        except RuntimeWarning:
            # This warning happens in acoustic branches at gamma point because the peak
            # position is zero (If this warning is raised at GAMMA it is OK!)
            occupancy_lor = np.nan
            occupancy_tot = np.nan
            fit_temperature = np.nan
            fit_temperature_tot = np.nan
            error = np.nan
            pass

        #Print section
        print ('\nPeak # {0}'.format(i+1))
        print ('----------------------------------------------')
        print ('Width                      {0:15.6f} THz'.format(width))
        print ('Position                   {0:15.6f} THz'.format(position))
        print ('Area (<K>)    ({0:.10s}) {1:15.6f} eV'.format(fitting_function.curve_name, area))  # Kinetic energy
        print ('Area (<K>)    (Total)      {0:15.6f} eV'.format(total_integral))   # 1/2 Kinetic energy
        print ('<|dQ/dt|^2>                {0:15.6f} eV'.format(dt_Q2_lor))        # Kinetic energy
        # print '<|dQ/dt|^2> (tot):        ', dt_Q2_tot, 'eV'        # Kinetic energy
        # print '<|Q|^2> (lor):          ', Q2_lor, 'eV' #  potential energy
        # print '<|Q|^2> (tot):          ', Q2_tot, 'eV' #  potential energy
        if show_occupancy:
            print ('Occupation number          {0:15.6f}'.format(occupancy_lor))
            print ('Fit temperature            {0:15.6f} K'.format(fit_temperature))
            #print ('Fit temperature (Total)    {0:15.6f} K'.format(fit_temperature_tot))

        print ('Base line                  {0:15.6f} eV * ps'.format(base_line))
        print ('Maximum height             {0:15.6f} eV * ps'.format(maximum))
        print ('Fitting global error       {0:15.6f}'.format(error))

        if 'asymmetry' in fitting_parameters:
            asymmetry = fitting_parameters['asymmetry']
            print ('Peak asymmetry             {0:15.6f}'.format(asymmetry))

        if harmonic_frequencies is not None:
            print ('Frequency shift            {0:15.6f} THz'.format(position - harmonic_frequencies[i]))

        if thermal_expansion_shift is not None:
            print ('Frequency shift (+T. exp.) {0:15.6f} THz'.format(position - harmonic_frequencies[i] + thermal_expansion_shift[i]))
            position += thermal_expansion_shift[i]

        positions.append(position)
        widths.append(width)
        errors.append(error/maximum)
        dt_Q2_s.append(dt_Q2_lor)

        if show_plots:
            plt.figure(i+1)

            plt.xlabel('Frequency [THz]')
            plt.ylabel('eV * ps')

            plt.title('Curve fitting')

            plt.suptitle('Phonon {0}'.format(i+1))
            plt.text(position+width, guess_height/2, 'Width: ' + "{:10.4f}".format(width),
                     fontsize=12)

            plt.plot(ps_frequencies, power_spectrum,
                     label='Power spectrum')

            plt.plot(ps_frequencies, fitting_function.get_curve(ps_frequencies),
                     label=fitting_function.curve_name,
                     linewidth=3)

#            plt.plot(test_frequencies_range, dumped_harmonic(test_frequencies_range, *fit_params[:4]),
#                     label='Lorentzian fit',
#                     linewidth=3)

#            plt.plot(test_frequencies_range, lorentzian_asymmetric(test_frequencies_range, *fit_params),
#                     label=('As. Lorentzian fit' if asymmetric_peaks else 'Lorentizian fit'),
#                     linewidth=3)

            plt.axvline(x=position, color='k', ls='dashed')
            plt.ylim(bottom=0)
            plt.xlim([ps_frequencies[0], ps_frequencies[-1]])
            try:
                plt.fill_between([position-width/2, position+width/2], plt.gca().get_ylim()[1], color='red', alpha='0.2')
            except TypeError:
                pass
            plt.legend()

    if show_plots:
        plt.show()

    return {'positions': positions,
            'widths': widths,
            'error': errors,
            'dt_Q2': dt_Q2_s}