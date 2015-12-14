import numpy as np
import sys

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dynaphopy.mem import mem
import dynaphopy.correlation as correlation

from dynaphopy.analysis.fitting import lorentzian, get_error_from_covariance


def progress_bar(progress, label):
    bar_length = 30
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Progress error\r\n"
    if progress < 0:
        progress = 0
        status = "Halt ...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    text = "\r{0}: [{1}] {2:.2f}% {3}".format(label, "#"*block + "-"*(bar_length-block),
                                              progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def get_correlation_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    correlation_vector = []
    progress_bar(0, "Fourier")
    for i in range(vq.shape[1]):
        correlation_vector.append(correlation.correlation_par(test_frequency_range,
                                                              vq[:, i],
                                                              # np.lib.pad(vq[:, i], (2500, 2500), 'constant'),
                                                              trajectory.get_time_step_average(),
                                                              step=parameters.correlation_function_step,
                                                              integration_method=parameters.integration_method))
        progress_bar(float(i+1)/vq.shape[1], "Fourier")

    correlation_vector = np.array(correlation_vector).T

    return correlation_vector


def get_mem_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    correlation_vector = []
    progress_bar(0, "M.E. Method")
    for i in range(vq.shape[1]):
        correlation_vector.append(mem(test_frequency_range,
                                      vq[:, i],
                                      trajectory.get_time_step_average(),
                                      coefficients=parameters.number_of_coefficients_mem))

        progress_bar(float(i+1)/vq.shape[1], "M.E. Method")

    correlation_vector = np.array(correlation_vector).T

    return correlation_vector


def mem_coefficient_scan_analysis(vq, trajectory, parameters):

    mem_full_dict = {}

    for i in range(vq.shape[1]):
        test_frequency_range = parameters.frequency_range
        fit_data = []
        scan_params = []
        power_spectra = []
        progress_bar(0, "M.E. Method")
        for number_of_coefficients in parameters.mem_scan_range:

            power_spectrum = mem(test_frequency_range,
                                 vq[:, i],
                                 trajectory.get_time_step_average(),
                                 coefficients=number_of_coefficients)

            height = np.max(power_spectrum)
            position = test_frequency_range[np.argmax(power_spectrum)]

            try:
                fit_params, fit_covariances = curve_fit(lorentzian,
                                                        test_frequency_range,
                                                        power_spectrum,
                                                        p0=[position, 0.1, height, 0.0])
            except:
                print('Warning: Fitting error, skipping point {0}'.format(number_of_coefficients))
                continue

            maximum = fit_params[2] / (fit_params[1] * np.pi)
            error = get_error_from_covariance(fit_covariances)/maximum
            width = 2.0 * fit_params[1]
            fit_data.append([number_of_coefficients, width, error])
            scan_params.append(fit_params)
            power_spectra.append(power_spectrum)

            progress_bar(float(number_of_coefficients+1)/parameters.mem_scan_range[-1], "M.E. Method")

        fit_data = np.array(fit_data).T

        best_width = np.average(fit_data[1], weights=np.sqrt(1./fit_data[2]))

        best_index = int(np.argmin(fit_data[2]))
        power_spectrum = power_spectra[best_index]

        mem_full_dict.update({i: [power_spectrum, best_width, best_index, fit_data, scan_params]})

    for i in range(vq.shape[1]):

        print "Peak # {0}".format(i+1)
        print("------------------------------------")
        print "Estimated width(FWHM): {0} THz".format(mem_full_dict[i][1])

        fit_data = mem_full_dict[i][3]
        scan_params = mem_full_dict[i][4]
        best_index = mem_full_dict[i][2]

        print "Position: {0} THz".format(scan_params[best_index][0])
        print "Optimum coefficients num: {0}".format(fit_data[0][best_index])
        print "Fitting Error: {0}".format(np.min(fit_data[2]))
        print ("\n")

        plt.figure(i+1)
        plt.suptitle('Peak {0}'.format(i+1))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        ax1.set_xlabel('Number of coefficients')
        ax1.set_ylabel('Width [THz]')
        ax1.set_title('Peak width')
        ax1.plot(fit_data[0], fit_data[1])
        ax1.plot((fit_data[0][0], fit_data[0][-1]), (mem_full_dict[i][1], mem_full_dict[i][1]), 'k-')

        ax2.set_xlabel('Number of coefficients')
        ax2.set_ylabel('RMS^-1')
        ax2.set_title('Fitting error/Max. (RMS)')
        ax2.plot(fit_data[0], np.sqrt(1./fit_data[2]))

        ax3.set_xlabel('Frequency [THz]')
        ax3.set_title('Best curve fitting')
        ax3.plot(test_frequency_range, mem_full_dict[i][0], label='Power spectrum')
        ax3.plot(test_frequency_range,
                 lorentzian(test_frequency_range, *scan_params[best_index]),
                 label='Lorentzian fit')

        plt.show()
