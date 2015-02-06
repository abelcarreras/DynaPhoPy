import numpy as np
import sys
import multiprocessing

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dynaphopy.mem import mem

from dynaphopy.analysis.fitting import lorentzian, get_error_from_covariance


def progress_bar(progress):
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
    text = "\rM.E. Method: [{0}] {1:.2f}% {2}".format("#"*block + "-"*(bar_length-block),
                                                      progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def mem_worker(n_pos, velocity, trajectory, parameters):
    power_spectrum = mem(parameters.frequency_range,
                         velocity.real,
                         trajectory.get_time_step_average(),
                         coefficients=parameters.number_of_coefficients_mem)

    return {n_pos: power_spectrum}


def get_mem_spectra_par_python(velocity, trajectory, parameters):

    mem_full_dict = {}
    progress_bar(0)

    def log_result(result):
        mem_full_dict.update(result)
        progress_bar(float(len(mem_full_dict))/velocity.shape[1])

#    print ('found',multiprocessing.cpu_count(), 'CPU')
    pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count()-1, 1))
#    print('using:', pool._processes)

    for i in range(velocity.shape[1]):
        pool.apply_async(mem_worker,
                         args=(i, velocity[:, i], trajectory, parameters),
                         callback = log_result)

    pool.close()
    pool.join()

    correlation_vector = np.array([mem_full_dict[i] for i in mem_full_dict.keys()]).T

    return correlation_vector

def get_mem_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    correlation_vector = []
    progress_bar(0)
    for i in range (vq.shape[1]):
        correlation_vector.append(mem(test_frequency_range,
                                      vq[:, i],
                                      trajectory.get_time_step_average(),
                                      coefficients=parameters.number_of_coefficients_mem))

        progress_bar(float(i+1)/vq.shape[1])

    correlation_vector = np.array(correlation_vector).T

    return correlation_vector

def phonon_width_scan_analysis_openmp(vq, trajectory, parameters):

    mem_full_dict = {}

    for i in range(vq.shape[1]):
        test_frequency_range = parameters.frequency_range
        fit_data = []
        scan_params = []
        power_spectra = []
        progress_bar(0)
        for number_of_coefficients in parameters.mem_scan_range:


            power_spectrum = mem(test_frequency_range,
                                          vq[:, i],
                                          trajectory.get_time_step_average(),
                                          coefficients=number_of_coefficients)



            height = np.max(power_spectrum)
            position = test_frequency_range[np.argmax(power_spectrum)]

            try:
                fitParams, fitCovariances = curve_fit(lorentzian,
                                                      test_frequency_range,
                                                      power_spectrum,
                                                      p0=[position, 0.1, height, 0.0])
            except:
                print('Warning: Fitting error, skipping point!', number_of_coefficients)
                continue

            error = get_error_from_covariance(fitCovariances)
            width = 2.0 * fitParams[1]
            fit_data.append([number_of_coefficients, width, error])
            scan_params.append(fitParams)
            power_spectra.append(power_spectrum)

            progress_bar(float(number_of_coefficients+1)/parameters.mem_scan_range[-1])

        fit_data = np.array(fit_data).T

        best_width = np.average(fit_data[1], weights=np.sqrt(1./fit_data[2]))

        best_index = int(np.argmin(fit_data[2]))
        power_spectrum = power_spectra[best_index]

        mem_full_dict.update({i: [power_spectrum, best_width, best_index, fit_data, scan_params]})


    for i in range(vq.shape[1]):

        print "Peak #", i+1
        print("------------------------------------")
        print "Estimated width(FWHM):", mem_full_dict[i][1], "THz"

        fit_data = mem_full_dict[i][3]
        scan_params = mem_full_dict[i][4]
        best_index = mem_full_dict[i][2]

        print "Position:", scan_params[best_index][0], "THz"
        print "Optimum coefficients num:", fit_data[0][best_index]
        print "Fitting Error:", np.min(fit_data[2])
        print ("\n")

        plt.figure(0)
        plt.xlabel('Number of coefficients')
        plt.ylabel('MMSE^-1')
        plt.title('Fitting error (MMSE)')
        plt.plot(fit_data[0], np.sqrt(1./fit_data[2]))

        plt.figure(1)
        plt.xlabel('Number of coefficients')
        plt.ylabel('Width [THz]')
        plt.title('Peak width')
        plt.plot(fit_data[0], fit_data[1])
        plt.plot((fit_data[0][0], fit_data[0][-1]), (mem_full_dict[i][1], mem_full_dict[i][1]), 'k-')

        plt.figure(2)
        plt.xlabel('Frequency [THz]')
        plt.title('Best curve fitting')
        plt.plot(test_frequency_range, mem_full_dict[i][0], label='Power spectrum')
        plt.plot(test_frequency_range,
                 lorentzian(test_frequency_range, *scan_params[best_index]),
                 label='Lorentzian fit')

        plt.show()



def mem_worker_scan(n_pos, velocity, trajectory, parameters):

    test_frequencies_range = parameters.frequency_range

    fit_data = []
    scan_params = []
    power_spectra = []
    for number_of_coefficients in parameters.mem_scan_range:
        power_spectrum = mem(test_frequencies_range,
                             velocity.real,
                             trajectory.get_time_step_average(),
                             coefficients=number_of_coefficients)

        height = np.max(power_spectrum)
        position = test_frequencies_range[np.argmax(power_spectrum)]

        try:
            fitParams, fitCovariances = curve_fit(lorentzian,
                                                  test_frequencies_range,
                                                  power_spectrum,
                                                  p0=[position, 0.1, height, 0.0])
        except:
            print('Warning: Fitting error, skipping point!', number_of_coefficients)
            continue

        error = get_error_from_covariance(fitCovariances)
        width = 2.0 * fitParams[1]

        fit_data.append([number_of_coefficients, width, error])
        scan_params.append(fitParams)
        power_spectra.append(power_spectrum)

    fit_data = np.array(fit_data).T

    best_width = np.average(fit_data[1], weights=np.sqrt(1./fit_data[2]))

    best_index = int(np.argmin(fit_data[2]))
    power_spectrum = power_spectra[best_index]

    return {n_pos: [power_spectrum, best_width, best_index, fit_data, scan_params]}


def phonon_width_scan_analysis(vq, trajectory, parameters):

    mem_full_dict = {}
    progress_bar(0)

    def log_result(result):
        mem_full_dict.update(result)
        progress_bar(float(len(mem_full_dict))/vq.shape[1])

#    print ('found',multiprocessing.cpu_count(), 'CPU')
    pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count()-1, 1))
#    pool = multiprocessing.Pool(processes=1)

#    print('using:', pool._processes)
    for i in range(vq.shape[1]):
        pool.apply_async(mem_worker_scan,
                         args=(i, vq[:, i], trajectory, parameters),
                         callback = log_result)

    pool.close()
    pool.join()

    test_frequencies_range = parameters.frequency_range

    for i in range(vq.shape[1]):

        print "Peak #", i+1
        print("------------------------------------")
        print "Estimated width(FWHM):", mem_full_dict[i][1], "THz"

        fit_data = mem_full_dict[i][3]
        scan_params = mem_full_dict[i][4]
        best_index = mem_full_dict[i][2]

        print "Position:", scan_params[best_index][0], "THz"
        print "Optimum coefficients num:", fit_data[0][best_index]
        print "Fitting Error:", np.min(fit_data[2])
        print ("\n")

        plt.figure(0)
        plt.xlabel('Number of coefficients')
        plt.ylabel('Fitting error^-1')
        plt.title('fitting error ^-1')
        plt.plot(fit_data[0], np.sqrt(1./fit_data[2]))

        plt.figure(1)
        plt.xlabel('Number of coefficients')
        plt.ylabel('Width [THz]')
        plt.title('Peak width')
        plt.plot(fit_data[0], fit_data[1])
        plt.plot((fit_data[0][0], fit_data[0][-1]), (mem_full_dict[i][1], mem_full_dict[i][1]), 'k-')

        plt.figure(2)
        plt.xlabel('Frequency [THz]')
        plt.title('Best curve fitting')
        plt.plot(test_frequencies_range, mem_full_dict[i][0], label='Power spectrum')
        plt.plot(test_frequencies_range,
                 lorentzian(test_frequencies_range, *scan_params[best_index]),
                 label='Lorentzian fit')

        plt.show()
