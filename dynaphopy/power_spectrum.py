import numpy as np
import sys

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dynaphopy.mem import mem
import dynaphopy.correlation as correlation
from dynaphopy.analysis.fitting import lorentzian, get_error_from_covariance

unit_conversion = 6.651206285e-4 # u * A^2 * THz -> eV*ps

def progress_bar(progress, label):
    bar_length = 30
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'Progress error\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt ...\r\n'
    if progress >= 1:
        progress = 1
        status = 'Done...\r\n'
    block = int(round(bar_length*progress))
    text = '\r{0}: [{1}] {2:.2f}% {3}'.format(label, '#'*block + '-'*(bar_length-block),
                                              progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


#####################################
#   Fourier transform  method       #
#####################################
def get_fourier_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    psd_vector = []
    if not(parameters.silent):
        progress_bar(0, "Fourier")
    for i in range(vq.shape[1]):
        psd_vector.append(correlation.correlation_par(test_frequency_range,
                                                      vq[:, i],
                                                      # np.lib.pad(vq[:, i], (2500, 2500), 'constant'),
                                                      trajectory.get_time_step_average(),
                                                      step=parameters.correlation_function_step,
                                                      integration_method=parameters.integration_method))
        if not(parameters.silent):
            progress_bar(float(i+1)/vq.shape[1], "Fourier")

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion

#####################################
#   Maximum entropy method method   #
#####################################
def get_mem_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    # Check number of coefficients
    if vq.shape[0] <= parameters.number_of_coefficients_mem+1:
        print('Number of coefficients should be smaller than the number of time steps')
        exit()

    psd_vector = []
    if not(parameters.silent):
        progress_bar(0, 'M. Entropy')
    for i in range(vq.shape[1]):
        psd_vector.append(mem(test_frequency_range,
                              vq[:, i],
                              trajectory.get_time_step_average(),
                              coefficients=parameters.number_of_coefficients_mem))

        if not(parameters.silent):
            progress_bar(float(i+1)/vq.shape[1], 'M. Entropy')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion

# Coefficient analysis for MEM
def mem_coefficient_scan_analysis(vq, trajectory, parameters):

    mem_full_dict = {}

    for i in range(vq.shape[1]):
        test_frequency_range = parameters.frequency_range
        fit_data = []
        scan_params = []
        power_spectra = []
        if not(parameters.silent):
            progress_bar(0, 'M.E. Method')
        for number_of_coefficients in parameters.mem_scan_range:

            power_spectrum = mem(test_frequency_range,
                                 vq[:, i],
                                 trajectory.get_time_step_average(),
                                 coefficients=number_of_coefficients)

            power_spectrum *= unit_conversion

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
            error = get_error_from_covariance(fit_covariances)
            width = 2.0 * fit_params[1]
            area = fit_params[2]/(2.0*np.pi)

            fit_data.append([number_of_coefficients, width, error/maximum, area])
            scan_params.append(fit_params)
            power_spectra.append(power_spectrum)

            if not(parameters.silent):
                progress_bar(float(number_of_coefficients+1)/parameters.mem_scan_range[-1], "M.E. Method")

        fit_data = np.array(fit_data).T

        best_width = np.average(fit_data[1], weights=np.sqrt(1./fit_data[2]))

        best_index = int(np.argmin(fit_data[2]))
        power_spectrum = power_spectra[best_index]

        mem_full_dict.update({i: [power_spectrum, best_width, best_index, fit_data, scan_params]})

    for i in range(vq.shape[1]):

        print ('Peak # {0}'.format(i+1))
        print('------------------------------------')
        print ('Estimated width(FWHM): {0} THz'.format(mem_full_dict[i][1]))

        fit_data = mem_full_dict[i][3]
        scan_params = mem_full_dict[i][4]
        best_index = mem_full_dict[i][2]

        print ('Position: {0} THz'.format(scan_params[best_index][0]))
        print ('Area: {0} eV'.format(fit_data[3][best_index]))
        print ('Optimum coefficients num: {0}'.format(fit_data[0][best_index]))
        print ('Fitting Error: {0}'.format(np.min(fit_data[2])))
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
        ax2.set_ylabel('(RMSD/max)^-1')
        ax2.set_title('Fitting error/Max. (RMSD)')
        ax2.plot(fit_data[0], np.sqrt(1./fit_data[2]))

        ax3.set_xlabel('Frequency [THz]')
        ax3.set_title('Best curve fitting')
        ax3.plot(test_frequency_range, mem_full_dict[i][0], label='Power spectrum')
        ax3.plot(test_frequency_range,
                 lorentzian(test_frequency_range, *scan_params[best_index]),
                 label='Lorentzian fit')

        plt.show()


#####################################
#   FFT method                      #
#####################################

def autocorrelation(x):
    result = np.correlate(x, x, mode='same')/x.size
    return result

#   FFT Numpy

def division_of_data(resolution, number_of_data, time_step):

    piece_size = round(1./(time_step*resolution))
#    print 'N', piece_size

    number_of_pieces = int((number_of_data-1)/piece_size)
#    print'Number of pieces', number_of_pieces
#    print'Size', data.size

    if number_of_pieces > 0:
        interval = (number_of_data - piece_size)/number_of_pieces
    else:
        interval = 0
        number_of_pieces = 1
        piece_size = number_of_data

    pieces = []
    for i in range(number_of_pieces+1):
        ini = int((piece_size/2+i*interval)-piece_size/2)
        fin = int((piece_size/2+i*interval)+piece_size/2)
        pieces.append([ini, fin])

    return pieces


def fft_power(frequency_range, data, time_step):

    pieces = division_of_data(frequency_range[1]-frequency_range[0],
                              data.size,
                              time_step)

    ps = []
    for i_p in pieces:

        data_piece = data[i_p[0]:i_p[1]]

        data_piece = autocorrelation(data_piece)
        ps.append(np.abs(np.fft.fft(data_piece))*time_step/2.0)

    ps = np.average(ps,axis=0)

    freqs = np.fft.fftfreq(data_piece.size, time_step)
    idx = np.argsort(freqs)

    return np.interp(frequency_range, freqs[idx], ps[idx])


def get_fft_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    requested_resolution = test_frequency_range[1]-test_frequency_range[0]
    maximum_resolution = 1./(trajectory.get_time_step_average()*(vq.shape[0]+parameters.zero_padding))
    if requested_resolution < maximum_resolution:
        print('Power spectrum resolution requested unavailable, using maximum: {0:9.6f} THz'.format(maximum_resolution))
        print('If you need higher resolution increase the number of data')

    psd_vector = []
    if not(parameters.silent):
        progress_bar(0, 'FFT')
    for i in range(vq.shape[1]):
        psd_vector.append(fft_power(test_frequency_range,vq[:, i],
                                    trajectory.get_time_step_average()),
                          )

        if not(parameters.silent):
            progress_bar(float(i+1)/vq.shape[1], 'FFT')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion

#   FFTW

def fftw_power(frequency_range, data, time_step):
    import pyfftw
    from multiprocessing import cpu_count

    pieces = division_of_data(frequency_range[1]-frequency_range[0],
                              data.size,
                              time_step)

    ps = []
    for i_p in pieces:

        data_piece = data[i_p[0]:i_p[1]]

        data_piece = autocorrelation(data_piece)
        ps.append(np.abs(pyfftw.interfaces.numpy_fft.fft(data_piece, threads=cpu_count()))*time_step/2.0)

    ps = np.average(ps,axis=0)

    freqs = np.fft.fftfreq(data_piece.size, time_step)
    idx = np.argsort(freqs)

    return np.interp(frequency_range, freqs[idx], ps[idx])


def get_fft_fftw_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    psd_vector = []
    if not(parameters.silent):
        progress_bar(0, 'FFTW')
    for i in range(vq.shape[1]):
        psd_vector.append(fftw_power(test_frequency_range,vq[:, i],
                                     trajectory.get_time_step_average()),
                          )

        if not(parameters.silent):
            progress_bar(float(i+1)/vq.shape[1], 'FFTW')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion
