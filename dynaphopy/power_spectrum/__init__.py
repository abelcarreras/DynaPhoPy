import numpy as np
import sys

import matplotlib.pyplot as plt

from dynaphopy.power_spectrum import mem
from dynaphopy.power_spectrum import correlation

unit_conversion = 0.00010585723  # u * A^2 * THz -> eV*ps


def _progress_bar(progress, label):
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


def _division_of_data(resolution, number_of_data, time_step):

    piece_size = round(1./(time_step*resolution))
    number_of_pieces = int((number_of_data-1)/piece_size)

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


#############################################
#   Fourier transform - direct method       #
#############################################
def get_fourier_direct_power_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    psd_vector = []
    if not parameters.silent:
        _progress_bar(0, "Fourier")
    for i in range(vq.shape[1]):
        psd_vector.append(correlation.correlation_par(test_frequency_range,
                                                      vq[:, i],
                                                      # np.lib.pad(vq[:, i], (2500, 2500), 'constant'),
                                                      trajectory.get_time_step_average(),
                                                      step=parameters.correlation_function_step,
                                                      integration_method=parameters.integration_method))
        if not parameters.silent:
            _progress_bar(float(i + 1) / vq.shape[1], "Fourier")

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion


#####################################
#   Maximum entropy method method   #
#####################################
def get_mem_power_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    # Check number of coefficients
    if vq.shape[0] <= parameters.number_of_coefficients_mem+1:
        print('Number of coefficients should be smaller than the number of time steps')
        exit()

    psd_vector = []
    if not parameters.silent:
        _progress_bar(0, 'M. Entropy')
    for i in range(vq.shape[1]):
        psd_vector.append(mem.mem(np.ascontiguousarray(test_frequency_range),
                                  np.ascontiguousarray(vq[:, i]),
                                  trajectory.get_time_step_average(),
                                  coefficients=parameters.number_of_coefficients_mem))

        if not parameters.silent:
            _progress_bar(float(i + 1) / vq.shape[1], 'M. Entropy')

    psd_vector = np.nan_to_num(np.array(psd_vector).T)

    return psd_vector * unit_conversion


#####################################
#    Coefficient analysis (MEM)     #
#####################################
def mem_coefficient_scan_analysis(vq, trajectory, parameters):
    from dynaphopy.analysis.fitting import fitting_functions

    mem_full_dict = {}

    for i in range(vq.shape[1]):
        test_frequency_range = parameters.frequency_range
        fit_data = []
        scan_params = []
        power_spectra = []
        if not parameters.silent:
            _progress_bar(0, 'ME Coeff.')
        for number_of_coefficients in parameters.mem_scan_range:

            power_spectrum = mem.mem(np.ascontiguousarray(test_frequency_range),
                                     np.ascontiguousarray(vq[:, i]),
                                     trajectory.get_time_step_average(),
                                     coefficients=number_of_coefficients)

            power_spectrum *= unit_conversion

            guess_height = np.max(power_spectrum)
            guess_position = test_frequency_range[np.argmax(power_spectrum)]

            Fitting_function_class = fitting_functions.fitting_functions[parameters.fitting_function]

            if np.isnan(power_spectrum).any():
                print('Warning: power spectrum error, skipping point {0}'.format(number_of_coefficients))
                continue

          #  Fitting_curve = fitting_functions[parameters.fitting_function]
            fitting_function = Fitting_function_class(test_frequency_range,
                                                      power_spectrum,
                                                      guess_height=guess_height,
                                                      guess_position=guess_position)

            fitting_parameters = fitting_function.get_fitting()


            if not fitting_parameters['all_good']:
                print('Warning: Fitting error, skipping point {0}'.format(number_of_coefficients))
                continue

#            frequency = fitting_parameters['peak_position']
            area = fitting_parameters['area']
            width = fitting_parameters['width']
#            base_line = fitting_parameters['base_line']
            maximum = fitting_parameters['maximum']
            error = fitting_parameters['global_error']

            fit_data.append([number_of_coefficients, width, error, area])
            scan_params.append(fitting_function._fit_params)
            power_spectra.append(power_spectrum)
            if not(parameters.silent):
                _progress_bar(float(number_of_coefficients + 1) / parameters.mem_scan_range[-1], "M.E. Method")


        fit_data = np.array(fit_data).T
        if fit_data.size == 0:
            continue

        best_width = np.average(fit_data[1], weights=np.sqrt(1./fit_data[2]))

        best_index = int(np.argmin(fit_data[2]))
        power_spectrum = power_spectra[best_index]

        mem_full_dict.update({i: [power_spectrum, best_width, best_index, fit_data, scan_params]})

    for i in range(vq.shape[1]):
        if not i in mem_full_dict.keys():
            continue

        print ('Peak # {0}'.format(i+1))
        print('------------------------------------')
        print ('Estimated width      : {0} THz'.format(mem_full_dict[i][1]))

        fit_data = mem_full_dict[i][3]
        scan_params = mem_full_dict[i][4]
        best_index = mem_full_dict[i][2]

        print ('Position (best fit): {0} THz'.format(scan_params[best_index][0]))
        print ('Area (best fit): {0} eV'.format(fit_data[3][best_index]))
        print ('Coefficients num (best fit): {0}'.format(fit_data[0][best_index]))
        print ('Fitting global error (best fit): {0}'.format(fit_data[2][best_index]))
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
        ax2.set_ylabel('(Global error)^-1')
        ax2.set_title('Fitting error')
        ax2.plot(fit_data[0], np.sqrt(1./fit_data[2]))

        ax3.set_xlabel('Frequency [THz]')
        ax3.set_title('Best curve fitting')
        ax3.plot(test_frequency_range, mem_full_dict[i][0], label='Power spectrum')
        ax3.plot(test_frequency_range,
                 fitting_function._function(test_frequency_range, *scan_params[best_index]),
                 label='{} fit'.format(fitting_function.curve_name))

        plt.show()


#####################################
#        FFT method (NUMPY)         #
#####################################

def _numpy_power(frequency_range, data, time_step):

    pieces = _division_of_data(frequency_range[1] - frequency_range[0],
                               data.size,
                               time_step)

    ps = []
    for i_p in pieces:

        data_piece = data[i_p[0]:i_p[1]]

        data_piece = np.correlate(data_piece, data_piece, mode='same') / data_piece.size
        ps.append(np.abs(np.fft.fft(data_piece))*time_step)

    ps = np.average(ps,axis=0)

    freqs = np.fft.fftfreq(data_piece.size, time_step)
    idx = np.argsort(freqs)

    return np.interp(frequency_range, freqs[idx], ps[idx])


def get_fft_numpy_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    requested_resolution = test_frequency_range[1]-test_frequency_range[0]
    maximum_resolution = 1./(trajectory.get_time_step_average()*(vq.shape[0]+parameters.zero_padding))
    if requested_resolution < maximum_resolution:
        print('Power spectrum resolution requested unavailable, using maximum: {0:9.6f} THz'.format(maximum_resolution))
        print('If you need higher resolution increase the number of data')

    psd_vector = []
    if not(parameters.silent):
        _progress_bar(0, 'FFT')
    for i in range(vq.shape[1]):
        psd_vector.append(_numpy_power(test_frequency_range, vq[:, i],
                                       trajectory.get_time_step_average()),
                          )

        if not(parameters.silent):
            _progress_bar(float(i + 1) / vq.shape[1], 'FFT')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion

#####################################
#         FFT method (FFTW)         #
#####################################

def _fftw_power(frequency_range, data, time_step):
    import pyfftw
    from multiprocessing import cpu_count

    pieces = _division_of_data(frequency_range[1] - frequency_range[0],
                               data.size,
                               time_step)

    ps = []
    for i_p in pieces:

        data_piece = data[i_p[0]:i_p[1]]
        data_piece = np.correlate(data_piece, data_piece, mode='same') / data_piece.size
        ps.append(np.abs(pyfftw.interfaces.numpy_fft.fft(data_piece, threads=cpu_count()))*time_step)

    ps = np.average(ps,axis=0)

    freqs = np.fft.fftfreq(data_piece.size, time_step)
    idx = np.argsort(freqs)

    return np.interp(frequency_range, freqs[idx], ps[idx])


def get_fft_fftw_power_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    psd_vector = []
    if not(parameters.silent):
        _progress_bar(0, 'FFTW')
    for i in range(vq.shape[1]):
        psd_vector.append(_fftw_power(test_frequency_range, vq[:, i],
                                      trajectory.get_time_step_average()),
                          )

        if not(parameters.silent):
            _progress_bar(float(i + 1) / vq.shape[1], 'FFTW')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion

#####################################
#        FFT method (CUDA)          #
#####################################

def _cuda_power(frequency_range, data, time_step):
    from cuda_functions import cuda_acorrelate, cuda_fft

    pieces = _division_of_data(frequency_range[1] - frequency_range[0],
                               data.size,
                               time_step)

    ps = []
    for i_p in pieces:

        data_piece = data[i_p[0]:i_p[1]]

        data_piece = cuda_acorrelate(data_piece, mode='same')/data_piece.size
        ps.append(np.abs(cuda_fft(data_piece)*time_step))

    ps = np.average(ps,axis=0)

    freqs = np.fft.fftfreq(data_piece.size, time_step)
    idx = np.argsort(freqs)

    return np.interp(frequency_range, freqs[idx], ps[idx])


def get_fft_cuda_power_spectra(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    psd_vector = []
    if not(parameters.silent):
        _progress_bar(0, 'CUDA')
    for i in range(vq.shape[1]):
        psd_vector.append(_cuda_power(test_frequency_range, vq[:, i],
                                      trajectory.get_time_step_average()),
                          )

        if not(parameters.silent):
            _progress_bar(float(i + 1) / vq.shape[1], 'CUDA')

    psd_vector = np.array(psd_vector).T

    return psd_vector * unit_conversion


#######################
#  Functions summary  #
#######################

power_spectrum_functions = {
    0: [get_fourier_direct_power_spectra, 'Fourier transform'],
    1: [get_mem_power_spectra, 'Maximum entropy method'],
    2: [get_fft_numpy_spectra, 'Fast Fourier transform (Numpy)'],
    3: [get_fft_fftw_power_spectra, 'Fast Fourier transform (FFTW)'],
    4: [get_fft_cuda_power_spectra, 'Fast Fourier transform (CUDA)']
}

