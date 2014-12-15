import numpy as np
import sys
import multiprocessing
#import matplotlib.pyplot as plt
import dynaphopy.correlation as correlation


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
    text = "\rCorrelation: [{0}] {1:.2f}% {2}".format("#"*block + "-"*(bar_length-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def correlation_worker(n_pos, test_frequencies_range, vq, trajectory,correlation_function_step):
#    print('starting:',n_pos,'Time step:',trajectory.get_time_step_average(),'Frame skip:',correlation_function_step)

    correlation_range = []
    for k, frequency in enumerate(test_frequencies_range):
        angular_frequency = frequency * 2 * np.pi # Frequency(THz) -> angular frequency (rad/ps)
        # integration_method:        0 Trapezoid method (slow)     1 Rectangle method (fast)
        #correlation_range.append(correlation.correlation(angular_frequency,vq,trajectory.get_time(),step=correlation_function_step,integration_method=1))

        correlation_range.append(correlation.correlation2(angular_frequency,
                                                          vq,
                                                          trajectory.get_time_step_average(),
                                                          step=correlation_function_step,
                                                          integration_method=1))
#    print('finishing',n_pos)
    return {n_pos:correlation_range}


def get_correlation_spectra_par_python(vq, trajectory, parameters):
    test_frequencies_range = parameters.frequency_range
    correlation_function_step = parameters.correlation_function_step

    correlation_full_dict = {}
    progress_bar(0)
    def log_result(result):
        correlation_full_dict.update(result)
        progress_bar(float(len(correlation_full_dict))/vq.shape[1])

#    print ('found',multiprocessing.cpu_count(), 'CPU')
    pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count()-1,1))
#    pool = multiprocessing.Pool(processes=1)

#    print('using:', pool._processes)
    for i in range(vq.shape[1]):
        pool.apply_async(correlation_worker,
                         args = (i, test_frequencies_range,
                            vq[:,i],
                            trajectory,
                            correlation_function_step),
                         callback = log_result)


    pool.close()
    pool.join()


    correlation_vector = np.array([correlation_full_dict[i] for i in correlation_full_dict.keys()]).T

    return correlation_vector



def get_correlation_spectra_serial(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    correlation_vector = np.zeros((len(test_frequency_range),vq.shape[1]),dtype=float)
    progress_bar(0)
    for i in range (vq.shape[1]):

        for k, frequency in enumerate(test_frequency_range):
            angular_frequency = frequency * 2 * np.pi # Frequency(THz) -> angular frequency (rad/ps)
            correlation_vector[k,i] = correlation.correlation2(angular_frequency,
                                                               vq[:, i],
                                                               trajectory.get_time_step_average(),
                                                               step=parameters.correlation_function_step,
                                                               integration_method=parameters.integration_method)
        progress_bar(float(i+1)/vq.shape[1])

    return correlation_vector



def get_correlation_spectra_par_openmp(vq, trajectory, parameters):
    test_frequency_range = np.array(parameters.frequency_range)

    correlation_vector = []
    progress_bar(0)
    for i in range (vq.shape[1]):
        correlation_vector.append(correlation.correlation2_par(test_frequency_range,
                                                               vq[:, i],
                                                               trajectory.get_time_step_average(),
                                                               step=parameters.correlation_function_step,
                                                               integration_method=parameters.integration_method))
        progress_bar(float(i+1)/vq.shape[1])

    correlation_vector = np.array(correlation_vector).T

    return correlation_vector

