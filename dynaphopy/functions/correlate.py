import numpy as np
import sys
import multiprocessing
import matplotlib.pyplot as plt
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
    for k in range (test_frequencies_range.shape[0]):
        angular_frequency = test_frequencies_range[k] * 2 * np.pi # Frequency(THz) -> angular frequency (rad/ps)
        # integration_method:        0 Trapezoid method (slow)     1 Rectangle method (fast)
        #correlation_range.append(correlation.correlation(angular_frequency,vq,trajectory.get_time(),step=correlation_function_step,integration_method=1))
        correlation_range.append(correlation.correlation2(angular_frequency,vq,trajectory.get_time_step_average(),step=correlation_function_step,integration_method=1))
#    print('finishing',n_pos)
    return {n_pos:correlation_range}


def get_correlation_spectra_par(vq,trajectory,parameters):
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


#################### functions Below testing only (Not maintained)###############


def get_correlation_spectrum(vq,test_frequencies_range):

    correlation_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)

    #print('Average: ',trajectory.get_time_step_average())

    #out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
#    pool = multiprocessing.Pool()


    print(vq.shape[1])
    for i in range (vq.shape[1]):

        print 'Frequency:',i
        for k in range (test_frequencies_range.shape[0]):
            angular_frequency = test_frequencies_range[k] * 2 * np.pi # Frequency(THz) -> angular frequency (rad/ps)
#            correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time(),correlation_function_step)
 #           correlation_vector[k,i] = correlation.correlation2(Frequency,vq[:,i],trajectory.get_time_step_average(),correlation_function_step)
            print angular_frequency,correlation_vector[k,i].real

        print('\n')
        #print(Time)

        plt.plot(test_frequencies_range,correlation_vector[:,i].real)
        plt.show()

#        pool.close()
#        pool.join()
 #   plt.show()


    plt.plot(test_frequencies_range,correlation_vector.sum(axis=1).real)
    plt.show()

    return correlation_vector
