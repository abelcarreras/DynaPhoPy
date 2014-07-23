import numpy as np
import correlation
import math
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import Queue


def correlation_worker(n_pos,test_frequencies_range, vq, trajectory):

    correlation_function_step = 10

    print('starting:',n_pos,'Time step:',trajectory.get_time_step_average(),'Frame skip:',correlation_function_step)

    correlation_range = []
    for k in range (test_frequencies_range.shape[0]):
        Frequency = test_frequencies_range[k]
        #correlation_range.append(correlation.correlation(Frequency,vq,trajectory.get_time(),correlation_function_step))
        correlation_range.append(correlation.correlation2(Frequency,vq,trajectory.get_time_step_average(),correlation_function_step))
    print('finishing',n_pos)
    correlation_range = correlation_range/np.sum(correlation_range)
    return {n_pos:correlation_range}


def get_correlation_spectrum_par(vq,trajectory,test_frequencies_range):

    correlation_full_dict = {}

    def log_result(result):
        correlation_full_dict.update(result)

    print ('found',multiprocessing.cpu_count(), 'CPU')

    pool = multiprocessing.Pool(processes=3)
    print('using:', pool._processes)
    for i in range(vq.shape[1]):
        pool.apply_async(correlation_worker,
                         args = (i,test_frequencies_range,
                            vq[:,i],
                            trajectory),
                         callback = log_result)
    pool.close()
    pool.join()

    correlation_vector = np.array([correlation_full_dict[i] for i in correlation_full_dict.keys()]).T

    for i in range(correlation_vector.shape[1]):
        plt.plot(test_frequencies_range,correlation_vector[:,i].real)
        plt.show()

    plt.plot(test_frequencies_range,correlation_vector.sum(axis=1).real,'r-')
    plt.show()

    return correlation_vector



def correlation_worker2(n_pos,test_frequencies_range, vq, trajectory, correlation_function_step, out_queue):

    print('startingr:',n_pos)

    correlation_dict = {}
    for i in range (vq.shape[1]):
        correlation_range = []
        for k in range (test_frequencies_range.shape[0]):
            Frequency = test_frequencies_range[k]
            #correlation_range.append(correlation.correlation(Frequency,vq,trajectory.get_time(),correlation_function_step))
            correlation_range.append(correlation.correlation2(Frequency,vq[:,i],trajectory.get_time_step_average(),correlation_function_step))

        correlation_dict[n_pos+i] = correlation_range
        print('finishing',n_pos+i)

    out_queue.put(correlation_dict)


def get_correlation_spectrum_par2(vq,trajectory,test_frequencies_range):

    #def worker(test_frequencies_range, vq, trajectory, correlation_function_step, out_queue):

    correlation_function_step = 1

    number_or_processes = 3
    processes = []
    out_queue = Queue()
    correlation_full_dict = {}

    phonon_numbers = range(vq.shape[1])
    chunk_size = int(math.ceil(vq.shape[1] / float(number_or_processes)))

    for i in range(number_or_processes):
        p = Process(
            target=correlation_worker2,
            args=(chunk_size * i,test_frequencies_range,
                  vq[:,chunk_size * i:chunk_size * (i + 1)],
                  trajectory,
                  correlation_function_step,
                  out_queue))

        processes.append(p)
        p.start()


    for p in processes:
        correlation_full_dict.update(out_queue.get())
        p.join()

    correlation_vector = np.array([correlation_full_dict[i] for i in correlation_full_dict.keys()]).T

    plt.plot(test_frequencies_range,correlation_vector.sum(axis=1).real)
    plt.show()

    return correlation_vector


def get_correlation_spectrum(vq,trajectory,test_frequencies_range):

#   Parameters to be taken in account
    correlation_function_step = 20

    correlation_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)

    #print('Average: ',trajectory.get_time_step_average())

    #out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
#    pool = multiprocessing.Pool()


    print(vq.shape[1])
    for i in range (vq.shape[1]):

        print 'Frequency:',i
        for k in range (test_frequencies_range.shape[0]):
            Frequency = test_frequencies_range[k]
#            correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time(),correlation_function_step)
 #           correlation_vector[k,i] = correlation.correlation2(Frequency,vq[:,i],trajectory.get_time_step_average(),correlation_function_step)
            print Frequency,correlation_vector[k,i].real

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
