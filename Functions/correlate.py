import numpy as np
import correlation
import math
#import multiprocessing
import matplotlib.pyplot as plt

from multiprocessing import Process
from multiprocessing import Queue


#Correlation worker definition
def worker2(nums,a, out_q):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """
        outdict = {}
        for n in nums:
            outdict[n] = math.sin(n)+a
            print(n)
        out_q.put(outdict)

def worker(n_proc,test_frequencies_range, vq, trajectory, correlation_function_step, out_queue):

    print('entering the process',n_proc)

    correlation_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)
    test_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)
    test_vector = []
    for i in range (vq.shape[1]):

        for k in range (test_frequencies_range.shape[0]):
            Frequency = test_frequencies_range[k]
        #            correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time(),correlation_function_step)
 #           correlation_vector[k,i] = correlation.correlation2(Frequency,vq,trajectory.get_time_step_average(),correlation_function_step)
#        test_vector[k,i] = i + k
    print('fi')
    out_queue.put([[3,3,4],1,2,n_proc])



def get_correlation_spectrum(vq,trajectory,test_frequencies_range):

#   Parameters to be taken in account
    correlation_function_step = 1

    correlation_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)

    #print('Average: ',trajectory.get_time_step_average())

    #out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
#    pool = multiprocessing.Pool()



    for i in range (vq.shape[1]):

        print 'Frequency:',i
        for k in range (test_frequencies_range.shape[0]):
            Frequency = test_frequencies_range[k]
#            correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time(),correlation_function_step)
            correlation_vector[k,i] = correlation.correlation2(Frequency,vq[:,i],trajectory.get_time_step_average(),correlation_function_step)
            print Frequency,correlation_vector[k,i].real

        print('\n')
        #print(Time)

#        plt.plot(test_frequencies_range,correlation_vector[:,i].real)
#        plt.show()

#        pool.close()
#        pool.join()
 #   plt.show()


    plt.plot(test_frequencies_range,correlation_vector.sum(axis=1).real)
    plt.show()

    return correlation_vector


def get_correlation_spectrum_par(vq,trajectory,test_frequencies_range):

    #def worker(test_frequencies_range, vq, trajectory, correlation_function_step, out_queue):

    correlation_function_step = 1

    nprocs = 5
    procs = []
    out_queue = Queue()

    phonon_numbers = range(vq.shape[1])
    chunksize = int(math.ceil(vq.shape[1] / float(nprocs)))

    for i in range(nprocs):
        p = Process(
            target=worker,
            args=(i,test_frequencies_range,
                  vq[:,chunksize * i:chunksize * (i + 1)],
                  trajectory,
                  correlation_function_step,
                  out_queue))

        procs.append(p)
        p.start()


    print('final')
    for p in procs:
        print(out_queue.get())

    print(p.name)
#    print('test')
    for p in procs:
        p.join()



    exit()