import numpy as np
import correlation
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Process

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
            correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time(),correlation_function_step)
#            correlation_vector[k,i] = correlation.correlation2(Frequency,vq[:,i],trajectory.get_time_step_average(),correlation_function_step)
            print Frequency,correlation_vector[k,i].real

        print('\n')
        #print(Time)

        plt.plot(test_frequencies_range,correlation_vector[:,i].real)
#        plt.show()

#        pool.close()
#        pool.join()
    plt.show()


    plt.plot(test_frequencies_range,correlation_vector.sum(axis=1).real)
    plt.show()
    return correlation_vector
