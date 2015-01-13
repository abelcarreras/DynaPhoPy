from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def get_frequencies_from_correlation(correlation_vector,test_frequencies_range):

    frequencies = []
    for branch in range(correlation_vector.shape[1]):
        peakind = signal.find_peaks_cwt(correlation_vector[:,branch].real, np.arange(1,200) )

     #   plt.plot(test_frequencies_range,correlation_vector[:,branch].real)
     #   plt.plot([test_frequencies_range[i] for i in peakind],[correlation_vector[i,branch].real for i in peakind],'ro')
     #   plt.show()

        heights = [correlation_vector[i,branch] for i in peakind]
        max_height_index = heights.index(max(heights))
        frequencies.append(test_frequencies_range[peakind[max_height_index]])

    return np.array(frequencies)


