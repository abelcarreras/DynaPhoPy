import numpy as np
import matplotlib.pylab as pl
import warnings
from scipy import integrate

N_a = 6.022140857e23
k_b = 1.38064852e-23  # J / K
h_bar = 6.626070040e-22  # J * ps

warnings.simplefilter("ignore")


def get_dos(temp, frequency, power_spectrum, n_size, bose_einstein_statistics=False):

    conversion_factor = 1.60217662e-19 # eV -> J

    def n(temp, freq):
        return pow(np.exp(freq*h_bar/(k_b*temp))-1, -1)

    if bose_einstein_statistics:
        def energy(freq, temp):
            return h_bar*freq*(0.5+n(temp, freq))
    else:
        def energy(freq, temp):
            return k_b * temp

    dos = np.nan_to_num([2.0*conversion_factor*power_spectrum[i]/(energy(freq, temp)*n_size)
                         for i, freq in enumerate(frequency)])
    return dos


def get_total_energy(temperature, frequency, dos):

    def n(temp, freq):
        return pow(np.exp(freq*h_bar/(k_b*temp))-1, -1)

    total_energy = np.nan_to_num([dos[i] * h_bar * freq * (0.5 + n(temperature, freq))
                                 for i, freq in enumerate(frequency)])

    total_energy = integrate.simps(total_energy, frequency) * N_a / 1000  # KJ/K/mol
    return total_energy


def get_free_energy(temperature, frequency, dos):

    free_energy = np.nan_to_num([dos[i] * k_b * temperature * np.log(2 * np.sinh(h_bar * freq / (2 * k_b * temperature)))
                                 for i, freq in enumerate(frequency)])

    free_energy[0] = 0
    free_energy = integrate.simps(free_energy, frequency) * N_a / 1000  # KJ/K/mol
    return free_energy


def get_free_energy_correction_shift(temperature, frequency, dos, shift):

    def n(temp, freq):
        return pow(np.exp(freq*h_bar/(k_b*temp))-1, -1)

    free_energy_c = np.nan_to_num([dos[i] * -h_bar/2 *shift*(n(temperature, freq) + 1 / 2.)
                                   for i, freq in enumerate(frequency)])

    free_energy_c = integrate.simps(free_energy_c, frequency) * N_a / 1000 # KJ/K/mol
    return free_energy_c


def get_free_energy_correction_dos(temperature, frequency, dos, dos_r):

    def n(temp, freq):
        return pow(np.exp(freq*h_bar/(k_b*temp))-1, -1)

    free_energy_1 = np.nan_to_num([ dos_r[i] * -h_bar/2 * freq*(n(temperature, freq) + 1 / 2.)
                                      for i, freq in enumerate(frequency)])

    free_energy_2 = np.nan_to_num([ dos[i] * -h_bar/2 * freq*(n(temperature, freq) + 1 / 2.)
                                      for i, freq in enumerate(frequency)])

    free_energy_c = free_energy_1 - free_energy_2

    free_energy_c = integrate.simps(free_energy_c, frequency) * N_a / 1000 # KJ/K/mol
    return free_energy_c


def get_entropy(temperature, frequency, dos):

    def coth(x):
        return  np.cosh(x)/np.sinh(x)

    entropy = np.nan_to_num([dos[i]*(1.0 / (2. * temperature) * h_bar * freq * coth(h_bar * freq / (2 * k_b * temperature))
                                     - k_b * np.log(2 * np.sinh(h_bar * freq / (2 * k_b * temperature))))
                             for i, freq in enumerate(frequency)])
    entropy = integrate.simps(entropy, frequency) * N_a # J/K/mol
    return entropy

# Alternative way to calculate entropy (not used)
def get_entropy2(temperature, frequency, dos):

    def n(temp, freq):
        return pow(np.exp(freq*h_bar/(k_b*temp))-1, -1)

    entropy = np.nan_to_num([dos[i] * k_b * ((n(temperature, freq) + 1) * np.log(n(temperature, freq) + 1)
                                             - n(temperature, freq) * np.log(n(temperature, freq)))
                         for i, freq in enumerate(frequency)])
    entropy = integrate.simps(entropy, frequency) * N_a # J/K/mol
    return entropy


def get_cv(temperature, frequency, dos):

    def z(temp, freq):
        return h_bar*freq/(k_b*temp)

    c_v = np.nan_to_num([dos[i] * k_b * pow(z(temperature, freq), 2) * np.exp(z(temperature, freq)) / pow(np.exp(z(temperature, freq)) - 1, 2)
                         for i, freq in enumerate(frequency)])
    c_v = integrate.simps(c_v, frequency) * N_a # J/K/mol

    return c_v

if __name__ == "__main__":

    shift = 0.05

    #temp = 300
    #dos_file = open('/Users/abel/TEST_GPU/GaN/total_dos.dat', mode='r')
    #dos_r_file = open('/Users/abel/TEST_GPU/GaN/total_dos_o.dat', mode='r')
    #power_file = open('/Users/abel/TEST_GPU/GaN/power_spectrum.dat', mode='r')

    temp=900
    dos_file = open('/home/abel/LAMMPS/Si/total_dos_h.dat', mode='r')
    dos_r_file = open('/home/abel/LAMMPS/Si/total_dos_o.dat', mode='r')
    power_file = open('/home/abel/LAMMPS/Si/power_spectrum_900_12_fft_vlong.dat', mode='r')

    frequency = []
    dos = []
    for line in dos_file.readlines()[1:]:
        frequency.append(float(line.split()[0]))
        dos.append(float(line.split()[1]))

    frequency_r = []
    dos_r = []
    for line in dos_r_file.readlines()[1:]:
        frequency_r.append(float(line.split()[0]))
        dos_r.append(float(line.split()[1]))

    frequency_p = []
    power_spectrum = []
    for line in power_file.readlines():
        frequency_p.append(float(line.split()[0]))
        power_spectrum.append(float(line.split()[1]))

    # power_spectrum = get_dos(temp,frequency_p,power_spectrum, 12*12*6)

    power_spectrum = get_dos(temp, frequency_p, power_spectrum, 12*12*12)

    pl.plot(frequency_p, power_spectrum, label='power')
    pl.plot(frequency, dos,label='dos')
    pl.plot(frequency_r, dos_r, label='dos_r')
    pl.legend()
    pl.show()

    # free_energy = get_free_energy(temp,frequency,dos) + get_free_energy_correction(temp, frequency, dos, shift)

    print (get_free_energy_correction_shift(temp, frequency, dos, shift),
           get_free_energy_correction_dos(temp, frequency, dos, dos_r))

    free_energy = get_free_energy(temp, frequency_r, dos_r) + get_free_energy_correction_dos(temp, frequency, dos_r, dos)
    entropy = get_entropy(temp, frequency_r, dos_r)
    c_v = get_cv(temp, frequency_r, dos_r)
    print ('Renormalized')
    print ('-------------------------')
    print ('Free energy: {0} KJ/K/mol'.format(free_energy))
    print ('Entropy: {0} J/K/mol'.format(entropy))
    print ('Cv: {0} J/K/mol'.format(c_v))
    print (np.trapz(dos_r, x=frequency_r))/(8*3)
    print (integrate.simps(dos_r,x=frequency_r)/(8*3))

    print ('\nFrom MD')
    print ('-------------------------')
    free_energy = get_free_energy(temp, frequency_p, power_spectrum)
    entropy = get_entropy(temp, frequency_p, power_spectrum)
    c_v = get_cv(temp, frequency_p, power_spectrum)

    print ('Free energy: {0} KJ/K/mol'.format(free_energy))
    print ('Entropy: {0} J/K/mol'.format(entropy))
    print ('Cv: {0} J/K/mol'.format(c_v))
    print (np.trapz(power_spectrum, x=frequency_p))/(8*3)
    print (integrate.simps(power_spectrum, x=frequency_p))/(8*3)

    print ('\nHARMONIC')
    print ('-------------------------')
    free_energy = get_free_energy(temp, frequency, dos)
    entropy = get_entropy(temp, frequency, dos)
    c_v = get_cv(temp, frequency, dos)

    print ('Free energy: {0} KJ/K/mol'.format(free_energy))
    print ('Entropy: {0} J/K/mol'.format(entropy))
    print ('Cv: {0} J/K/mol'.format(c_v))
    print (np.trapz(dos, x=frequency)/(8*3))
    print (integrate.simps(dos, x=frequency)/(8*3))