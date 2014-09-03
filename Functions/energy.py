import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def bolzmann_distribution(velocity,structure):

    masses = structure.get_masses(super_cell=structure.get_super_cell_matrix())

#    unit_factor = 1.03642653457E-10 # (A/ps)^2*uma -> eV
    #Energy = 1/2 * m * V^2
#    energy = unit_factor/2 * np.reshape(np.multiply(masses, np.linalg.norm(velocity, axis=2)),-1)
    unit_factor = 3.335640333E-7 # eV
    energy = unit_factor * np.reshape(np.linalg.norm(velocity, axis=2),-1)

    average = np.average(energy)
    deviation = np.std(energy)
    print('average:',average)
    print('deviation',deviation)
    energy_distribution = np.histogram(energy,bins=25,density=True,normed=True)
    print(energy_distribution)
    maxwell = stats.maxwell

    params = maxwell.fit(energy,floc=0)
    print('Fit parameter:',params)
    bolzmann_constant = 8.6173324E-5   #eV*K^-1
#    temperature = 600  #K
    mass = 28.085 * 931.494061E6 # eV/c^2

    print('Temp Fit:',pow(params[1],2)*mass/bolzmann_constant)
    np.savetxt('Data Files/bolzmann.dat',np.array([maxwell.pdf(x,*params) for x in np.linspace(0, average+3*deviation, 100)]))
    np.savetxt('Data Files/bolzmann_his.dat',energy_distribution[0])
    np.savetxt('Data Files/bolzmann_his_x.dat',energy_distribution[1])



    x = np.linspace(0, average+3*deviation, 100)
    plt.plot(x, maxwell.pdf(x,*params), lw=3)
    plt.suptitle('Energy distribution')
    plt.hist(energy,bins=25,normed=True)
    plt.show()


