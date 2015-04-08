import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def boltzmann_distribution(trajectory):


    print("\n***Velocity distribution analysis***")
#    unit_factor = 1.03642653457E-10 # (A/ps)^2*uma -> eV
#    Energy = 1/2 * m * V^2
#    energy = unit_factor/2 * np.reshape(np.multiply(masses, np.linalg.norm(velocity, axis=2)),-1)

    velocity_unit = 3.335640951981E-7 # 1E^-2/c (A/ps -> c)
    velocity = velocity_unit * np.reshape(np.linalg.norm(trajectory.get_velocity_mass_average(), axis=2),-1)

    average = np.average(velocity)
    deviation = np.std(velocity)
    print('Average: {0:3.7e}'.format(average))
    print('Deviation {0:3.7e} '.format(deviation))
    maxwell = stats.maxwell

    params = maxwell.fit(velocity, floc=0)
    print('Fit parameter: {0:3.7e}'.format(params[1]))
    boltzmann_constant = 8.6173324E-5   #eV/K
    mass_unit = 931.494061E6 # u(Atomic mass unit) -> eV/c^2
    temperature = pow(params[1],2)*mass_unit/boltzmann_constant
    print('Temperature Fit: {0:7.6f}'.format(temperature))

    x = np.linspace(0, average+3*deviation, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, maxwell.pdf(x,*params), lw=3)
    ax.text(0.95, 0.90, 'Temperature: {0:7.1f} K'.format(temperature),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=15)
    fig.suptitle('Velocity distribution')
    ax.hist(velocity, bins=25,normed=True)

    plt.show()

#   Read info in files
#    np.savetxt('Data Files/bolzmann.dat',np.array([maxwell.pdf(x,*params) for x in np.linspace(0, average+3*deviation, 100)]))
#    np.savetxt('Data Files/bolzmann_his.dat',energy_distribution[0])
#    np.savetxt('Data Files/bolzmann_his_x.dat',energy_distribution[1])


    print("***End of velocity analysis***\n")