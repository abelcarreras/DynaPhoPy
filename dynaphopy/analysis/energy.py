import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

kb_boltzmann = 0.831446 # u * A^2 / ( ps^2 * K )

def boltzmann_distribution(trajectory):


    print("\n***Velocity distribution analysis***")


    velocity = np.reshape(np.linalg.norm(trajectory.get_velocity_mass_average(), axis=2), -1)

    average = np.average(velocity)
    deviation = np.std(velocity)
    print('Average: {0:3.7e} Angstrom/ps'.format(average))
    print('Deviation {0:3.7e} Angstrom/ps'.format(deviation))
    maxwell = stats.maxwell

    params = maxwell.fit(velocity, floc=0)
    print('Distribution parameter: {0:3.7e} Amstrong/ps'.format(params[1]))


    temperature = pow(params[1],2)/ kb_boltzmann
    print('Temperature fit: {0:7.6f} K'.format(temperature))

    x = np.linspace(0, average+3*deviation, 100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Velocity [Angstrom/ps]')
    ax.plot(x, maxwell.pdf(x,*params), lw=3)
    ax.text(0.95, 0.90, 'Temperature: {0:7.1f} K'.format(temperature),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=15)
    fig.suptitle('Velocity distribution')
    ax.hist(velocity, bins=25, normed=True)

    plt.show()

#   Read info in files
#    np.savetxt('Data Files/bolzmann.dat',np.array([maxwell.pdf(x,*params) for x in np.linspace(0, average+3*deviation, 100)]))
#    np.savetxt('Data Files/bolzmann_his.dat',energy_distribution[0])
#    np.savetxt('Data Files/bolzmann_his_x.dat',energy_distribution[1])


    print("***End of velocity analysis***\n")