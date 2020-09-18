import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

kb_boltzmann = 0.831446  # u * A^2 / ( ps^2 * K )


def boltzmann_distribution(trajectory, parameters):

    print("\nMaxwell-Boltzmann distribution analysis")
    print("----------------------------------------------")

    velocity = np.reshape(np.linalg.norm(trajectory.get_velocity_mass_average(), axis=2), -1)

    average = np.average(velocity)
    deviation = np.std(velocity)
    print('Average: {0:3.7e} Angstrom/ps'.format(average))
    print('Deviation: {0:3.7e} Angstrom/ps'.format(deviation))
    maxwell = stats.maxwell

    params = maxwell.fit(velocity, floc=0, scale=np.average([average, deviation]))
    print('Distribution parameter: {0:3.7e} Angstrom/ps'.format(params[1]))

    temperature = pow(params[1], 2)/ kb_boltzmann
    print('Temperature fit: {0:7.6f} K'.format(temperature))

    if not parameters.silent:
        x = np.linspace(0, float(average + 3 * deviation), 100)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('Velocity [Angstrom/ps]')
        ax.plot(x, maxwell.pdf(x, *params), lw=3)
        ax.text(0.95, 0.90, 'Temperature: {0:7.1f} K'.format(temperature),
                verticalalignment='bottom',
                horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=15)

        fig.suptitle('Velocity distribution')
        ax.hist(velocity, bins=parameters.number_of_bins_histogram, density=True)

        plt.show()



    return temperature