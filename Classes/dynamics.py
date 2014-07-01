import atomstest
import numpy as np
import derivative
import correlation
import matplotlib.pyplot as plt


def obtain_velocity_from_positions(cell,trajectory,time):
    velocity = trajectory.copy()

    for i in range(trajectory.shape[1]):
        velocity [:,i,:] = derivative.derivative(cell,trajectory[:,i,:],time)
#        plt.plot(velocity[:,i,:])
#        plt.show()

    print('Velocity obtained from trajectory derivative')
    return velocity


def obtain_velocity_from_positions2(trajectory):
    velocity = trajectory.copy()
    for i in range(trajectory.shape[1]):
        for j in range(trajectory.shape[2]):
#            print(i,j)
    #        velocity[:,i,j] = np.gradient(position[:,i,j])
 #           print(trajectory[:,i,j])
            velocity [:,i,j] =  np.resize( np.diff(trajectory[:,i,j],n=4), velocity.shape[0])
#            print(velocity)
    print('Velocity obtained from trajectory derivative')
    return velocity


class Dynamics:

    def __init__(self,
                 structure=atomstest.Structure,
                 trajectory=None,
                 velocity=None,
                 energy = None,
                 time=None):


        self._time=time
        self._trajectory = trajectory
        self._energy = energy
        self._time_step_average = None


        if structure:
            self._structure = structure
        else:
            print('Warining: Initalization without structure')
            self._structure = None

        if velocity == None:
            print('No velocity provided')
            self._velocity = obtain_velocity_from_positions(self._structure.get_cell(),self._trajectory,self._time)
        else:
            self._velocity = velocity


    def set_trajectory(self,trajectory):
        self._trajectory = trajectory

    def get_trajectory(self):
        return self._trajectory

    def set_time(self, time):
        self._time = time

    def get_time(self):
        return self._time

    def get_energy(self):
        return  self._energy

    def get_time_step_average(self):

        if self._time_step_average :
            return self._time_step_average
        else:
            self._time_step_average = 0
            for i in range(self._time.shape[0]-1):
                self._time_step_average += self._time[i+1] - self._time[i]
            self._time_step_average /= (self._time.shape[0]-1)
            return self._time_step_average

    def set_structure(self, structure):
        self._structure = structure

    def get_velocity(self):
        return self._velocity

    def get_velocity_mass_average(self):
        self._velocity_mass_average = np.copy(self._velocity)
        for i in range(self._structure.get_number_of_atoms()):
            self._velocity_mass_average[:,i,:] = self._velocity[:,i,:] /np.sqrt(self._structure.get_masses()[i])
        return np.array(self._velocity_mass_average)

    @property
    def structure(self):
        return self._structure

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self,velocity):
        self._velocity = velocity
