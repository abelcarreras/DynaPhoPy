import atomstest
import numpy as np
import derivative
import correlation
import matplotlib.pyplot as plt


def obtain_velocity_from_positions(cell,trajectory,time):
    velocity = trajectory.copy()

    for i in range(trajectory.shape[1]):
        velocity [:,i,:] = derivative.derivative1(cell,trajectory[:,i,:],time)
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


        self.time=time
        self.trajectory = trajectory
        self.energy = energy
        self.time_step_average = None


        if structure:
            self.structure = structure
        else:
            print('Warining: Initalization without structure')
            self.structure = None

        if velocity == None:
            print('No velocity provided')
            self.velocity = obtain_velocity_from_positions(self.structure.cell,self.trajectory,self.time)
        else:
            self.velocity = velocity


    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_trajectory(self):
        return self.trajectory

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def get_energy(self):
        return  self.energy

    def get_time_step_average(self):

        if self.time_step_average :
            return self.time_step_average
        else:
            self.time_step_average = 0
            for i in range(self.time.shape[0]-1):
                self.time_step_average += self.time[i+1] - self.time[i]
            self.time_step_average /= (self.time.shape[0]-1)
            return self.time_step_average

    def set_structure(self, structure):
        self.structure = structure

    def get_velocity(self):
        return self.velocity

    def get_velocity_mass_average(self):
        self.velocity_mass_average = np.copy(self.velocity)
        for i in range(self.structure.number_of_atoms):
            self.velocity_mass_average[:,i,:] = self.velocity[:,i,:] /np.sqrt(self.structure.masses[i])
        return np.array(self.velocity_mass_average)

    @property
    def structure(self):
        return self.structure

    @property
    def velocity(self):
        return self.velocity

    @velocity.setter
    def velocity(self,velocity):
        self.velocity = velocity
