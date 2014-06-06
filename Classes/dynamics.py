import atomstest
import numpy as np


class Dynamics:

    def __init__(self,
                 structure=atomstest.Structure,
                 trajectory=None,
                 velocity=None,
                 time=None):


        self.velocity = velocity
        self.time=time
        self.trajectory = trajectory
        self.time_step_average = None

        if structure:
            self.structure = structure
        else:
            print('Warining: Initalization without structure')
            self.structure = None

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_trajectory(self):
        return self.trajectory

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

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
