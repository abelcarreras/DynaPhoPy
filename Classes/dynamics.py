import atoms
import numpy as np
import derivative
#import correlation
#import matplotlib.pyplot as plt


def obtain_velocity_from_positions(cell,trajectory,time):
    velocity = np.empty_like(trajectory)
    for i in range(trajectory.shape[1]):
        velocity [:,i,:] = derivative.derivative(cell, trajectory[:,i,:], time)
    print('Velocity obtained from trajectory derivative')
    return velocity


def obtain_velocity_from_positions2(trajectory):  #Using python (Very slow)
    print("Using pure python function for obtaining velocity (slow)")
    velocity = trajectory.copy()
    for i in range(trajectory.shape[1]):
        for j in range(trajectory.shape[2]):
            velocity [:,i,j] =  np.resize( np.diff(trajectory[:,i,j],n=4), velocity.shape[0])
    print('Velocity obtained from trajectory derivative')
    return velocity


class Dynamics:

    def __init__(self,
                 structure=atoms.Structure,
                 trajectory=None,
                 velocity=None,
                 energy = None,
                 time=None,
                 super_cell=None):

        self._time=time
        self._trajectory = trajectory
        self._energy = energy
        self._velocity = velocity
        self._super_cell = super_cell

        self._time_step_average = None
        self._velocity_mass_average = None
        self._super_cell_matrix = None

        if structure:
            self._structure = structure
        else:
            print('Warning: Initialization without structure (not recommended)')
            self._structure = None


    def set_trajectory(self,trajectory):
        self._trajectory = trajectory


    def get_trajectory(self):
        return self._trajectory


    def set_time(self, time):
        self._time = time

    def get_time(self):
        return self._time


    def set_super_cell(self, super_cell):
        self._super_cell = super_cell

    def get_super_cell(self):
        return self._super_cell


    def get_energy(self):
        return  self._energy


    def get_time_step_average(self):

        if not self._time_step_average :
            self._time_step_average = 0
            for i in range(self._time.shape[0]-1):
                self._time_step_average += self._time[i+1] - self._time[i]
            self._time_step_average /= (self._time.shape[0]-1)

        return self._time_step_average


    def set_structure(self, structure):
        self._structure = structure


    def get_velocity_mass_average(self):
        if self._velocity_mass_average is None:
            self._velocity_mass_average = np.empty_like(self.velocity)
            super_cell= self.get_super_cell_matrix()

            for i in range(self._structure.get_number_of_atoms()):
                self._velocity_mass_average[:,i,:] = self.velocity[:,i,:] * np.sqrt(self.structure.get_masses(super_cell=super_cell)[i])

        return np.array(self._velocity_mass_average)


    def get_super_cell_matrix(self,tolerance=0.1):
        if self._super_cell_matrix is None:
            super_cell_matrix_real = np.diagonal(np.dot(self.get_super_cell(),np.linalg.inv(self.structure.get_cell())))
            self._super_cell_matrix = np.around(super_cell_matrix_real).astype("int")

            if abs(sum(self._super_cell_matrix - super_cell_matrix_real)) > tolerance:
                print('Warning! Structure matrix and trajectory matrix does not fit!')
                print('Matrix expansion vector is not integer:',super_cell_matrix_real)
                exit()
        return self._super_cell_matrix


    #Properties
    @property
    def structure(self):
        return self._structure


    @property
    def velocity(self):
        if self._velocity is None:
            print('No velocity provided! calculating it!')
            self._velocity = obtain_velocity_from_positions(self.get_super_cell(),self.get_trajectory(),self.get_time())
        return self._velocity


    @velocity.setter
    def velocity(self,velocity):
        self._velocity = velocity
