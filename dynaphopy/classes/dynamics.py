import numpy as np
from dynaphopy.classes import atoms
from dynaphopy.derivative import derivative as derivative
from dynaphopy.analysis.coordinates import relativize_trajectory

#import matplotlib.pyplot as plt


def obtain_velocity_from_positions(cell, trajectory, time):
    velocity = np.empty_like(trajectory)
    for i in range(trajectory.shape[1]):
     #   velocity[:, i, :] = derivative(cell, trajectory[:, i, :], time)
         velocity[:, i, :] = derivative(cell, trajectory[:, i, :], time, precision_order=8)

    print('Velocity obtained from trajectory derivative')
    return velocity


class Dynamics:

    def __init__(self,
                 structure=atoms.Structure,
                 trajectory=None,
                 velocity=None,
                 energy=None,
                 time=None,
                 super_cell=None):

        self._time = time
        self._trajectory = trajectory
        self._energy = energy
        self._velocity = velocity
        self._super_cell = super_cell

        self._time_step_average = None
        self._velocity_mass_average = None
        self._relative_trajectory = None
        self._super_cell_matrix = None
        self._number_of_atoms = None

        if structure:
            self._structure = structure
        else:
            print('Warning: Initialization without structure (not recommended)')
            self._structure = None

# A bit messy, has to be fixed
    def crop_trajectory(self, last_steps):
        if last_steps is None or last_steps < 0:
            return

        if self._trajectory is not None:
            if last_steps > self._trajectory.shape[0]:
                print("Warning: specified step number larger than available")
            self._trajectory = self._trajectory[-last_steps:, :, :]

        if self._energy is not None:
            self._energy = self._energy[-last_steps:]
        if self._time is not None:
            self._time = self._time[-last_steps:]

        if last_steps > self.velocity.shape[0]:
            print("Warning: specified step number larger than available")

        self.velocity = self.velocity[-last_steps:, :, :]

        self._velocity_mass_average = None
        self._relative_trajectory = None


    def get_number_of_atoms(self):
        if self._number_of_atoms is None:
            self._number_of_atoms = self.structure.get_number_of_atoms()*np.product(self.get_super_cell_matrix())
        return self._number_of_atoms

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
            for i in range(len(self.get_time()) - 1):
                self._time_step_average += (self.get_time()[i+1] - self.get_time()[i])/(len(self.get_time()) - 1)
   #         self._time_step_average /= (self.get_time().shape[0]-1)
        return self._time_step_average

    def set_structure(self, structure):
        self._structure = structure

    def get_velocity_mass_average(self):
        if self._velocity_mass_average is None:
            self._velocity_mass_average = np.empty_like(self.velocity)
            super_cell= self.get_super_cell_matrix()
            for i in range(self.get_number_of_atoms()):
                self._velocity_mass_average[:,i,:] = self.velocity[:,i,:] * np.sqrt(self.structure.get_masses(super_cell=super_cell)[i])

        return np.array(self._velocity_mass_average)

    def get_relative_trajectory(self):
        if self._relative_trajectory is None:
                self._relative_trajectory = relativize_trajectory(self)

        return self._relative_trajectory

    def get_super_cell_matrix(self,tolerance=0.1):
        if self._super_cell_matrix is None:
            super_cell_matrix_real = np.diagonal(np.dot(self.get_super_cell(),np.linalg.inv(self.structure.get_cell())))
            self._super_cell_matrix = np.around(super_cell_matrix_real).astype("int")

            if abs(sum(self._super_cell_matrix - super_cell_matrix_real)) > tolerance:
                print('Warning! Structure cell and MD cell do not fit!')
                print('Cell size relation is not integer:',super_cell_matrix_real)
                exit()
        return self._super_cell_matrix

    #Properties
    @property
    def structure(self):
        return self._structure

    @property
    def trajectory(self):
        if self._trajectory is None:
            print('No trajectory loaded')
            exit()
        else:
            return self._trajectory

    @property
    def velocity(self):
        if self._velocity is None:
            print('No velocity provided! calculating...')
            self._velocity = obtain_velocity_from_positions(self.get_super_cell(),self.trajectory,self.get_time_step_average())
 #           self._velocity = obtain_velocity_from_positions(self.get_super_cell(),self.trajectory,self.get_time())

        return self._velocity

    @velocity.setter
    def velocity(self,velocity):
        self._velocity = velocity



