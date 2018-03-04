import numpy as np
from dynaphopy.displacements import atomic_displacements
import os


class Dynamics:

    def __init__(self,
                 structure=None,
                 trajectory=None,
                 scaled_trajectory=None,
                 velocity=None,
                 energy=None,
                 time=None,
                 supercell=None,
                 memmap=False):

        self._time = time
        self._trajectory = trajectory
        self._scaled_trajectory = scaled_trajectory
        self._energy = energy
        self._velocity = velocity
        self._supercell = supercell
        self._memmap=memmap

        self._time_step_average = None
        self._velocity_mass_average = None
        self._relative_trajectory = None
        self._supercell_matrix = None
        self._number_of_atoms = None
        self._mean_displacement_matrix = None

        if structure is not None:
            self._structure = structure

            if trajectory is not None:
                self._trajectory = trajectory

        else:
            print('Warning: Initialization without structure')
            self._structure = None

        #Read environtment variables
        try:
            self._temp_directory = os.environ["DYNAPHOPY_TEMPDIR"]
            if os.path.isdir(self._temp_directory):
                self._temp_directory = self._temp_directory + '/'
            else:
                self._temp_directory = ''
        except KeyError:
            self._temp_directory = ''

    def __del__(self):
        #Clean all temporal files from memmap
        if self._memmap:
            for mapped_array in [self._velocity, self._trajectory, self._relative_trajectory, self._velocity_mass_average]:
                try:
                    filename = mapped_array.filename
                except AttributeError:
                    continue
                del mapped_array
                os.remove(filename)

# A bit messy, has to be fixed
    def crop_trajectory(self, last_steps):

        if last_steps is None or last_steps <= 0:
            return

        if self._trajectory is not None:
            self._trajectory = self._trajectory[-last_steps:, :, :]
            self._relative_trajectory = None
            self._scaled_trajectory = None

        if self._relative_trajectory is not None:
            self._relative_trajectory = self._relative_trajectory[-last_steps:, :, :]
            self._trajectory = None
            self._scaled_trajectory = None

        if self._scaled_trajectory is not None:
            self._scaled_trajectory = self._scaled_trajectory[-last_steps:, :, :]
            self._trajectory = None
            self._relative_trajectory = None

        if self._energy is not None:
            self._energy = self._energy[-last_steps:]
        if self._time is not None:
            self._time = self._time[-last_steps:]

        if last_steps > self.velocity.shape[0]:
            print("Warning: specified step number larger than available")

        self.velocity = self.velocity[-last_steps:, :, :]

        self._velocity_mass_average = None

        if self._memmap:
            filename = self._relative_trajectory.filename
            self._relative_trajectory = None
            try:
                os.remove(filename)
            except:
                pass
        else:
            self._relative_trajectory = None

        # print("Using {0} steps".format(self.velocity.shape[0]))

    def get_number_of_atoms(self):
        if self._number_of_atoms is None:
            self._number_of_atoms = self.structure.get_number_of_atoms()*np.product(self.get_supercell_matrix())
        return self._number_of_atoms

    def set_time(self, time):
        self._time = time

    def get_time(self):
        return self._time

    #def set_supercell(self, supercell):
    #    self._supercell = supercell

    def get_supercell(self):
        return self._supercell

    def get_energy(self):
        return self._energy

    def get_time_step_average(self):

        if not self._time_step_average :
            self._time_step_average = 0
            for i in range(len(self.get_time()) - 1):
                self._time_step_average += (self.get_time()[i+1] - self.get_time()[i])/(len(self.get_time()) - 1)
   #         self._time_step_average /= (self.get_time().shape[0]-1)
        self._time_step_average = np.round(self._time_step_average, decimals=8)

        return self._time_step_average

    def set_structure(self, structure):
        self._structure = structure

    def get_velocity_mass_average(self):
        if self._velocity_mass_average is None:
            if self._memmap:
                self._velocity_mass_average = np.memmap(self._temp_directory+'velocity_mass.{0}'.format(os.getpid()),
                                                        dtype='complex', mode='w+', shape=self.velocity.shape)
            else:
                self._velocity_mass_average = np.empty_like(self.velocity)

            supercell = self.get_supercell_matrix()
            for i in range(self.get_number_of_atoms()):
                self._velocity_mass_average[:, i, :] = (self.velocity[:, i, :] *
                                                        np.sqrt(self.structure.get_masses(supercell=supercell)[i]))

        return self._velocity_mass_average

    def get_relative_trajectory(self):
        if self._relative_trajectory is None:

            trajectory = self.trajectory

            supercell = self.get_supercell()
            number_of_atoms = self.trajectory.shape[1]
            supercell_matrix = self.get_supercell_matrix()
            position = self.structure.get_positions(supercell=supercell_matrix)


            if self._memmap:
                normalized_trajectory = np.memmap(self._temp_directory+'r_trajectory.{0}'.format(os.getpid()),
                                                  dtype='complex', mode='w+', shape=trajectory.shape)
            else:
                normalized_trajectory = self.trajectory.copy()

            for i in range(number_of_atoms):
                normalized_trajectory[:, i, :] = atomic_displacements(trajectory[:, i, :], position[i], supercell)

            self._relative_trajectory = normalized_trajectory
        return self._relative_trajectory

    def get_supercell_matrix(self, tolerance=0.01):

        def parameters2(h):
            a = np.linalg.norm(h[:,0])
            b = np.linalg.norm(h[:,1])
            c = np.linalg.norm(h[:,2])
            return [a, b, c]

        def parameters(h):
            #return [np.linalg.norm(h[i, :]) for i in range(h.shape[1])]
            return np.linalg.norm(h, axis=1)

        if self._supercell_matrix is None:
            supercell_matrix_real = np.divide(parameters(self.get_supercell()), parameters(self.structure.get_cell()))
            self._supercell_matrix = np.around(supercell_matrix_real).astype("int")

            if abs(sum(self._supercell_matrix - supercell_matrix_real)/np.linalg.norm(supercell_matrix_real)) > tolerance:
                print(abs(sum(self._supercell_matrix - supercell_matrix_real) / np.linalg.norm(supercell_matrix_real)))
                print('Warning! Defined unit cell and MD supercell do not match!')
                print('Cell size relation is not integer: {0}'.format(supercell_matrix_real))
                exit()

            print('MD cell size relation: {0}'.format(self._supercell_matrix))

        return self._supercell_matrix

    def get_mean_displacement_matrix(self, use_average_positions=False):

        if self._mean_displacement_matrix is None:

            atom_type = self.structure.get_atom_type_index()
            number_atom_primitive_equivalent = np.unique(atom_type, return_counts=True)[1]

            supercell = self.get_supercell_matrix()
            atom_type_index = self.structure.get_atom_type_index(supercell=supercell)
            number_of_atom_types = self.structure.get_number_of_atom_types()
            number_of_dimensions = self.structure.get_number_of_dimensions()
            displacements = self.get_relative_trajectory()
            number_of_data = displacements.shape[0]

            # Correct the atom positions by position average
            if use_average_positions:
                position_difference = np.average(displacements, axis=0)
            else:
                position_difference = np.zeros_like(self.structure.get_positions(supercell=supercell))

            number_of_equivalent_atoms = np.prod(supercell)

            mean_displacement_matrix = np.zeros((number_of_atom_types, number_of_dimensions, number_of_dimensions))

            for i in range(displacements.shape[1]):
                primtive_normalization = number_atom_primitive_equivalent[atom_type_index[i]]
                mean_displacement_matrix[atom_type_index[i], :, :] += np.dot(np.conj(displacements[:, i, :]).T,
                                                                             displacements[:, i, :] - position_difference[i]
                                                                             ).real / primtive_normalization

            self._mean_displacement_matrix = mean_displacement_matrix / (number_of_equivalent_atoms * number_of_data)

        return self._mean_displacement_matrix

    def average_positions(self, number_of_samples=None, to_unit_cell=False):

        supercell = self.get_supercell_matrix()
        number_of_dimensions = self.structure.get_number_of_dimensions()

        cell = self.get_supercell()
        number_of_atoms = self.trajectory.shape[1]
        positions = self.structure.get_positions(supercell=supercell)

        normalized_trajectory = self.get_relative_trajectory()

        if number_of_samples:
            length = normalized_trajectory.shape[0]
            if length < number_of_samples:
                number_of_samples = length
            normalized_trajectory = normalized_trajectory
            samples = np.random.random_integers(length, size=(number_of_samples,))-1
            normalized_trajectory = normalized_trajectory[samples, :]

        averaged_positions = np.average(normalized_trajectory, axis=0)

        # Average respect to unit cell
        if to_unit_cell:

            cell = self.structure.get_cell()
            number_of_atoms = self.structure.get_number_of_atoms()
            positions = self.structure.get_positions()

            index_type_unitcell = self.structure.get_atom_type_index()
            index_type = self.structure.get_atom_type_index(supercell=supercell)

            number_of_atom_types = self.structure.get_number_of_atom_types()
            normalization = np.prod(supercell)

            averaged_unit_cell = np.zeros((number_of_atom_types, number_of_dimensions), dtype=complex)

            for i, coordinates  in enumerate(averaged_positions):
                averaged_unit_cell[index_type[i], :] += coordinates/normalization

            averaged_positions = []
            for i in range(positions.shape[0]):
                averaged_positions.append(averaged_unit_cell[index_type_unitcell[i]])
            averaged_positions = np.array(averaged_positions)

        # Average respect to unit cell
        averaged_positions += positions

        for j in range(number_of_atoms):

            difference_matrix = np.around(np.dot(np.linalg.inv(cell.T),
                                                 averaged_positions[j, :] - 0.5 * np.dot(np.ones((number_of_dimensions)), cell)),
                                                 decimals=0)
            averaged_positions[j, :] -= np.dot(difference_matrix, cell)

        return averaged_positions

    # Properties
    @property
    def structure(self):
        return self._structure

    @property
    def trajectory(self):
        if self._trajectory is None:
            if self._scaled_trajectory is not None:
                self._trajectory = np.dot(self._scaled_trajectory, self.get_supercell())
            else:
                print('No trajectory loaded')
                exit()

        return self._trajectory

    @property
    def velocity(self):
        if self._velocity is None:
            print('No velocity provided! calculating it from coordinates...')
            if self._memmap:
                self._velocity = np.memmap(self._temp_directory+'velocity.{0}'.format(os.getpid()),
                                           dtype='complex',
                                           mode='w+',
                                           shape=self.get_relative_trajectory().shape)
            else:
                self._velocity = np.zeros_like(self.get_relative_trajectory(), dtype=complex)
            for i in range(self.get_number_of_atoms()):
                for j in range(self.structure.get_number_of_dimensions()):
                    self._velocity[:, i, j] = np.gradient(self.get_relative_trajectory()[:, i, j],
                                                          self.get_time_step_average())

        return self._velocity

    @velocity.setter
    def velocity(self,velocity):
        self._velocity = velocity