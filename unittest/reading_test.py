#!/usr/bin/env python

import numpy as np
import dynaphopy.interface.iofile.trajectory_parsers as trajectory_parsers
import dynaphopy.interface.iofile as io
from dynaphopy import Quasiparticle
from dynaphopy.interface.phonopy_link import get_force_constants_from_file

import unittest

class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        self.structure = io.read_from_file_structure_poscar('Si_data/POSCAR')

        self.structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                             [0.5, 0.0, 0.5],
                                             [0.5, 0.5, 0.0]])

        self.structure.set_force_constants(get_force_constants_from_file(file_name='Si_data/FORCE_CONSTANTS',
                                                                    fc_supercell=[[2, 0, 0],
                                                                                  [0, 2, 0],
                                                                                  [0, 0, 2]]))

    def test_XDATCAR(self):
        defined_time_step = 0.0005
        trajectory = trajectory_parsers.read_VASP_XDATCAR('Si_data/XDATCAR', self.structure,
                                                          initial_cut=3, end_cut=14, time_step=defined_time_step)

        rel_traj_ref = [[ 3.83203119e-09,  1.41927075e-09,  1.70312506e-09],
                        [ 4.25781249e-10,  3.12239576e-09,  5.10937512e-09],
                        [-1.27734381e-09, -3.69010421e-09,  3.40625005e-09],
                        [-1.27734384e-09,  4.82552079e-09, -5.10937504e-09],
                        [-4.68359383e-09,  4.82552082e-09,  5.10937504e-09],
                        [ 4.25781233e-10, -2.83854171e-10, -6.81249987e-09],
                        [-2.98046880e-09, -8.79947915e-09, -3.40625001e-09],
                        [ 2.12890625e-09, -1.98697915e-09,  6.24500451e-17],
                        [ 2.12890630e-09, -2.83854242e-10,  1.00613962e-16],
                        [ 3.83203120e-09,  1.41927079e-09,  3.46944695e-17],
                        [ 2.12890631e-09,  3.12239582e-09,  3.40625004e-09],
                        [-4.68359377e-09, -3.69010417e-09, -3.40624993e-09]]
        rel_traj = np.array([np.average(trajectory.average_positions().real - traj, axis=0).real
                             for traj in trajectory.trajectory])

        # print(rel_traj)
        check_traj = np.allclose(rel_traj_ref, rel_traj, rtol=1e-6, atol=1.e-18)
        print('trajectory:', check_traj)

        check_time = trajectory.get_time_step_average() == defined_time_step
        print('time:', check_time)

        mean_matrix_ref = [[ 0.00065253,  0.00016701,  0.00033313],
                           [ 0.00016701,  0.00063413,  0.00023646],
                           [ 0.00033313,  0.00023646,  0.00072747]]

        mean_matrix = np.average(trajectory.get_mean_displacement_matrix(), axis=0)

        check_mean_matrix = np.allclose(mean_matrix, mean_matrix_ref, rtol=1e-3, atol=1.e-6)
        print('mean matrix:', check_mean_matrix)

        self.assertEqual(check_traj and check_time and check_mean_matrix, True)

    def test_lammpstraj(self):
        defined_time_step = 0.001
        trajectory = trajectory_parsers.read_lammps_trajectory('Si_data/si.lammpstrj', self.structure, initial_cut=3,
                                                               end_cut=14, time_step=defined_time_step)

        rel_traj_ref = [[-3.51562529e-12,  -1.82290120e-12,   2.99481620e-12],
                        [-3.90621563e-13,   4.42701258e-12,  -1.30178854e-13],
                        [ 2.73439431e-12,  -3.38547246e-12,   4.55729551e-12],
                        [-3.90609420e-13,  -2.60453117e-13,  -4.81767751e-12],
                        [ 4.29688403e-12,   2.86455235e-12,  -1.30132016e-13],
                        [-1.95310261e-12,  -9.63547298e-12,   2.99480753e-12],
                        [ 1.17189591e-12,   2.86454020e-12,  -3.25514442e-12],
                        [-3.90574725e-13,  -2.60357708e-13,   1.43223974e-12],
                        [ 2.73435614e-12,  -2.60437505e-13,  -4.81760118e-12],
                        [ 5.85947263e-12,  -1.82294110e-12,  -1.69266164e-12],
                        [-5.07803868e-12,   4.42703513e-12,   1.43230393e-12],
                        [-5.07806297e-12,   2.86455755e-12,   1.43231781e-12]]

        rel_traj = np.array([np.average(trajectory.average_positions().real - traj, axis=0).real
                             for traj in trajectory.trajectory])

        check_traj = np.allclose(rel_traj_ref, rel_traj, rtol=1e-6, atol=1.e-18)
        print('trajectory:', check_traj)

        check_time = float(trajectory.get_time_step_average()) == float(defined_time_step)
        print('time:', check_time)

        mean_matrix_ref = [[ 0.00263396,  0.00053198,  0.00040298],
                           [ 0.00053198,  0.00383308,  0.00034559],
                           [ 0.00040298,  0.00034559,  0.00381229]]

        mean_matrix = np.average(trajectory.get_mean_displacement_matrix(), axis=0)
        check_mean_matrix = np.allclose(mean_matrix, mean_matrix_ref, rtol=1e-3, atol=1.e-6)
        print('mean matrix:', check_mean_matrix)

        self.assertEqual(check_traj and check_time and check_mean_matrix, True)


if __name__ == '__main__':
    unittest.main()