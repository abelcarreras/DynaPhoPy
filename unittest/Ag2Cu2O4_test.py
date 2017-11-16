#!/usr/bin/env python

import unittest
import numpy as np
import dynaphopy
import dynaphopy.interface.iofile as io
from dynaphopy.interface.phonopy_link import get_force_sets_from_file
from dynaphopy.orm.atoms import Structure


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        #structure = io.read_from_file_structure_poscar('Ag2Cu2O4_data/POSCAR')

        scaled_positions = [[0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
                            [0.5000000000000000, 0.5000000000000000, 0.0000000000000000],
                            [0.0000000000000000, 0.5000000000000000, 0.5000000000000000],
                            [0.5000000000000000, 0.0000000000000000, 0.5000000000000000],
                            [0.3516123588593323, 0.5000000000000000, 0.2900275174711524],
                            [0.6483876411406685, 0.5000000000000000, 0.7099724825288461],
                            [0.8516123588593315, 0.0000000000000000, 0.2900275174711524],
                            [0.1483876411406668, 0.0000000000000000, 0.7099724825288461]]

        unit_cell = [[6.061576847284,  0.0000000000000, -0.00897996218323],
                     [0.000000000000,  2.8140830059241,  0.00000000000000],
                     [-1.94175993509,  0.0000000000000,  5.39757575753155]]

        atomic_symbols = ['Ag', 'Ag', 'Cu', 'Cu', 'O', 'O', 'O', 'O']

        self.structure = Structure(cell=unit_cell,
                                   scaled_positions=scaled_positions,
                                   atomic_elements=atomic_symbols)

        self.structure.set_primitive_matrix([[0.5, -0.5, 0.0],
                                             [0.5,  0.5, 0.0],
                                             [0.0,  0.0, 1.0]])

        force_sets = get_force_sets_from_file(file_name='Ag2Cu2O4_data/FORCE_SETS',
                                              fs_supercell=[[2, 0, 0],
                                                            [0, 2, 0],
                                                            [0, 0, 2]])

        self.structure.set_force_set(force_sets)

    def test_force_constants_self_consistency(self):

        trajectory = io.generate_test_trajectory(self.structure,
                                                 supercell=[2, 2, 2],
                                                 total_time=3,
                                                 time_step=0.001,
                                                 temperature=300,
                                                 silent=False)

        self.calculation = dynaphopy.Quasiparticle(trajectory)
        self.calculation.select_power_spectra_algorithm(2)

        renormalized_force_constants = self.calculation.get_renormalized_force_constants().get_array()
        harmonic_force_constants = self.calculation.dynamic.structure.get_force_constants().get_array()

        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=5.e-2), True)

    def test_average(self):

        trajectory = io.generate_test_trajectory(self.structure,
                                                 supercell=[1, 2, 3],
                                                 total_time=5,
                                                 silent=False,
                                                 temperature=10)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        positions_average = self.calculation.dynamic.average_positions(to_unit_cell=True).real
        print(positions_average)
        reference = [[-1.94173041e+00,   2.81396066e+00,   5.39755913e+00],
                     [ 1.08905801e+00,   1.40691916e+00,   5.39306914e+00],
                     [ 5.09064600e+00,   1.40718064e+00,   2.68983555e+00],
                     [ 2.05985758e+00,   1.39139630e-04,   2.69432553e+00],
                     [ 1.56812170e+00,   1.40719153e+00,   1.56230533e+00],
                     [ 2.55169805e+00,   1.40716376e+00,   3.82629282e+00],
                     [ 4.59891013e+00,   1.50025412e-04,   1.55781535e+00],
                     [-4.79090372e-01,   1.22256709e-04,   3.83078280e+00]]

        np.testing.assert_array_almost_equal(positions_average, reference, decimal=0)


if __name__ == '__main__':
    unittest.main()
