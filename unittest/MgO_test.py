#!/usr/bin/env python

import numpy as np
import dynaphopy.interface.iofile as io
import dynaphopy
from dynaphopy.interface.phonopy_link import get_force_constants_from_file

import unittest


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        structure = io.read_from_file_structure_poscar('MgO_data/POSCAR')

        structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                        [0.5, 0.0, 0.5],
                                        [0.5, 0.5, 0.0]])

        structure.set_force_constants(get_force_constants_from_file(file_name='MgO_data/FORCE_CONSTANTS',
                                                                    fc_supercell=[[2, 0, 0],
                                                                                  [0, 2, 0],
                                                                                  [0, 0, 2]]))

        trajectory = io.generate_test_trajectory(structure, supercell=[2, 2, 2], total_time=5, silent=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

    def test_force_constants_self_consistency(self):
        self.calculation.select_power_spectra_algorithm(2)
        renormalized_force_constants = self.calculation.get_renormalized_force_constants().get_array()
        harmonic_force_constants = self.calculation.dynamic.structure.get_force_constants().get_array()
        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=1.e-2), True)


if __name__ == '__main__':
    unittest.main()
