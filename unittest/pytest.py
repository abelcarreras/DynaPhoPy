#!/usr/bin/env python

import numpy as np
import dynaphopy.interface.iofile as io
import dynaphopy
from phonopy.file_IO import parse_FORCE_SETS

import unittest


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        structure = io.read_from_file_structure_poscar('data/POSCAR')
        structure.set_force_set(parse_FORCE_SETS(filename='data/FORCE_SETS'))
        structure.set_primitive_matrix([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])
        structure.set_supercell_phonon([[2, 0, 0],
                                        [0, 2, 0],
                                        [0, 0, 2]])

        trajectory = io.generate_test_trajectory(structure, supercell=[2, 2, 2], total_time=1, silent=True)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

    def test_force_constants_fft(self):
        self.calculation.select_power_spectra_algorithm(2)
        force_constants = self.calculation.get_renormalized_force_constants()
        force_constants2 = dynaphopy.pho_interface.get_force_constants_from_file(file_name='data/FORCE_CONSTANTS_FFT')
        self.assertEqual(np.allclose(force_constants, force_constants2, rtol=1.e-3, atol=1.e-5), True)

    def test_force_constants_mem(self):
        self.calculation.set_number_of_mem_coefficients(100)
        self.calculation.select_power_spectra_algorithm(1)
        force_constants = self.calculation.get_renormalized_force_constants()
        force_constants2 = dynaphopy.pho_interface.get_force_constants_from_file(file_name='data/FORCE_CONSTANTS_MEM')
        self.assertEqual(np.allclose(force_constants, force_constants2, rtol=1.e-1, atol=1.e-2), True)


if __name__ == '__main__':
    unittest.main()