#!/usr/bin/env python

import numpy as np
import os
import dynaphopy.interface.iofile as io
import dynaphopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
import unittest


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        self.structure = io.read_from_file_structure_poscar('GaN_data/POSCAR')
        self.structure.set_force_constants(parse_FORCE_CONSTANTS(filename='GaN_data/FORCE_CONSTANTS'))

        self.structure.set_primitive_matrix([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])
        self.structure.set_supercell_phonon([[3, 0, 0],
                                        [0, 3, 0],
                                        [0, 0, 3]])

        if not os.path.exists('test_gan.h5'):

            trajectory = io.generate_test_trajectory(self.structure, supercell=[3, 3, 3], total_time=8, silent=False)
            self.calculation = dynaphopy.Quasiparticle(trajectory)
            self.calculation.save_velocity_hdf5('test_gan.h5', save_trajectory=True)

    def test_adp(self):
        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=True,
                                                  initial_cut=1,
                                                  final_cut=4000,
                                                  memmap=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.get_anisotropic_displacement_parameters()

        positions_average = self.calculation.dynamic.average_positions(to_unit_cell=True).real
        positions = self.structure.get_positions()
        difference = positions - positions_average

        norm = np.linalg.norm(self.structure.get_cell(), axis=0)
        difference = np.mod(difference, norm)
        multiples = np.divide(difference, norm)

        self.assertLess(np.max(np.abs(multiples - np.round(multiples))), 1e-4)

    def test_thermal_properties(self):
        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=False,
                                                  initial_cut=1000,
                                                  final_cut=3000,
                                                  memmap=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.select_power_spectra_algorithm(1)
        harmonic = np.array(self.calculation.get_thermal_properties())
        anharmonic = np.array(self.calculation.get_thermal_properties(force_constants=self.calculation.get_renormalized_force_constants()))

        print harmonic
        print anharmonic
        maximum = np.max((harmonic-anharmonic)**2/harmonic)
        self.assertLess(maximum, 0.05)

    def test_force_constants_self_consistency(self):
        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=False,
                                                  initial_cut=1,
                                                  final_cut=3000,
                                                  memmap=True)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.select_power_spectra_algorithm(2)
        renormalized_force_constants = self.calculation.get_renormalized_force_constants()
        harmonic_force_constants = self.calculation.dynamic.structure.get_force_constants()
        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=1.e-2), True)

    def __del__(self):
        os.remove('test_gan.h5')
        print ('end')

if __name__ == '__main__':
    unittest.main()
