#!/usr/bin/env python
import numpy as np
import os
import dynaphopy.interface.iofile as io
import dynaphopy
from dynaphopy.interface.phonopy_link import get_force_constants_from_file
import unittest


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        self.structure = io.read_from_file_structure_poscar('GaN_data/POSCAR')

        self.structure.set_primitive_matrix([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]])

        self.structure.set_force_constants(get_force_constants_from_file(file_name='GaN_data/FORCE_CONSTANTS',
                                                                         fc_supercell=[[3, 0, 0],
                                                                                       [0, 3, 0],
                                                                                       [0, 0, 3]]))

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

        norm = np.linalg.norm(self.structure.get_cell(), axis=1)
        difference = np.mod(difference, norm)
        multiples = np.divide(difference, norm)

        self.assertLess(np.max(np.abs(multiples - np.round(multiples))), 1e-4)

    def test_thermal_properties(self):
        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=False,
                                                  initial_cut=1000,
                                                  final_cut=4000,
                                                  memmap=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)
        self.calculation.select_power_spectra_algorithm(2)
        harmonic = np.array(self.calculation.get_thermal_properties())
        anharmonic = np.array(self.calculation.get_thermal_properties(
            force_constants=self.calculation.get_renormalized_force_constants()))

        print(harmonic)
        print(anharmonic)
        maximum = np.max((harmonic-anharmonic)**2/harmonic)
        print('maximum: {}'.format(maximum))
        self.assertLess(maximum, 0.1)

    def test_force_constants_self_consistency(self):
        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=False,
                                                  initial_cut=1,
                                                  final_cut=3000,
                                                  memmap=True)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.select_power_spectra_algorithm(2)
        renormalized_force_constants = self.calculation.get_renormalized_force_constants().get_array()
        harmonic_force_constants = self.calculation.dynamic.structure.get_force_constants().get_array()
        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=1.e-2), True)

    def test_q_points_data(self):

        import yaml

        trajectory = io.initialize_from_hdf5_file('test_gan.h5',
                                                  self.structure,
                                                  read_trajectory=True,
                                                  initial_cut=1,
                                                  final_cut=3000,
                                                  memmap=True)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.select_power_spectra_algorithm(2)
        self.calculation.write_atomic_displacements([0, 0, 1], 'atomic_displacements.dat')
        self.calculation.write_quasiparticles_data(filename='quasiparticles_data.yaml')
        self.calculation.write_renormalized_phonon_dispersion_bands(filename='bands_data.yaml')

        reference = np.loadtxt('GaN_data/atomic_displacements.dat')
        data = np.loadtxt('atomic_displacements.dat')
        test_range = np.arange(-5, 5, 0.1)

        for i in range(1, data.shape[1]):
                diff_square = np.square(np.interp(test_range, data[:,0], data[:,i], right=0, left=0) -
                              np.interp(test_range, reference[:,0], reference[:,i], right=0, left=0))
                rms = np.sqrt(np.average(diff_square))
                self.assertLess(rms, 0.05)

        def assertDictAlmostEqual(dict, reference, decimal=6):
            for key, value in dict.items():
                np.testing.assert_array_almost_equal(np.array(value),
                                                     np.array(reference[key]),
                                                     decimal=decimal)

        files = ['quasiparticles_data.yaml']
        for file in files:
            print ('file: {}'.format(file))
            with open(file) as stream:
                data = yaml.load(stream)

            with open('GaN_data/' + file) as stream:
                reference = yaml.load(stream)

            for dict_data, dict_reference in zip(data, reference):
                assertDictAlmostEqual(dict_data, dict_reference, decimal=1)

if __name__ == '__main__':

    unittest.main()

    os.remove('test_gan.h5')
    os.remove('atomic_displacements.dat')
    os.remove('quasiparticles_data.yaml')
    os.remove('bands_data.yaml')