#!/usr/bin/env python
import unittest
import os
import numpy as np
import dynaphopy.interface.iofile as io
import dynaphopy
from dynaphopy.interface.phonopy_link import get_force_constants_from_file


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        import phonopy
        print ('Using phonopy {}'.format(phonopy.__version__))

        # structure = io.read_from_file_structure_poscar('Si_data/POSCAR')
        self.structure = io.read_from_file_structure_outcar('Si_data/OUTCAR')

        self.structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                             [0.5, 0.0, 0.5],
                                             [0.5, 0.5, 0.0]])

        self.structure.set_force_constants(get_force_constants_from_file(file_name='Si_data/FORCE_CONSTANTS',
                                                                         fc_supercell=[[2, 0, 0],
                                                                                       [0, 2, 0],
                                                                                       [0, 0, 2]]))

    def test_force_constants_self_consistency(self):
        trajectory = io.generate_test_trajectory(self.structure, supercell=[2, 2, 2], total_time=10, silent=False)
        calculation = dynaphopy.Quasiparticle(trajectory)
        calculation.select_power_spectra_algorithm(2)
        renormalized_force_constants = calculation.get_renormalized_force_constants().get_array()
        harmonic_force_constants = calculation.dynamic.structure.get_force_constants().get_array()
        calculation.get_thermal_properties()

        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=1.e-2), True)

    def _test_q_points_data(self):

        import yaml
        trajectory = io.generate_test_trajectory(self.structure, supercell=[2, 2, 2], total_time=5, silent=False)
        calculation = dynaphopy.Quasiparticle(trajectory)
        calculation.select_power_spectra_algorithm(1)
        calculation.write_atomic_displacements([0, 0, 1], 'atomic_displacements_mem.dat')
        calculation.write_quasiparticles_data(filename='quasiparticles_data_mem.yaml')
        calculation.write_renormalized_phonon_dispersion_bands(filename='bands_data_mem.yaml')

        reference = np.loadtxt('Si_data/atomic_displacements_mem.dat')
        data = np.loadtxt('atomic_displacements_mem.dat')
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

        files = ['quasiparticles_data_mem.yaml']
        for file in files:
            print('file: {}'.format(file))
            with open(file) as stream:
                data = yaml.safe_load(stream)

            with open('Si_data/' + file) as stream:
                reference = yaml.safe_load(stream)

            for dict_data, dict_reference in zip(data, reference):
                assertDictAlmostEqual(dict_data, dict_reference, decimal=1)


if __name__ == '__main__':

    unittest.main()

    os.remove('atomic_displacements_mem.dat')
    os.remove('quasiparticles_data_mem.yaml')
    os.remove('bands_data_mem.yaml')