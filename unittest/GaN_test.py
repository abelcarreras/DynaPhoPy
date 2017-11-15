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
                                                  final_cut=3000,
                                                  memmap=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

        self.calculation.select_power_spectra_algorithm(1)
        harmonic = np.array(self.calculation.get_thermal_properties())
        anharmonic = np.array(self.calculation.get_thermal_properties(
            force_constants=self.calculation.get_renormalized_force_constants()))

        print(harmonic)
        print(anharmonic)
        maximum = np.max((harmonic-anharmonic)**2/harmonic)
        print('maximum: {}'.format(maximum))
        self.assertLess(maximum, 0.025)

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

        np.testing.assert_almost_equal(np.loadtxt('GaN_data/atomic_displacements.dat'),
                                       np.loadtxt('atomic_displacements.dat'), decimal=8)

        files = ['quasiparticles_data.yaml']
        for file in files:
            print ('file: {}'.format(file))
            with open(file) as stream:
                data = yaml.load(stream)

            with open('GaN_data/' + file) as stream:
                reference = yaml.load(stream)

            self.assertDictEqual(data, reference)
            self.assertDictContainsSubset(data, reference)

            files = ['bands_data.yaml']
            for file in files:
                print ('file: {}'.format(file))
                with open(file) as stream:
                    data = yaml.load(stream)

                with open('GaN_data/' + file) as stream:
                    reference = yaml.load(stream)

                for i, dict_data in enumerate(data):
                    self.assertDictEqual(dict_data, reference[i])
                    self.assertDictContainsSubset(dict_data, reference[i])

                #def qha_shift check

    def __del__(self):
        os.remove('test_gan.h5')
        os.remove('atomic_displacements.dat')
        os.remove('quasiparticles_data.yaml')
        os.remove('bands_data.yaml')

        print('end')

if __name__ == '__main__':
    unittest.main()
