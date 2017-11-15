#!/usr/bin/env python

import unittest
import numpy as np
import dynaphopy.interface.iofile as io
import dynaphopy
from dynaphopy.interface.phonopy_link import get_force_constants_from_file


class TestDynaphopy(unittest.TestCase):

    def setUp(self):
        # structure = io.read_from_file_structure_poscar('Si_data/POSCAR')
        structure = io.read_from_file_structure_outcar('Si_data/OUTCAR')

        structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                        [0.5, 0.0, 0.5],
                                        [0.5, 0.5, 0.0]])

        structure.set_force_constants(get_force_constants_from_file(file_name='Si_data/FORCE_CONSTANTS',
                                                                    fc_supercell=[[2, 0, 0],
                                                                                  [0, 2, 0],
                                                                                  [0, 0, 2]]))

        trajectory = io.generate_test_trajectory(structure, supercell=[2, 2, 2], total_time=5, silent=False)
        self.calculation = dynaphopy.Quasiparticle(trajectory)

    def etest_force_constants_self_consistency(self):
        self.calculation.select_power_spectra_algorithm(2)
        renormalized_force_constants = self.calculation.get_renormalized_force_constants().get_array()
        harmonic_force_constants = self.calculation.dynamic.structure.get_force_constants().get_array()

        self.assertEqual(np.allclose(renormalized_force_constants, harmonic_force_constants, rtol=1, atol=1.e-2), True)

    def test_q_points_data(self):

        import yaml

        self.calculation.select_power_spectra_algorithm(2)
        self.calculation.write_atomic_displacements([0, 0, 1], 'atomic_displacements.dat')
        self.calculation.write_quasiparticles_data(filename='quasiparticles_data.yaml')
        self.calculation.write_renormalized_phonon_dispersion_bands(filename='bands_data.yaml')

        np.loadtxt('Si_data/atomic_displacements.dat')
        np.loadtxt('atomic_displacements.dat')

        np.testing.assert_almost_equal(np.loadtxt('Si_data/atomic_displacements.dat'),
                                       np.loadtxt('atomic_displacements.dat'), decimal=8)

        files = ['quasiparticles_data.yaml']
        for file in files:
            print ('file: {}'.format(file))
            with open(file) as stream:
                data = yaml.load(stream)

            with open('Si_data/' + file) as stream:
                reference = yaml.load(stream)

            self.assertDictEqual(data, reference)
            self.assertDictContainsSubset(data, reference)

            files = ['bands_data.yaml']
            for file in files:
                print ('file: {}'.format(file))
                with open(file) as stream:
                    data = yaml.load(stream)

                with open('Si_data/' + file) as stream:
                    reference = yaml.load(stream)

                for i, dict_data in enumerate(data):
                    self.assertDictEqual(dict_data, reference[i])
                    self.assertDictContainsSubset(dict_data, reference[i])

                #def qha_shift check

if __name__ == '__main__':

    import yaml
    with open('quasiparticles_data.yaml') as stream:
        data = yaml.load(stream)

    print data

    unittest.main()
