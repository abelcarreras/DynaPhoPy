import numpy as np
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller

print("Program start")


directory ='/Users/abel/VASP_TESTS/'
structure = reading.read_from_file_structure(directory+'OUTCAR_cell')
structure.set_force_set( file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])

structure.set_super_cell_phonon([[2, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 2]])

structure.set_super_cell_matrix([2, 2, 2])


trajectory = reading.read_from_file_trajectory('/Users/abel/VASP_TESTS/OUTCAR_traj',structure)

calculation = controller.Calculation(trajectory)

calculation.set_reduced_q_vector([0.25, 0.25, 0.25])

calculation.get_correlation_direct()
calculation.get_correlation_wave_vector()
calculation.get_correlation_phonon()

exit()