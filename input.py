import numpy as np
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller

print("Program start")


directory ='/home/abel/VASP/Si-phonon/2x2x2/'
structure = reading.read_from_file_structure(directory+'OUTCAR')
structure.set_force_set( file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                [0.5, 0.0, 0.5],
                                [0.5, 0.5, 0.0]])

structure.set_super_cell_phonon([[2, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 2]])

structure.set_super_cell_matrix([2, 2, 2])


trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_600/RUN6B/OUTCAR',structure)

calculation = controller.Calculation(trajectory)

calculation.set_reduced_q_vector([0.0, 0.0, 0.0])

#calculation.get_correlation_direct()
calculation.plot_correlation_wave_vector()
calculation.plot_correlation_phonon()

calculation.write_correlation_wave_vector('Data Files/correlation_w.out')
calculation.write_correlation_phonon('Data Files/correlation_p.out')

exit()