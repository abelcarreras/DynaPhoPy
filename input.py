import numpy as np
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller

print("Program start")


directory ='/home/abel/VASP/Si-phonon/4x4x4B/'
structure = reading.read_from_file_structure(directory+'OUTCAR')
structure.set_force_set( file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])

structure.set_super_cell_phonon([[4, 0, 0],
                                 [0, 4, 0],
                                 [0, 0, 4]])

structure.set_super_cell_matrix([2, 2, 2])


trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_300/RUN1/OUTCAR',structure)

calculation = controller.Calculation(trajectory)

calculation.set_reduced_q_vector([0.0, 0.0, 0.0])

calculation.set_frequency_range(np.array([0.01*i + 14.0 for i in range (100)]))
#calculation.show_bolzmann_distribution()

#calculation.get_correlation_direct()
#calculation.plot_correlation_wave_vector()
calculation.plot_correlation_phonon()
#calculation.write_correlation_direct('Data Files/correlation_d.out')
#calculation.write_correlation_wave_vector('Data Files/correlation_w2.out')
calculation.write_correlation_phonon('Data Files/correlation_p.out')


exit()