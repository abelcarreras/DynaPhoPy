#!/usr/bin/env python

import numpy as np
import phonopy.file_IO as file_IO
from dynaphopy.functions.phonopy_link import get_force_sets_from_file
import dynaphopy.functions.iofile as reading
import dynaphopy.classes.controller as controller
import matplotlib.pyplot as pl
import analysis.modes as modes

##################################  STRUCTURE FILES #######################################
# 1. Set the directory in where the FORCE_SETS and structure OUTCAR are placed
# FORCE_SETS : force set file obtained from PHONOPY calculation
# OUTCAR : Single Point calculation of the unit cell structure used in PHONOPY calculation

directory ='/home/abel/VASP/Si/Si-FINAL3/PHONON/2x2x2/'
#directory = '/home/abel/VASP/MgO-phonon/4x4x4/'
#directory = '/home/abel/VASP/Bi2O3-phonon/'
#directory = '/home/abel/VASP/GaN/GaN-phonon/2x2x2/'
#directory = '/home/abel/VASP/GaN/GaN-phonon/6x6x3_GGA/'
#directory = '/home/abel/VASP/CaSioO3/PHONON/4x4x4/'

#structure = reading.read_from_file_structure_outcar(directory+'OUTCAR')
structure = reading.read_from_file_structure_poscar(directory+'POSCAR')
#print(structure.get_scaled_positions())
#print(structure.get_positions())


structure.set_force_set(get_force_sets_from_file(file_name=directory+'FORCE_SETS'))



############################### PHONOPY CELL INFORMATION ####################################
# 2. Set primitive matrix, this matrix fulfills that:
#    Primitive_cell = Unit_cell x Primitive_matrix
#    This matrix is the same needed for PHONOPY calculation

structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])

#structure.set_primitive_matrix([[0.0, 0.5, 0.5],
#                                [0.5, 0.0, 0.5],
#                                [0.5, 0.5, 0.0]])

#structure.set_primitive_matrix([[1.0, 0.0, 0.0],
#                                [0.0, 1.0, 0.0],
#                                [0.0, 0.0, 1.0]])

# 3. Set super cell phonon, this matrix denotes the super cell used in PHONOPY for creating
# the finite displacements

structure.set_super_cell_phonon([[2, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 2]])



#print(pho_interface.obtain_eigenvectors_from_phonopy(structure,[0,0,0.5])[1])


#Checking values
#print(structure.get_atom_type_index(super_cell=[4,4,2]))
#print(structure.get_atomic_numbers (super_cell=[4,4,2]))
#print(structure.get_atomic_types(super_cell=[4,4,2]))
#print(structure.get_masses(super_cell=[4,4,2]))
#print(structure.get_number_of_atom_types())

reading.write_xsf_file("test.xfs", structure)

################################### TRAJECTORY FILES ##########################################
# 4. Set the location of OUTCAR file containing the Molecular Dynamics trajectory

#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_600/RUN6/OUTCAR',structure)
#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/MgO-dynamic_1200/RUN2/OUTCAR',structure,limit_number_steps=5000)
#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/GaN/GaN-dynamic_900/RUN4/OUTCAR',structure)
#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Bi2O3-dynamic_1100/OUTCAR',structure,limit_number_steps=20000)

trajectory = reading.generate_test_trajectory(structure,[0.5, 0.0, 0.5],super_cell=[2, 1,1])
#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Bi2O3-dynamic_1100/OUTCAR',structure,limit_number_steps=20000)

#trajectory = reading.initialize_from_file('test.hdf5', structure)
#trajectory = reading.initialize_from_file('/home/abel/VASP/CaSioO3/VELOCITY2/velocity_500', structure)
#print(reading.check_file_type('/home/abel/VASP/Si/Si-dynamic_600/RUN6/OUTCAR'))
#print(reading.check_file_type('/home/abel/LAMMPS/eim/dump.lammpstrj'))

#exit()
#trajectory = reading.read_lammps_trajectory('/home/abel/LAMMPS/eim/dump.lammpstrj', structure=structure, time_step=0.001, last_steps=50000)

from dynaphopy.classes.dynamics import obtain_velocity_from_positions
#obtain_velocity_from_positions(structure.get_cell(),trajectory.trajectory,trajectory.get_time())

#exit()

calculation = controller.Calculation(trajectory, last_steps=80000)#, save_hfd5="test.hdf5")

calculation.set_reduced_q_vector([0.5, 0.0, 0.5])

#modes.plot_phonon_modes(structure, calculation.get_eigenvectors(), draw_primitive=True, super_cell=[1, 1, 1])
#calculation.plot_eigenvectors()


calculation.set_frequency_range(np.linspace(0, 25, 1000))
calculation.select_power_spectra_algorithm(4)
calculation.set_number_of_mem_coefficients(50)
#calculation.set_NAC(True)

#calculation.get_phonon_dispersion_spectra()


print(calculation.get_frequencies())
#print(calculation.get_q_vector())
#print(structure.get_primitive_cell())

#Show phonon dispersion spectra
#calculation.print_phonon_dispersion_spectrum()
#calculation.get_phonon_dispersion_spectra()
#calculation.plot_energy()
#exit()

#calculation.save_velocity('test.h5')
#calculation.dynamic.velocity = None
#calculation.read_velocity('test.h5')

#calculation.save_vq("vq.out")
#calculation.save_vc("vc.out")

#exit()

#print(structure.get_cell())
#structure.__dict__['_'+'cell'] = [2]
#print(structure.__dict__['_'+'cell'])



#################################### GET PROPERTIES #########################################
#calculation.plot_trajectory()
#calculation.plot_energy()
calculation.plot_trajectory(atoms=[0], coordinates=[2])
calculation.plot_velocity(atoms=[0], coordinates=[2])

#print(structure.get_cell())
#exit()

#calculation.write_trajectory_distribution([0, 0, 1], 'distribution.out')

#calculation.plot_trajectory_distribution([0, 1, 1])
#calculation.plot_trajectory_distribution([0, 1, 0])
#calculation.plot_trajectory_distribution([0, 0, 1])

#exit()

calculation.plot_vc(atoms=[0, 1])
calculation.plot_vq(modes=[0, 1, 2, 3, 4])

#print(structure.get_number_of_atoms())

#calculation.print_phonon_dispersion_spectrum()
#calculation.get_phonon_dispersion_spectra()
#calculation.set_band_ranges([[[0.2,0.0,0.2],[0.5,0.5,0.5]], [[0.5, 0.5, 0.5], [0.2, 0.0, 0.2]]])

#spectrum = calculation.get_anharmonic_dispersion_spectra(band_resolution=15)

#pl.plot(spectrum)
#pl.show()
#calculation.phonon_width_individual_analysis()
#exit()

############################## DEFINE CALCULATION REQUESTS #####################################
# All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at controller,py)
# The requests will be satisfied by request order
# Python scripting features may be used for more complex requests

# 5a. Set wave vector into which is going to be projected the velocity (default: gamma point)
#calculation.set_reduced_q_vector([0.5, 0.0, 0.5])

# 5b. Define range of frequency to analyze (default: 0-20THz)
#calculation.set_frequency_range(np.linspace(0,40,200))

# 5c. Request Boltzmann distribution trajectory analysis
#calculation.show_boltzmann_distribution()

# 5d. Request calculate plot of direct velocity correlation function (without projection)
#calculation.plot_correlation_direct()

# 5e. Request calculate plot of wave vector projected velocity correlation function
#calculation.plot_correlation_wave_vector()

#exit()
# 5f. Request calculate plot of phonon mode projected velocity correlation function
#calculation.plot_correlation_phonon()

# 5g. Request save direct velocity correlation function into file
#calculation.write_correlation_direct('Data Files/correlation_d.out')

# 5h. Request save wave vector projected velocity correlation function into file
#calculation.write_correlation_wave_vector('Data Files/correlation_w.out')

# 5i. Request save phonon projected velocity correlation function into file
#calculation.write_correlation_phonon('correlation_p.out')

# 5j. Peak analysis
calculation.phonon_individual_analysis()

exit()
