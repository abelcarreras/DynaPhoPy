#!/usr/bin/env python

#import numpy as np
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller
import Functions.phonopy_interface as pho_interface

##################################  STRUCTURE FILES #######################################
# 1. Set the directory in where the FORCE_SETS and structure OUTCAR are placed
# FORCE_SETS : force set file obtained from PHONOPY calculation
# OUTCAR : Single Point calculation of the unit cell structure used in PHONOPY calculation

#directory ='/home/abel/VASP/Si-phonon/4x4x4B/'
#directory = '/home/abel/VASP/MgO-phonon/3x3x3/'
directory = '/home/abel/VASP/GaN-phonon/3x3x3/'

structure = reading.read_from_file_structure(directory+'OUTCAR')
structure.set_force_set(file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


############################### PHONOPY CELL INFORMATION ####################################
# 2. Set primitive matrix, this matrix fulfills that:
#    Primitive_cell = Unit_cell x Primitive_matrix
#    This matrix is the same needed for PHONOPY calculation

structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])

structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                [0.5, 0.0, 0.5],
                                [0.5, 0.5, 0.0]])

structure.set_primitive_matrix([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

# 3. Set super cell phonon, this matrix denotes the super cell used in PHONOPY for creating
# the finite displacements

structure.set_super_cell_phonon([[3, 0, 0],
                                 [0, 3, 0],
                                 [0, 0, 3]])


#Checking values
#print(structure.get_atom_type_index(super_cell=[1,1,1]))
#print(structure.get_atomic_numbers(super_cell=[1,1,1]))
#print(structure.get_atomic_types())
#print(structure.get_masses(super_cell=[1,1,1]))
#print(structure.get_number_of_atom_types())


################################### TRAJECTORY FILES ##########################################
# 4. Set the location of OUTCAR file containing the Molecular Dynamics trajectory

#trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_300/RUN1/OUTCAR',structure)
trajectory = reading.read_from_file_trajectory('/home/abel/VASP/MgO-dynamic_300/RUN3/OUTCAR',structure)

eigenvectors, frequencies = pho_interface.obtain_eigenvectors_from_phonopy(structure,[0,0,0],NAC=True)

trajectory = reading.generate_test_trajectory(structure,[0.0, 0.0, 0.0],super_cell=[4,4,2])

calculation = controller.Calculation(trajectory)


############################## DEFINE CALCULATION REQUESTS #####################################
# All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at controller,py)
# The requests will be satisfied by request order
# Python scripting features may be used for more complex requests

# 5a. Set wave vector into which is going to be projected the velocity (default: gamma point)
calculation.set_reduced_q_vector([0.5, 0.0, 0.0])

# 5b. Define range of frequency to analyze (default: 0-20THz)
#calculation.set_frequency_range(np.array([0.01*i + 14.0 for i in range (100)]))

# 5c. Request Boltzmann distribution trajectory analysis
#calculation.show_bolzmann_distribution()

# 5d. Request calculate plot of direct velocity correlation function (without projection)
#calculation.plot_correlation_direct()

# 5e. Request calculate plot of wave vector projected velocity correlation function
calculation.plot_correlation_wave_vector()

# 5f. Request calculate plot of phonon mode projected velocity correlation function
#calculation.plot_correlation_phonon()

# 5g. Request save direct velocity correlation function into file
#calculation.write_correlation_direct('Data Files/correlation_d.out')

# 5h. Request save wave vector projected velocity correlation function into file
calculation.write_correlation_wave_vector('Data Files/correlation_w.out')

# 5i. Request save phonon projected velocity correlation function into file
calculation.write_correlation_phonon('Data Files/correlation_p.out')

exit()