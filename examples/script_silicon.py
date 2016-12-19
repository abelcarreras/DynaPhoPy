#!/usr/bin/env python

import numpy as np
import phonopy.file_IO as file_IO
import dynaphopy.interface.iofile as io
import dynaphopy.interface.iofile.trajectory_parsers as parsers
import dynaphopy

##################################  STRUCTURE FILES #######################################
# 1. Set the directory in where the FORCE_SETS and structure POSCAR are placed
# FORCE_SETS : force set file obtained from PHONOPY calculation

directory ='/home/abel/VASP/Si/Si-phonon/4x4x4B/'
structure = io.read_from_file_structure_poscar(directory + 'POSCAR')
structure.set_force_set(file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


############################### PHONOPY CELL INFORMATION ####################################
# 2. Set primitive matrix that defines the primitive cell respect the unit cell
structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])


# 3. Set super cell phonon, this matrix denotes the super cell used in PHONOPY
# to calculate the force constants
structure.set_supercell_phonon([[4, 0, 0],
                                [0, 4, 0],
                                [0, 0, 4]])


################################### TRAJECTORY FILES ##########################################
# 4. Set the location of OUTCAR/LAMMPS file containing the Molecular Dynamics trajectory

trajectory = parsers.read_vasp_trajectory('/home/abel/VASP/Si/Si-FINAL3/Si_0.5_400/No1/OUTCAR', structure, initial_cut=1000, end_cut=50000)
# or
#trajectory = parsers.read_lammps_trajectory('/home/abel/LAMMPS/Si/Si_400.lammpstrj', structure, initial_cut=10000, end_cut=12000)

quasiparticle = dynaphopy.Quasiparticle(trajectory)

############################## DEFINE CALCULATION REQUESTS #####################################
# All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at controller,py)
# The requests will be satisfied by request order
# Python scripting features may be used for more complex requests

# 5a. Set the power spectrum algorithm
# 0: Direct Fourier transform
# 1: Maximum entropy method
# 2; numpy FFT
# 3: FFTW (Needs FFTW installed)
# 4: CUDA (Needs cuda_functions)
quasiparticle.select_power_spectra_algorithm(1)  # MEM
quasiparticle.set_number_of_mem_coefficients(1000)

# 5b. Set wave vector into which is going to be projected the velocity (default: gamma point)
quasiparticle.set_reduced_q_vector([0.5, 0.0, 0.0])  # X Point

# 5c. Define range of frequency to analyze (example: 0 - 20 THz)
quasiparticle.set_frequency_limits([0, 20])

# 5c. Define power spectrum resolution (example: 0.05 THz)
quasiparticle.set_spectra_resolution(0.05)

# 5d. Request Boltzmann distribution trajectory analysis
quasiparticle.show_boltzmann_distribution()

# 5e. Request plot full power spectrum
quasiparticle.plot_power_spectrum_full()

# 5f. Request plot wave-vector-projected power spectrum
quasiparticle.plot_power_spectrum_wave_vector()

# 5g. Request plot phonon-mode-projected power spectra
quasiparticle.plot_power_spectrum_phonon()

# 5h. Request save full power spectrum into file
quasiparticle.write_power_spectrum_full('/home/abel/full_ps.out')

# 5i. Request save wave-vector-projected power spectrum into file
quasiparticle.write_power_spectrum_wave_vector('/home/abel/correlation_wave_vector.out')

# 5j. Request save phonon-mode-projected power spectra into file
quasiparticle.write_power_spectrum_phonon('/home/abel/mem_phonon.out')

# 5k. Request calculation of renormalized force constants and write into file
# - Use MD supercell comensurate points instead of lattice dynamics supercell commensurate points
quasiparticle.parameters.use_MD_cell_commensurate = True
# - Write force constants
quasiparticle.write_renormalized_constants(filename="FORCE_CONSTANTS")

# 5l. Request calculation of thermal properties
quasiparticle.display_thermal_properties()

#5m. Request the calculation of the anisotropic displacement parameters
quasiparticle.get_anisotropic_displacement_parameters()