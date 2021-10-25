#!/usr/bin/env python

import numpy as np
import phonopy.file_IO as file_IO
import dynaphopy.interface.iofile as io
import dynaphopy.interface.iofile.trajectory_parsers as parsers
import dynaphopy

from dynaphopy.interface.phonopy_link import get_force_sets_from_file, get_force_constants_from_file

##################################  STRUCTURE FILES #######################################
# 1. Set the directory in where the FORCE_SETS and structure POSCAR are placed

directory ='/home/user/VASP/Si/2x2x2/'
structure = io.read_from_file_structure_poscar(directory + 'POSCAR')


############################### PHONOPY CELL INFORMATION ####################################
# 2. Set primitive matrix that defines the primitive cell respect the unit cell
structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                [0.5, 0.0, 0.5],
                                [0.5, 0.5, 0.0]])


# 3. Set the hamonic phonon data (input for phonopy)
# fs_supercell: supercell matrix used in PHONOPY to obtain the force_sets
# FORCE_SETS : force set file obtained from PHONOPY calculation that contains the forces
structure.set_force_set(get_force_sets_from_file(file_name=directory + 'FORCE_SETS',
                                                 fs_supercell=[[2, 0, 0],
                                                               [0, 2, 0],
                                                               [0, 0, 2]]))

# Alternatively get_force_constants_from_file function can be used to obtain the harmonic information.
# Check unittest files (unittest folder)

############################### READING TRAJECTORY FILES #####################################
# 4. Set the location of OUTCAR/LAMMPS file containing the Molecular Dynamics trajectory

# trajectory = parsers.read_vasp_trajectory('/home/user/VASP/Si/2x2x2//OUTCAR', structure, initial_cut=10000, end_cut=60000)
# or
trajectory = parsers.read_VASP_XDATCAR('/home/user/VASP/Si/2x2x2/XDATCAR', structure, initial_cut=10000, end_cut=40000, time_step=0.0005)
# or
#trajectory = parsers.read_lammps_trajectory('/home/user/LAMMPS/Si/Si_400.lammpstrj', structure, initial_cut=10000, end_cut=12000, time_step=0.001)

quasiparticle = dynaphopy.Quasiparticle(trajectory)


############################# DEFINE CALCULATION PARAMETERS ##################################
# 5. All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at dynaphopt/__init__.py)

# 5a. Set the power spectrum algorithm
# 0: Direct Fourier transform (Not recommended)
# 1: Maximum entropy method (MEM)
# 2; numpy FFT
# 3: FFTW (Needs FFTW installed in the system)
# 4: CUDA (Needs cuda_functions installed in the system)
quasiparticle.select_power_spectra_algorithm(1)  # MEM
quasiparticle.set_number_of_mem_coefficients(1000)  # Only is used if MEM is selected

# 5b. Set wave vector into which is going to be projected the velocity (default: gamma point)
quasiparticle.set_reduced_q_vector([0.5, 0.0, 0.0])  # X Point

# 5c. Define range of frequency to analyze (example: 0 - 20 THz)
quasiparticle.set_frequency_limits([0, 20])

# 5c. Define power spectrum resolution (example: 0.05 THz)
quasiparticle.set_spectra_resolution(0.05)

# 5d. Define phonon dispersion relations path
quasiparticle.set_band_ranges([[[0.0,  0.0,   0.0],  [0.5,   0.0,  0.5]],
                              [[0.5,   0.0,   0.5],  [0.625, 0.25, 0.625]],
                              [[0.375, 0.375, 0.75], [0.0,   0.0,  0.0]],
                              [[0.0,   0.0,   0.0],  [0.5,   0.5,  0.5]]])


############################## DEFINE CALCULATION REQUESTS #####################################
# 6. All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at dynaphopt/__init__.py)

# 6a. Request Boltzmann distribution trajectory analysis
quasiparticle.show_boltzmann_distribution()

# 6b. Request plot full power spectrum
quasiparticle.plot_power_spectrum_full()

# 6c. Request plot wave-vector-projected power spectrum
quasiparticle.plot_power_spectrum_wave_vector()

# 6d. Request plot phonon-mode-projected power spectra
quasiparticle.plot_power_spectrum_phonon()

# 6e. Request save full power spectrum into file
quasiparticle.write_power_spectrum_full('/home/user/full_ps.out')

# 6f. Request save wave-vector-projected power spectrum into file
quasiparticle.write_power_spectrum_wave_vector('/home/user/correlation_wave_vector.out')

# 6g. Request save phonon-mode-projected power spectra into file
quasiparticle.write_power_spectrum_phonon('/home/user/mem_phonon.out')

# 6h. Request peak analysis
quasiparticle.phonon_individual_analysis()

# 6i. Request calculation of renormalized force constants and write them into a file
quasiparticle.write_renormalized_constants(filename="FORCE_CONSTANTS")

# 6j. Request calculation of thermal properties
quasiparticle.display_thermal_properties()

# 6k. Request to display the renormalized phonon dispersion relations
quasiparticle.plot_renormalized_phonon_dispersion_bands()
quasiparticle.plot_renormalized_phonon_dispersion_bands()

# 6l. Request the calculation of the anisotropic displacement parameters
quasiparticle.get_anisotropic_displacement_parameters()