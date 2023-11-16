#
#  This is an example of how to call dynaphopy from python API using
#  python objects and obtaining the LAMMPS trajectory using python API
#
from dynaphopy import Quasiparticle
from dynaphopy.atoms import Structure
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from dynaphopy.interface.phonopy_link import ForceConstants
from dynaphopy.interface.lammps_link import generate_lammps_trajectory
import matplotlib.pyplot as pl


# reading force constants in phonopy format
force_constants_phonopy = parse_FORCE_CONSTANTS(filename='FORCE_CONSTANTS')  # FORCE_CONSTANTS file in phonopy format
force_constants = ForceConstants(force_constants=force_constants_phonopy,
                                 supercell=[[2, 0, 0],   # force constants supercell
                                            [0, 2, 0],
                                            [0, 0, 2]])

# prepare unit cell structure
structure = Structure(scaled_positions=[[0.7500000000000000, 0.7500000000000000, 0.7500000000000000],
                                        [0.5000000000000000,  0.0000000000000000,  0.0000000000000000],
                                        [0.7500000000000000,  0.2500000000000000,  0.2500000000000000],
                                        [0.5000000000000000,  0.5000000000000000,  0.5000000000000000],
                                        [0.2500000000000000,  0.7500000000000000,  0.2500000000000000],
                                        [0.0000000000000000,  0.0000000000000000,  0.5000000000000000],
                                        [0.2500000000000000,  0.2500000000000000,  0.7500000000000000],
                                        [0.0000000000000000,  0.5000000000000000,  0.0000000000000000]],
                      atomic_elements=['Si'] * 8,
                      cell=[[5.45, 0.00, 0.00],
                            [0.00, 5.45, 0.00],
                            [0.00, 0.00, 5.45]],
                      primitive_matrix=[[0.0, 0.5, 0.5],
                                        [0.5, 0.0, 0.5],
                                        [0.5, 0.5, 0.0]],
                      force_constants=force_constants)

# generate trajectory using LAMMPS API
trajectory = generate_lammps_trajectory(structure,
                                        'in.lammps',
                                        total_time=20,
                                        time_step=0.001,
                                        relaxation_time=1,
                                        supercell=[2, 2, 2],
                                        velocity_only=True,
                                        temperature=100)

# set main object
quasiparticle = Quasiparticle(trajectory)

# set parameters for calculation
quasiparticle.select_power_spectra_algorithm(1)  # MEM
quasiparticle.set_number_of_mem_coefficients(1000)  # Only is used if MEM is selected
quasiparticle.set_frequency_limits([0, 20])
quasiparticle.set_spectra_resolution(0.05)

frequencies = quasiparticle.get_frequency_range()

# compute q-point projected power spectrum at GAMMA
quasiparticle.set_reduced_q_vector([0.0, 0.0, 0.0])  # G Point
ps = quasiparticle.get_power_spectrum_wave_vector()
pl.plot(frequencies, ps, label='Gamma')

# compute q-point projected power spectrum at L
quasiparticle.set_reduced_q_vector([0.5, 0.5, 0.5])  # L Point
frequencies = quasiparticle.get_frequency_range()
ps = quasiparticle.get_power_spectrum_wave_vector()
pl.plot(frequencies, ps, label='L')

# plot data
pl.legend()
pl.show()
