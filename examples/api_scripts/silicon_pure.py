#
#  This is an example of how to call dynaphopy from python API using
#  python objects (trajectory is still read from file)
#
from dynaphopy import Quasiparticle
from dynaphopy.atoms import Structure
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from dynaphopy.interface.phonopy_link import ForceConstants
import dynaphopy.interface.iofile.trajectory_parsers as parsers

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

# get trajectory from XDATCAR
trajectory = parsers.read_VASP_XDATCAR('XDATCAR',   # trajectory filename
                                       structure,
                                       initial_cut=10000,
                                       end_cut=40000,
                                       time_step=0.0005)
# set main object
quasiparticle = Quasiparticle(trajectory)

# set parameters for calculation
quasiparticle.select_power_spectra_algorithm(1)  # MEM
quasiparticle.set_number_of_mem_coefficients(1000)  # Only is used if MEM is selected

quasiparticle.set_reduced_q_vector([0.5, 0.0, 0.0])  # X Point
quasiparticle.set_frequency_limits([0, 20])
quasiparticle.set_spectra_resolution(0.05)

# get the renormalized force constants
renormalized_fc = quasiparticle.get_renormalized_force_constants()

# store force constants in phonopy format
renormalized_fc_phonopy = renormalized_fc.get_array()
write_FORCE_CONSTANTS(renormalized_fc_phonopy, 'RENORMALIZED_FORCE_CONSTANTS')
