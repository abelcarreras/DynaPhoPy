import numpy as np
import matplotlib.pyplot as plt
import Functions.reading as reading
import Functions.projection as projection
import Functions.eigenvectors as eigen
import Functions.peaksearch as peaksearch
import Functions.correlate as correlate
import phonopy.file_IO as file_IO
import Functions.phonopy_interface as pho_interface
import Functions.energy as enerfunc
import pickle


print("Program start")



############# Real thing ##############
if True:
    #Reduced Wave vector
    q_vector_norm = np.array ([0.25, 0.25, 0.25])

    #Reading unit cell structure and force sets
    directory ='/home/abel/VASP/MgO-phonon/3x3x3/'
#    directory ='/home/abel/VASP/Si-phonon/3x3x3/'
    structure = reading.read_from_file_structure(directory+'OUTCAR')
    structure.set_force_set( file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


#How large is your primitive cell respect to unit cell (values has to be less than 1)
    structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                    [0.0, 0.5, 0.0],
                                    [0.0, 0.0, 0.5]])

    structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0]])


#Supercell used for PHONOPY phonon calculations
    structure.set_super_cell_phonon([[3, 0, 0],
                                     [0, 3, 0],
                                     [0, 0, 3]])

#How large is your cell respect to unit cell (values has to be larger than 1 and integers)
    structure.set_super_cell_matrix([2, 2, 2])


#    structure.set_super_cell([4,4,4])
    print('Unit cell')
    print(structure.get_cell())
    print('primitive cell')
    print(structure.get_primitive_cell())


    #Reading force constants from vasprun.xml (Not necessary now) (Maybe Alternative)
    #force_constants = get_force_constants_from_file('/home/abel/VASP/Si-test/FORCE_CONSTANTS')
 #   force_constants = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test/vasprun.xml')[0]
 #   structure.set_force_constants(force_constants)

    #Getting eigenvectors from somewhere
    print('Getting eigenvectors')
    eigenvectors, original_frequencies = pho_interface.obtain_eigenvectors_from_phonopy(structure,q_vector_norm)
    print(eigenvectors[4,:,:])
    print(eigenvectors[5,:,:])


    #Reading trajectory from test files
    trajectory = reading.read_from_file_trajectory('/home/abel/VASP/MgO-dynamic_300/RUN2/OUTCAR',structure)
#    trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_600/RUN6/OUTCAR',structure)
#    trajectory = reading.generate_test_trajectory(structure,q_vector_norm)


    print(structure.get_number_of_cell_atoms())
    print(structure.get_number_of_primitive_atoms())
    print(structure.get_number_of_atoms())

    #Plot energy
    plt.suptitle('Energy')
 #   print('time',trajectory.get_time().shape[0])
 #   print('trajectory',trajectory.get_energy().shape[0])
    plt.plot(trajectory.get_time().real,trajectory.get_energy().real)
    plt.show()

    #Analysis of velocity distribution
    enerfunc.bolzmann_distribution(trajectory.velocity,structure)


    #Frequency range
#    test_frequencies_range = np.array([0.005*i + 14.2 for i in range (400)])
#    test_frequencies_range = np.array([0.01*i + 3.0 for i in range (200)])
    test_frequencies_range = np.array([0.05*i + 0.1 for i in range (450)])

########################################


############# Test things #############
if False :
    q_vector = np.array ([1.5,0.5])
    trajectory = reading.read_from_file_test()
    eigenvectors, original_frequencies = eigen.get_eigenvectors_test(trajectory.structure)
    test_frequencies_range = np.array([0.01*i + 0.01 for i in range (200)])
#######################################

#Transform reduced wave vector to wave vector
#q_vector = np.prod([q_vector_norm,2*np.pi/structure.get_primitive_cell().diagonal()],axis=0)
q_vector = np.dot(q_vector_norm , (2*np.pi*np.linalg.inv(structure.get_primitive_cell())).T)
print'q_vector:',q_vector_norm,q_vector

#Show trajectory plot
plt.suptitle('Trajectory')
plt.plot(trajectory.get_time().real,trajectory.get_trajectory()[:,1].real)
plt.show()

#Show calculated harmonic frequencies
print('Original frequencies')
print(original_frequencies)


#Projection onto unit cell
vc = projection.project_onto_unit_cell(trajectory,q_vector)
plt.suptitle('Projection onto wave vector')
plt.plot(trajectory.get_time().real,vc[:,1,:].real)
plt.show()


#Projection onto phonon coordinates
vq = projection.project_onto_phonon(vc,eigenvectors)
plt.suptitle('Projection onto phonon')
plt.plot(trajectory.get_time().real,vq[:,0:5].real)
plt.show()


#Noise generation (for test only)
#for i in range(vq.shape[0]):
#    vq[i,:] += random.uniform(-0.5,0.5)


# Calculation of correlation
print ('Correlation')
correlation_vector =  correlate.get_correlation_spectra_par(vq,trajectory,test_frequencies_range)
reading.write_correlation_to_file(test_frequencies_range,correlation_vector,'Data Files/correlation.out')


#Search for frequencies in correlation spectra
frequencies = peaksearch.get_frequencies_from_correlation(correlation_vector,test_frequencies_range)
print 'Frequencies from peaks:',frequencies


#Reconstruct Dynamical Matrix from obtained new frequencies
print('Final Dynamical Matrix')
new_frequencies, new_eigenvectors, dynamical_matrix = eigen.build_dynamical_matrix(trajectory.structure,frequencies,eigenvectors)
print(dynamical_matrix.real)

print('')
print('EigenVectors & EigenValues')
print(new_eigenvectors)
print(new_frequencies)



exit()

