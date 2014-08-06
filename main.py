import numpy as np
import matplotlib.pyplot as plt
import Functions.reading as reading
import Functions.projection as projection
import Functions.eigenvectors as eigen
import Functions.peaksearch as peaksearch
import Functions.correlate as correlate
import phonopy.file_IO as file_IO
import Functions.phonopy_interface as pho_interface

print("Program start")



############# Real thing ##############
if True:
    #Parameters definition section (one parameter left)
#    q_vector = np.array ([1.149468,1.149468,1.149468])
#    q_vector = np.array ([1.149468/2,1.149468/2,1.149468/2])
    q_vector_norm = np.array ([0.0,0.0,0.5])


#    directory ='/home/abel/VASP/Bi2O3_phonon/'
    directory ='/home/abel/VASP/Si-test_petit/'

    #Reading structure
    structure = reading.read_from_file_structure(directory+'OUTCAR')
    structure.set_force_set( file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


    structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                    [0.0, 0.5, 0.0],
                                    [0.0, 0.0, 0.5]])

    structure.set_super_cell_phonon([[2, 0, 0],
                                     [0, 2, 0],
                                     [0, 0, 2]])

    structure.set_super_cell_matrix([2, 2, 2])


#    structure.set_super_cell([4,4,4])
    print('Unit cell')
    print(structure.get_cell())
    print('primitive cell')
    print(structure.get_primitive_cell())

#    q_vector = np.array(q_vector2 * 1.149468*2) #For cubic cell
    q_vector = np.array(q_vector_norm * 2*np.pi/structure.get_primitive_cell()[0,0]) #For cubic cell


    #Reading force constants from vasprun.xml
    #force_constants = get_force_constants_from_file('/home/abel/VASP/Si-test/FORCE_CONSTANTS')
 #   force_constants = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test/vasprun.xml')[0]
 #   structure.set_force_constants(force_constants)

    #Getting eigenvectors from somewhere
    eigenvectors, original_frequencies = pho_interface.obtain_eigenvectors_from_phonopy(structure,q_vector_norm)

    #Reading trajectory from test files
 #   trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Bi2O3_md/OUTCAR',structure)
 #   trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic_600/RUN5B/OUTCAR',structure)
    trajectory = reading.generate_test_trajectory(structure,eigenvectors,original_frequencies,q_vector)

    print(structure.get_number_of_cell_atoms())
    print(structure.get_number_of_primitive_atoms())
    print(structure.get_number_of_atoms())

    #Plot energy
    plt.suptitle('Energy')
 #   print('time',trajectory.get_time().shape[0])
 #   print('trajectory',trajectory.get_energy().shape[0])
    plt.plot(trajectory.get_time().real,trajectory.get_energy().real)
    plt.show()

    #Frequency range
#    test_frequencies_range = np.array([0.01*i + 14.5 for i in range (200)])
    test_frequencies_range = np.array([0.1*i*1 + 0.1 for i in range (200)])


########################################


############# Test things #############
if False :
    q_vector = np.array ([1.5,0.5])
    trajectory = reading.read_from_file_test()
    eigenvectors, original_frequencies = eigen.get_eigenvectors_test(trajectory.structure)
    test_frequencies_range = np.array([0.01*i + 0.01 for i in range (200)])
#######################################


plt.suptitle('Trajectory')
plt.plot(trajectory.get_time().real,trajectory.get_trajectory()[:,1].real)
plt.show()


##################Test s'ha de mirar que coincideixin #######################
#new_frequencies, new_eigenvectors, dynamical_matrix = eigen.build_dynamical_matrix(trajectory.structure,original_frequencies,eigenvectors)

#print('\n')
#print('EigenVectors & EigenValues')
#print(new_frequencies)

#exit()
#############################################################################

print('Original frequencies')
print(original_frequencies)


#Projection onto unit cell
vc = projection.project_onto_unit_cell(trajectory,q_vector)
plt.suptitle('Projection onto unit cell')
plt.plot(trajectory.get_time().real,vc[:,0,:].real)
plt.show()


#Projection onto phonon coordinates
vq = projection.project_onto_phonon(vc,eigenvectors)
plt.suptitle('Projection onto phonon')
plt.plot(trajectory.get_time().real,vq[:,0:4].real)
plt.show()

#vq=vc

#Noise generation (for test only)
#for i in range(vq.shape[0]):
#    vq[i,:] += random.uniform(-0.5,0.5)


# Correlation section (working on..)
print ('Correlation')
#test_frequencies_range = np.array([0.1*i + 0.01 for i in range (200)])
#test_frequencies_range = np.array([0.01*i + 0.01 for i in range (200)])

correlation_vector =  correlate.get_correlation_spectrum_par(vq,trajectory,test_frequencies_range)

reading.write_correlation_to_file(test_frequencies_range,correlation_vector,'Data Files/correlation.out')

#Search for frequencies
frequencies = peaksearch.get_frequencies_from_correlation(correlation_vector,test_frequencies_range)
print 'Frequencies from peaks:',frequencies


#Dynamical Matrix section
print('Final Dynamical Matrix')
new_frequencies, new_eigenvectors, dynamical_matrix = eigen.build_dynamical_matrix(trajectory.structure,frequencies,eigenvectors)
print(dynamical_matrix.real)


print('')
print('EigenVectors & EigenValues')
print(new_eigenvectors)
print(new_frequencies)

exit()

