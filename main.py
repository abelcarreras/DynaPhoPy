import numpy as np
import matplotlib.pyplot as plt
import correlation
import Functions.reading as reading
import Functions.projection as projection
import Functions.eigenvectors as eigen
import Functions.peaksearch as peaksearch
import phonopy.file_IO as file_IO
import Functions.phonopy_interface as pho_interface

print("Program start")


#Parameters definition section (one parameter left)
q_vector = np.array ([1.5,0.5,0.5])



#Reading structure
structure = reading.read_from_file_structure('/home/abel/VASP/Si-test/OUTCAR')

#Reading force constants from vasprun.xml
force_constants = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test/vasprun.xml')[0]
#force_constants = get_force_constants_from_file('/home/abel/VASP/Si-test/FORCE_CONSTANTS')
structure.set_force_constants(force_constants)

#Reading trajectory from test files
#trajectory = reading.read_from_file_test()
trajectory = reading.read_from_file_trajectory('/home/abel/VASP/Si-dynamic/OUTCAR',structure)

#Testing manual input (to be deprecated)
number_of_dimensions=trajectory.structure.get_number_of_dimensions()

#Getting eigenvectors from somewhere
#eigenvectors = eigen.get_eigenvectors_test(trajectory.structure)
eigenvectors,original_frequencies = pho_interface.obtain_eigenvectors_from_phonopy(trajectory.structure,q_vector)


#Projection onto unit cell
vc = projection.project_onto_unit_cell(trajectory,q_vector)
plt.plot(trajectory.get_time().real,vc[:,0,:].real)
plt.show()


#Projection onto phonon coordinates
vq = projection.project_onto_phonon(vc,eigenvectors)
plt.plot(trajectory.get_time().real,vq[:,2:4].real)
plt.show()


#Noise generation (for test only)
#for i in range(vq.shape[0]):
#    vq[i,:] += random.uniform(-0.5,0.5)

# Correlation section (working on..)
print ('Correlation')

test_frequencies_range = np.array([0.001*i + 0.55 for i in range (200)])

correlation_vector = np.zeros((test_frequencies_range.shape[0],vq.shape[1]),dtype=complex)

for i in range (vq.shape[1]):
    print 'Frequency:',i
    for k in range (test_frequencies_range.shape[0]):
        Frequency = test_frequencies_range[k]
#        correlation_vector[k,i] = correlation.correlation(Frequency,vq[:,i],trajectory.get_time())
        correlation_vector[k,i] = correlation.correlation2(Frequency,vq[:,i],(trajectory.get_time_step_average()),100)
        print (correlation_vector[k,i].real)

    print('\n')
    #print(Time)


plt.plot(test_frequencies_range,correlation_vector.real)
plt.show()

#Search for frequencies
frequencies = peaksearch.get_frequencies_from_correlation(correlation_vector,test_frequencies_range)
print 'Frequencies:',frequencies


#Dynamical Matrix section
print('Matrix section\n')

print('Final Dynamical Matrix')
new_frequencies, new_eigenvectors, dynamical_matrix = eigen.build_dynamical_matrix(trajectory.structure,frequencies,eigenvectors)
print(dynamical_matrix)

print('\n')
print('EigenVectors & EigenValues')
print(new_frequencies)

exit()

