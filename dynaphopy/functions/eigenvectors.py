
import numpy as np
#import scitools.numpyutils

def eigenvectors_normalization(eigenvector):
    for i in range(eigenvector.shape[0]):
        eigenvector[i,:] = eigenvector[i,:]/np.linalg.norm(eigenvector[i,:])
    return eigenvector

def get_eigenvectors_test(estructure):

    #Manual eigenvectors definition (from Wolfram Mathematica calculations)
    number_of_cell_atoms = estructure.get_number_of_primitive_atoms()
    number_of_dimensions = estructure.get_number_of_dimensions()
    eigenvectors = np.array([[0.707107,0,-0.707107,0],
                             [-7.542995783e-13,-0.707107,-6.91441e-16,0.707107],
                             [-0.3441510098,0.6177054982,-0.3441510098,0.6177054982],
                             [0.6177054982,0.3441510098,0.6177054982,0.3441510098]])

#    eigenvectors=np.mat(scitools.numpyutils.Gram_Schmidt(eigenvectors.real,normalize=True))
    eigenvectors = eigenvectors_normalization(eigenvectors)
    frequencies = [0.690841,0.690841,0.648592,0.648592]

    print('Eigenvectors')
    print(eigenvectors)

    print('Testing Orthogonality')
    print(np.dot(eigenvectors.T,np.ma.conjugate(eigenvectors)))

    arranged_EV = np.array([[[eigenvectors [j*number_of_dimensions+k,i] for k in range(number_of_dimensions)] for j in range(number_of_cell_atoms)] for i in range(number_of_cell_atoms*number_of_dimensions)])

    return arranged_EV, frequencies

def build_dynamical_matrix(structure, frequencies, eigenvectors):

    number_of_primitive_atoms = structure.get_number_of_primitive_atoms()
    number_of_dimensions = structure.get_number_of_dimensions()


    dynamical_matrix=np.mat(np.zeros((number_of_dimensions*number_of_primitive_atoms,number_of_dimensions*number_of_primitive_atoms)),dtype=complex)

    for i in range(number_of_primitive_atoms):
        for j in range(number_of_primitive_atoms):
            SubDynamicalMatrix=np.mat(np.zeros((number_of_dimensions,number_of_dimensions)),dtype=complex)
            for f in range(number_of_primitive_atoms*number_of_dimensions):
                SubDynamicalMatrix += frequencies[f]**2 *np.mat(eigenvectors[f,i,:]).T*np.mat(eigenvectors[f,j,:].conj())
            dynamical_matrix[i*number_of_dimensions:(i+1)*number_of_dimensions,j*number_of_dimensions:(j+1)*number_of_dimensions] = SubDynamicalMatrix

    new_frequencies, new_eigenvectors = np.linalg.eig (dynamical_matrix.real)
    new_frequencies = np.sqrt(new_frequencies)

#    new_eigenvectors = np.mat(scitools.numpyutils.Gram_Schmidt(new_eigenvectors.real,normalize=True))
    new_eigenvectors = eigenvectors_normalization(new_eigenvectors)

    return new_frequencies, new_eigenvectors, dynamical_matrix
