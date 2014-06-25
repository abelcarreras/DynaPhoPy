
import numpy as np
import scitools.numpyutils

def orthogonalize(V):
    "Orthogonalization by Gram-Schmidt process (Maybe needed for c implementation)"
    V = np.array(V)
    U = np.copy(V)

    for i in xrange(0, V.shape[1]):
        for j in xrange(i):
            U[:,i] -= U[:,j] * np.dot(U[:,j], V[:,i])/ np.dot(U[:,j],U[:,j])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return E

#Cal arreglar (i molt!!)
def GS(A):
    m,n = A.shape; Q=np.zeros([m,n]); R=np.zeros([n,n]); v=np.zeros(m)
    for j in range(n):
        v[:] = A[:,j]
        for i in range(j):
            r = np.dot(Q[:,i],A[:,j]); R[i,j] = r
            v[:] = v[:] - r*Q[:,i]
            r = np.linalg.norm(v); R[j,j]= r
            Q[:,j] = v[:]/r
    return Q


def get_eigenvectors_test(estructura):

    #Manual eigenvectors definition (from Wolfram Mathematica calculations)
    number_of_cell_atoms = estructura.get_number_of_cell_atoms()
    number_of_dimensions = estructura.get_number_of_dimensions()

    eigenvectors = np.array([[0.707107,0,-0.707107,0],
                             [-7.542995783e-13,-0.707107,-6.91441e-16,0.707107],
                             [-0.3441510098,0.6177054982,-0.3441510098,0.6177054982],
                             [0.6177054982,0.3441510098,0.6177054982,0.3441510098]])


    eigenvectors=np.mat(orthogonalize(eigenvectors))
#    eigenvectors=np.mat(scitools.numpyutils.Gram_Schmidt(eigenvectors,normalize=True))

    frequencies = [0.690841,0.690841,0.648592,0.648592]

    print('Eigenvectors')
    print(eigenvectors)

    print('\n')

    print('Testing Orthogonality')
    print(eigenvectors.T*eigenvectors.conj())

    arranged_EV = np.array([[[eigenvectors [i,j*number_of_dimensions+k] for k in range(number_of_dimensions)] for j in range(number_of_cell_atoms)] for i in range(number_of_cell_atoms*number_of_dimensions)])

    return arranged_EV, frequencies

def build_dynamical_matrix(structure, frequencies, eigenvectors):

    number_of_cell_atoms = structure.get_number_of_cell_atoms()
    number_of_dimensions = structure.get_number_of_dimensions()


    dynamical_matrix=np.mat(np.zeros((number_of_dimensions*number_of_cell_atoms,number_of_dimensions*number_of_cell_atoms)),dtype=complex)

    for i in range(number_of_cell_atoms):
        for j in range(number_of_cell_atoms):
            SubDynamicalMatrix=np.mat(np.zeros((number_of_dimensions,number_of_dimensions)),dtype=complex)
            for f in range(number_of_cell_atoms*number_of_dimensions):
                SubDynamicalMatrix += frequencies[f]**2 *np.mat(eigenvectors[f,i,:]).T*np.mat(eigenvectors[f,j,:].conj())
            dynamical_matrix[i*number_of_dimensions:(i+1)*number_of_dimensions,j*number_of_dimensions:(j+1)*number_of_dimensions] = SubDynamicalMatrix

    new_frequencies, new_eigenvectors = np.linalg.eig (dynamical_matrix.real)
    new_frequencies = np.sqrt(new_frequencies)

#    new_eigenvectors=np.mat(orthogonalize(new_eigenvectors))
    new_eigenvectors = np.mat(scitools.numpyutils.Gram_Schmidt(new_eigenvectors,normalize=True))

    return new_frequencies, new_eigenvectors, dynamical_matrix

