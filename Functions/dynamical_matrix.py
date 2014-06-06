__author__ = 'abel'
import numpy as np

def build_dynamical_matrix(structure, frequencies, eigenvectors):

    print ('Dynamical Matrix')
    number_of_cell_atoms = structure.get_number_of_cell_atoms()
    number_of_dimensions = structure.get_number_of_dimensions()



    dynamical_matrix=np.mat(np.zeros((number_of_dimensions*number_of_cell_atoms,number_of_dimensions*number_of_cell_atoms)))

    for i in range(number_of_cell_atoms):
        for j in range(number_of_cell_atoms):
            SubDynamicalMatrix=np.mat(np.zeros((number_of_dimensions,number_of_dimensions)))
            for f in range(number_of_cell_atoms*number_of_dimensions):
    #            for alfa in range(NumberOfDimensions):
    #                for beta in range(NumberOfDimensions):
    #                    print(ArrangedEV[f,i,alfa])
    #                    print(ArrangedEV[f,j,beta])
    #                    SubDynamicalMatrix[alfa,beta] += FreqTest[f]**2 *ArrangedEV[f,i,alfa]*ArrangedEV[f,j,beta].conj()
                SubDynamicalMatrix += frequencies[f]**2 *np.mat(eigenvectors[f,i,:]).T*np.mat(eigenvectors[f,j,:].conj())

    #            print(SubDynamicalMatrix)
            dynamical_matrix[i*number_of_dimensions:(i+1)*number_of_dimensions,j*number_of_dimensions:(j+1)*number_of_dimensions] = SubDynamicalMatrix

    print('Final Dynamical Matrix')
    print(dynamical_matrix)

    print('\n')
    print('EigenVectors & EigenValues')
    new_frequencies, new_eigenvectors = np.linalg.eig (dynamical_matrix)
    print(np.sqrt(new_frequencies))

    return new_frequencies, new_eigenvectors, dynamical_matrix