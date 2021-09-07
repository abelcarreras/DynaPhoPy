import numpy as np


def project_onto_wave_vector(trajectory, q_vector, project_on_atom=-1):

    number_of_primitive_atoms = trajectory.structure.get_number_of_primitive_atoms()
    velocity = trajectory.get_velocity_mass_average()
#    velocity = trajectory.velocity   # (use the velocity without mass average, just for testing)

    number_of_atoms = velocity.shape[1]
    number_of_dimensions = velocity.shape[2]
    supercell = trajectory.get_supercell_matrix()

    coordinates = trajectory.structure.get_positions(supercell)
    atom_type = trajectory.structure.get_atom_type_index(supercell=supercell)

    velocity_projected = np.zeros((velocity.shape[0], number_of_primitive_atoms, number_of_dimensions), dtype=complex)

    if q_vector.shape[0] != coordinates.shape[1]:
        print("Warning!! Q-vector and coordinates dimension do not match")
        exit()

    #Projection into wave vector
    for i in range(number_of_atoms):
        # Projection on atom
        if project_on_atom > -1:
            if atom_type[i] != project_on_atom:
                continue

        for k in range(number_of_dimensions):
            velocity_projected[:, atom_type[i], k] += velocity[:,i,k]*np.exp(-1j*np.dot(q_vector, coordinates[i,:]))

   #Normalize velocities (method 1)
  #  for i in range(velocity_projected.shape[1]):
  #      velocity_projected[:,i,:] /= atom_type.count(i)

   #Normalize velocities (method 2)
    number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms
    velocity_projected /= np.sqrt(number_of_primitive_cells)
    return velocity_projected


def project_onto_phonon(vc, eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    #Projection in phonon coordinate
    velocity_projected=np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)
    for k in range(number_of_frequencies):
        for i in range(number_of_cell_atoms):
            velocity_projected[:, k] += np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())

    return velocity_projected


#Just for testing (slower implementation) [but equivalent]
def project_onto_phonon2(vc,eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    #Projection in phonon coordinate
    velocity_projected=np.zeros((vc.shape[0],number_of_frequencies),dtype=complex)

    for i in range (vc.shape[0]):
        for k in range(number_of_frequencies):
            velocity_projected[i,k] = np.trace(np.dot(vc[i,:,:], eigenvectors[k,:,:].T.conj()))
#            velocity_projected[i,k] = np.sum(np.linalg.eigvals(np.dot(vc[i,:,:],eigenvectors[k,:,:].T.conj())))
    return velocity_projected

