from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import Functions.reading as reading
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.structure.cells import Primitive





def append_band(bands, q_start, q_end):
    band = []
    for i in range(51):
        band.append(np.array(q_start) +
                    (np.array(q_end) - np.array(q_start)) / 50 * i)
    bands.append(band)

#bulk = read_vasp("POSCAR")

#print (bulk.get_number_of_atoms())
#print (bulk.get_atomic_numbers())


#directory = '/home/abel/VASP/MgO-phonon/2x2x2/'
directory = '/home/abel/VASP/MgO-phonon/3x3x3/'










structure = reading.read_from_file_structure(directory+'OUTCAR')

structure.set_primitive_matrix([[0.0, 0.5, 0.5],
                                [0.5, 0.0, 0.5],
                                [0.5, 0.5, 0.0]])


print ('Cell atoms: ',structure.get_number_of_cell_atoms())
#exit()
print('N Atom types: ',structure.get_number_of_atom_types())
print('Atom types: ',structure.get_atomic_types())


bulk = PhonopyAtoms(symbols=structure.get_atomic_types(),
                    scaled_positions=structure.get_scaled_positions())
bulk.set_cell(structure.get_cell())



p_matrix = structure.get_primitive_matrix()

print(np.array(p_matrix))


phonon = Phonopy(bulk,
                 [[3, 0, 0],
                  [0, 3, 0],
                  [0, 0, 3]],
                 primitive_matrix=p_matrix,
                 is_auto_displacements=False)


#NAC PARAMETER
get_is_symmetry = True  #sfrom phonopy:   settings.get_is_symmetry()
primitive = phonon.get_primitive()
nac_params = parse_BORN(primitive, get_is_symmetry)
phonon.set_nac_params(nac_params=nac_params)



symmetry = phonon.get_symmetry()
print "Space group:", symmetry.get_international_table()
print ("Unit Cell")
print (structure.get_cell())
print ("Primitive cell")
print (structure.get_primitive_cell())
print("Number of primitive atoms", structure.get_number_of_primitive_atoms())

force_sets = parse_FORCE_SETS(phonon.get_supercell().get_number_of_atoms(),filename=directory+'FORCE_SETS')

phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()


#q_vector = [0.5, 0.5, 0.5]

#print (phonon.get_frequencies_with_eigenvectors(q_vector)[0])

print (phonon._build_primitive_cell())

print (bulk.get_cell())

q_vector = [0,0.5,0]

print("wave vector")
print(q_vector)
frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)
print("frequencies")
print(frequencies)
print("eigenvectors")
print(eigenvectors[:,0].real)



print (np.linalg.inv(p_matrix))
number_of_dimensions = 3
number_of_primitive_atoms = structure.get_number_of_primitive_atoms()

print(eigenvectors.real)

arranged_ev = np.array([[[eigenvectors [j*number_of_dimensions+k,i]
                                for k in range(number_of_dimensions)]
                                for j in range(number_of_primitive_atoms)]
                                for i in range(number_of_primitive_atoms*number_of_dimensions)])



print("###########################################################################")
print("Original")
print(arranged_ev[0,:,:].real)




print("projectat")
new_eigenvector0 = np.dot(np.array(p_matrix).T,arranged_ev[0,0,:])
new_eigenvector1 = np.dot(np.array(p_matrix).T,arranged_ev[0,1,:])
print(new_eigenvector0.real)
print(new_eigenvector1.real)



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

for i in range(2):
    x = [0,arranged_ev[0,i,0].real]
    y = [0,arranged_ev[0,i,1].real]
    z = [0,arranged_ev[0,i,2].real]
    ax.plot(x, y, z, label='Original',color='r')


x = [0,new_eigenvector0[0].real]
y = [0,new_eigenvector0[1].real]
z = [0,new_eigenvector0[2].real]

ax.plot(x, y, z, label='Projection',color='b')

x = [0,new_eigenvector1[0].real]
y = [0,new_eigenvector1[1].real]
z = [0,new_eigenvector1[2].real]

ax.plot(x, y, z,color='b')



print("#######################################################################")


print("wave vector")
print(q_vector)
frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q_vector)
arranged_ev2 = np.array([[[eigenvectors [j*number_of_dimensions+k,i]
                                for k in range(number_of_dimensions)]
                                for j in range(number_of_primitive_atoms)]
                                for i in range(number_of_primitive_atoms*number_of_dimensions)])


print("frequencies objectiu")
print(frequencies)
print("eigenvectors objectiu")
#print(eigenvectors[:,0].real)
print(arranged_ev2[0,:,:].real)


for i in range(2):
    x = [0,arranged_ev2[0,i,0].real]
    y = [0,arranged_ev2[0,i,1].real]
    z = [0,arranged_ev2[0,i,2].real]
    ax.plot(x, y, z, label='Objective',color='g')


ax.legend()
plt.show()







#exit()

if False:

    #Phonopy bands calculation
    bands = []
    q_start  = np.array([0.0, 0.0, 0.0])
    q_end    = np.array([0.0, 0.5, 0.0])
    band = []
    for i in range(51):
        print(i)
        q_vector = (q_start + (q_end - q_start) / 50 * i)
        frequencies = phonon.get_frequencies_with_eigenvectors(q_vector)[0]
        print(frequencies)
        band.append(frequencies)

#    print (band)
    q_start  = np.array([0.0, 0.5, 0.5])
    q_end    = np.array([0.0, 0.0, 0.0])
    for i in range(1,51):
        print(i)
        q_vector = (q_start + (q_end - q_start) / 50 * i)
        frequencies = phonon.get_frequencies_with_eigenvectors(q_vector)[0]
        print(frequencies)
        band.append(frequencies)


    q_start  = np.array([0.0, 0.0, 0.0])
    q_end    = np.array([0.25, 0.25, 0.25])
    for i in range(1,51):
        print(i)
        q_vector = (q_start + (q_end - q_start) / 50 * i)
        frequencies = phonon.get_frequencies_with_eigenvectors(q_vector)[0]
        print(frequencies)
        band.append(frequencies)


    np.savetxt('/home/abel/Dropbox/PycharmProjects/DynaPhoPy/Data Files/band4', np.array(band))

    plt.plot(band)
    plt.show()



#symmetry = phonon.get_symmetry()
#print "Space group:", symmetry.get_international_table()

#print phonon.get_supercell().get_number_of_atoms()


# BAND = 0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5
bands = []
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])

#append_band(bands, [0.0, 0.0, 0.0], [0.0, 0.5, 0.0])
#append_band(bands, [0.0, 0.5, 0.5], [0.0, 0.0, 0.0])
#append_band(bands, [0.0, 0.0, 0.0], [0.25, 0.25, 0.25])


phonon.set_band_structure(bands)
q_points, distances, frequencies, eigvecs = phonon.get_band_structure()

#print eigvecs

for q, d, freq in zip(q_points, distances, frequencies):
    print q.real, d.real, freq.real
phonon.plot_band_structure().show()

exit()


























# Mesh sampling 20x20x20
phonon.set_mesh([20, 20, 20])
phonon.set_thermal_properties(t_step=10,
                              t_max=1000,
                              t_min=0)

# DOS
phonon.set_total_DOS(sigma=0.1)
for omega, dos in np.array(phonon.get_total_DOS()).T:
    print "%15.7f%15.7f" % (omega, dos)
phonon.plot_total_DOS().show()

# Thermal properties
for t, free_energy, entropy, cv in np.array(phonon.get_thermal_properties()).T:
    print ("%12.3f " + "%15.7f" * 3) % ( t, free_energy, entropy, cv )
phonon.plot_thermal_properties().show()

# PDOS
phonon.set_mesh([20, 20, 20], is_eigenvectors=True)
phonon.set_partial_DOS(sigma=0.1)
omegas, pdos = phonon.get_partial_DOS()
pdos_indices = [[0], [1]]
phonon.plot_partial_DOS(pdos_indices=pdos_indices,
                        legend=pdos_indices).show()
