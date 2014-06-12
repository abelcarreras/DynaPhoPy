import numpy as np
import phonopy.file_IO as file_IO
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms



a = 5.404
bulk = PhonopyAtoms(symbols=['Si']*8,
                    scaled_positions=[(0, 0, 0),
                                      (0, 0.5, 0.5),
                                      (0.5, 0, 0.5),
                                      (0.5, 0.5, 0),
                                      (0.25, 0.25, 0.25),
                                      (0.25, 0.75, 0.75),
                                      (0.75, 0.25, 0.75),
                                      (0.75, 0.75, 0.25)] )

bulk.set_cell(np.diag((a, a, a)))
phonon = Phonopy(bulk, [[2,0,0],[0,2,0],[0,0,2]], distance=0.01)

fc_and_atom_types = file_IO.read_force_constant_vasprun_xml('/home/abel/VASP/Si-test/vasprun.xml')

force_constants, atom_types = fc_and_atom_types

print(force_constants[0,0,:,:])


def get_file ():

    f = open('/home/abel/VASP/Si-test/FORCE_CONSTANTS', 'r')
    force_constants = np.zeros((8,8,3,3))
    f.readline()
    for i in range(8):
        for j in range(8):
            f.readline()
            for x in range(3):
                row = f.readline().split()
                for y in range(len(row)): force_constants[i,j,x,y] = float(row[y])
    return  force_constants


force_constants2 = get_file()


phonon.set_force_constants(force_constants)


frequencies, eigvecs = phonon.get_frequencies_with_eigenvectors(np.array([0.0, 0.0, 0.0]))


print(eigvecs)



bands = []
q_start  = np.array([0.5, 0.5, 0.0])
q_end    = np.array([0.0, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

q_start  = np.array([0.0, 0.0, 0.0])
q_end    = np.array([0.5, 0.0, 0.0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

phonon.set_band_structure(bands)
phonon.plot_band_structure().show()


exit()