import numpy as np

def generate_VASP_structure(structure, scaled=False, super_cell=(1, 1, 1)):

    cell = structure.get_cell(super_cell=super_cell)

    types = structure.get_atomic_types(super_cell=super_cell)
    atom_type_unique = np.unique(types, return_counts=True)

    elements = atom_type_unique[0]
    elements_count = atom_type_unique[1]

    vasp_POSCAR = 'Generated using dynaphopy\n'
    vasp_POSCAR += '1.0\n'
    for row in cell.T:
        vasp_POSCAR += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*row)
    vasp_POSCAR += ' '.join(elements)
    vasp_POSCAR += ' \n'
    vasp_POSCAR += ' '.join([str(i) for i in elements_count])

    if scaled:
        scaled_positions = structure.get_scaled_positions(super_cell=super_cell)
        vasp_POSCAR += '\nDirect\n'
        for row in scaled_positions:
            vasp_POSCAR += '{0:15.15f}   {1:15.15f}   {2:15.15f}\n'.format(*row)

    else:
        positions = structure.get_positions(super_cell=super_cell)
        vasp_POSCAR += '\nCartesian\n'
        for row in positions:
            vasp_POSCAR += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*row)

    return vasp_POSCAR


def generate_LAMMPS_structure(structure, super_cell=(1, 1, 1), by_element=True):

    cell = structure.get_cell(super_cell=super_cell)
    types = structure.get_atomic_types(super_cell=super_cell)
    atom_index_unique = np.unique(types, return_index=True)[1]

    if by_element:
        count_index_unique = np.unique(types, return_counts=True)[1]

        atom_index = []
        for i, index in enumerate(count_index_unique):
            atom_index += [i for j in range(index)]


        #atom_index = structure.get_atom_type_index(super_cell=super_cell)

    else:
        atom_index = structure.get_atom_type_index(super_cell=super_cell)

    masses = structure.get_masses(super_cell=super_cell)

    atom_mass_unique = np.unique(masses, return_counts=True)

    positions = structure.get_positions(super_cell=super_cell)
    number_of_atoms = len(positions)


    lammps_data_file = 'Generated using dynaphopy\n\n'
    lammps_data_file += '{0} atoms\n\n'.format(number_of_atoms)

    lammps_data_file += '{0} atom types\n\n'.format(len(atom_index_unique))

    for row in cell.T:
        lammps_data_file += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*row)

    xlo = 0
    ylo = 0
    zlo = 0
    xhi = 0

    xy = 0
    xz = 0
    yz = 0

    lammps_data_file += '\n{0} {1} xlo xhi\n'.format(xlo, xhi)
    lammps_data_file += '{0} {1} ylo yhi\n'.format(xlo, xhi)
    lammps_data_file += '{0} {1} zlo zhi\n'.format(xlo, xhi)
    lammps_data_file += '{0} {1} {2} xy xz yz\n\n'.format(xy, xz, yz)

    lammps_data_file += 'Masses\n\n'

    for i, index in enumerate(atom_index_unique):
        lammps_data_file += '{0} {1:20.10f} \n'.format(i+1, masses[index])

    lammps_data_file += '\nAtoms\n\n'
    for i, row in enumerate(positions):
        lammps_data_file += '{0} {1} {2:20.10f} {3:20.10f} {4:20.10f}\n'.format(i, atom_index[i], row[0],row[1],row[2])

    return lammps_data_file



if __name__ == "__main__":

    import dynaphopy.interface.iofile as reading
    input_parameters = reading.read_parameters_from_input_file('/home/abel/VASP/Ag2Cu2O4/MD/input_dynaphopy')
    structure = reading.read_from_file_structure_poscar(input_parameters['structure_file_name_poscar'])
 #   print(generate_VASP_structure(structure))
    print(generate_LAMMPS_structure(structure))