import numpy as np
import itertools

class Structure:

    def __init__(self,
                 positions = None,
                 scaled_positions = None,
                 masses = None,
                 cell = None,
                 force_set = None,
                 force_constants = None,
                 atomic_numbers = None,
                 atomic_types = None,
                 atom_type_index = None,
                 primitive_cell = None):

        """
        :param atoms cartesian positions (array Ndim x Natoms):
        :param atom positions scaled to 1 (array Ndim x Natoms):
        :param masses of the atoms (vector NAtoms):
        :param real space cell matrix (array Ndim x Ndim):
        :param force constants:
        :param atomic numbers vector (1x Natoms):
        :param number of total atoms in the crystal:
        :param atomic names of each type of atom (ex: H, Be, Si,..) (vector Natoms):
        :param index vector that contains the number of different types of atoms in crystal (vector NdiferentAtoms):
        :param number of atoms in the defined cell:
        """

        self._cell = np.array(cell, dtype='double')
        self._masses = np.array(masses, dtype='double')
        self._atomic_numbers = np.array(atomic_numbers, dtype='double')
        self._force_constants = np.array(force_constants, dtype='double')

        self._force_set = force_set
        self._atomic_types = atomic_types
        self._atom_type_index = atom_type_index
        self._scaled_positions = scaled_positions
        self._positions = positions
        self._primitive_cell = primitive_cell

        self._super_cell_matrix = None
        self._super_cell_phonon = None
        self._number_of_cell_atoms = None
        self._number_of_atoms = None
        self._number_of_atom_types = None
        self._primitive_matrix = None
        self._number_of_primitive_atoms = None

        #Get normalized cell from cell
        self._cell_normalized = cell / np.linalg.norm(cell, axis=-1)[:, np.newaxis]

        #Get atomic types from masses i available
        if atomic_types is None and masses is not None:
            self._atomic_types = []
            for i in masses:
                for j in atom_data:
                    if "{0:.1f}".format(i) == "{0:.1f}".format(j[3]):
                        self._atomic_types.append(j[1])

        #Get atomic numbers and masses from available information
        if atomic_numbers is None and self._atomic_types is not None:
            self._atomic_numbers =  np.array([ symbol_map[i] for i in self._atomic_types ])
        else:
            self._atomic_numbers = np.array(atomic_numbers)

        if masses is None and self._atomic_numbers is not None:
            self._masses = np.array([ atom_data[i][3] for i in self._atomic_numbers ])
        else:
            self._masses = masses

    #-------- Methods start here -----------

    #Getting data
    def get_data_from_dict(self, data_dictionary):
        for data in self.__dict__:
            try:
                self.__dict__[data] = data_dictionary[data]
            except KeyError:
                continue


    # Cell related methods
    def set_cell(self, cell):
        self._cell = np.array(cell, dtype='double')


    def get_cell(self):
        return self._cell


    def set_super_cell_phonon(self,super_cell_phonon):
        self._super_cell_phonon = super_cell_phonon


    def get_super_cell_phonon(self):
        if self._super_cell_phonon is None:
            self._super_cell_phonon = np.identity(self.get_number_of_dimensions(),dtype=int)
        return self._super_cell_phonon


    def set_super_cell_matrix(self,super_cell_matrix):
        self._super_cell_matrix = super_cell_matrix


    def get_super_cell_matrix(self):
        if self._super_cell_matrix is None:
            self._super_cell_matrix = np.array(self.get_number_of_dimensions()*[1],dtype=int)
        return self._super_cell_matrix


    def get_primitive_cell(self):
        if self._primitive_cell is None:
            self._primitive_cell = np.dot(self.get_cell(),self.get_primitive_matrix())
        return self._primitive_cell


    #Cell matrix related methods
    def set_primitive_matrix(self,primitive_matrix):
        self._primitive_matrix = primitive_matrix
        self._number_of_atom_types = None
        self._number_of_primitive_atoms = None
        self._atom_type_index = None
        self._primitive_cell = None


    def get_primitive_matrix(self):
        if self._primitive_matrix is None:
            if self._primitive_cell is None:
                print('Warning: No primitive matrix defined! Using unit cell as primitive')
                self._primitive_matrix = np.identity(self.get_number_of_dimensions())
            else:
                self._primitive_matrix = np.dot(np.linalg.inv(self.get_cell().T),self._primitive_cell)
        return  self._primitive_matrix


    #Positions related methods
    def set_positions(self, cart_positions):
        self._scaled_positions = np.dot(cart_positions, np.linalg.inv(self.get_cell().T))


    def get_positions(self, super_cell=None):
        if self._positions is None:
            if self._scaled_positions is None:
                print('No positions provided')
                exit()
            else:
                self._positions = np.dot(self._scaled_positions, self.get_cell().T)

        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        position_super_cell = []
        for k in range(self._positions.shape[0]):
            for r in itertools.product(*[range (i) for i in super_cell[::-1]]):
                position_super_cell.append(self._positions[k,:] + np.dot(np.array(r[::-1]),self.get_cell().T))
        position_super_cell = np.array(position_super_cell)

        return position_super_cell

    def get_scaled_positions(self):
        if self._scaled_positions is  None:
            self._scaled_positions = np.dot(self.get_positions(), np.linalg.inv(self.get_cell().T))
        return self._scaled_positions


    #Force related methods
    def set_force_constants(self, force_constants):
        self._force_constants = np.array(force_constants,dtype='double')


    def get_force_constants(self):
        return np.array(self._force_constants)


    def set_force_set(self, force_set):
        self._force_set = force_set


    def get_force_set(self):
        if self._force_set is None:
            print('No force sets specified!')
            exit()
        return self._force_set


    def set_masses(self, masses):
        self._masses = np.array(masses, dtype='double')


    def get_masses(self,super_cell=None):
        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        mass_super_cell = []
        for j in range(self.get_number_of_cell_atoms()):
            mass_super_cell += [ self._masses[j] ] * np.prod(super_cell)
        return mass_super_cell


    #Number of atoms and dimensions related methods
    def get_number_of_cell_atoms(self):
        if self._number_of_cell_atoms is None:
            self._number_of_cell_atoms = self.get_positions().shape[0]
        return self._number_of_cell_atoms


    def get_number_of_dimensions(self):
        return self.get_cell().shape[0]


    def get_atomic_numbers(self,super_cell=None):
        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        atomic_numbers_super_cell = []
        for j in range(self.get_number_of_cell_atoms()):
            atomic_numbers_super_cell += [self._atomic_numbers[j] ] * np.prod(super_cell)

        return np.array(atomic_numbers_super_cell,dtype=int)


    def get_number_of_atoms(self):
        if self._number_of_atoms is None:
            self._number_of_atoms = self.get_number_of_cell_atoms()
        return self._number_of_atoms


    def set_number_of_atoms(self,number_of_atoms):
        self._number_of_atoms = number_of_atoms


    def get_number_of_atom_types(self):
        if self._number_of_atom_types is None:
            self._number_of_atom_types = len(set(self.get_atom_type_index()))
    #         print(self._number_of_atom_types)
        return  self._number_of_atom_types

    def get_number_of_primitive_atoms(self):
        if self._number_of_primitive_atoms is None:
            self._number_of_primitive_atoms =  len(set(self.get_atom_type_index()))
        return  self._number_of_primitive_atoms


    def set_number_of_primitive_atoms(self,number_of_primitive_atoms):
        self._number_of_primitive_atoms = number_of_primitive_atoms


    #Atomic types related methods
    def get_atomic_types(self,super_cell=None):
        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        atomic_types = []
        for j in range(self.get_number_of_cell_atoms()):
            atomic_types += [self._atomic_types[j] ] * np.prod(super_cell)
        return atomic_types


    def set_atom_type_index_by_element(self):
        if self._atom_type_index is None:
            self._atom_type_index = self._atomic_numbers.copy()
            for i in range(self.get_number_of_cell_atoms()):
                for index in range(self.get_number_of_atom_types()):
                    if self._atomic_numbers[index] == self._atomic_numbers[i]:
                        self._atom_type_index[i] = index


    def get_atom_type_index(self,super_cell=None):

#       Tolerance for accepting equivalent atoms in super cell
        masses = self.get_masses(super_cell=super_cell)
        tolerance = 0.001
        if self._atom_type_index is None:
            primitive_cell_inverse = np.linalg.inv(self.get_primitive_cell())

            self._atom_type_index = np.array(self.get_number_of_cell_atoms() * [None])
            counter = 0
            for i in range(self.get_number_of_cell_atoms()):
                if self._atom_type_index[i] is None:
                    self._atom_type_index[i] = counter
                    counter += 1
                for j in range(i+1, self.get_number_of_cell_atoms()):
                    coordinates_atom_i = self.get_positions()[i]
                    coordinates_atom_j = self.get_positions()[j]

                    difference_in_cell_coordinates = np.around((np.dot(primitive_cell_inverse,(coordinates_atom_j - coordinates_atom_i))))
                    projected_coordinates_atom_j = coordinates_atom_j - np.dot(self.get_primitive_cell(), difference_in_cell_coordinates)
                    separation = pow(np.linalg.norm(projected_coordinates_atom_j - coordinates_atom_i),2)

                    if separation < tolerance and  masses[i] == masses[j]:
                        self._atom_type_index[j] = self._atom_type_index[i]
        self._atom_type_index = np.array(self._atom_type_index,dtype=int)

        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        atom_type_index_super_cell = []
        for j in range(self.get_number_of_cell_atoms()):
            atom_type_index_super_cell += [self._atom_type_index[j] ] * np.prod(super_cell)
        return  atom_type_index_super_cell


    def get_commensurate_points(self, super_cell=None):

        if super_cell is None:
            super_cell = self.get_number_of_dimensions() * [1]

        primitive_matrix = self.get_primitive_matrix()

        commensurate_points = []
        for k1 in np.arange(-0.5, 0.5, 1./(super_cell[0]*2)):
            for k2 in np.arange(-0.5, 0.5, 1./(super_cell[1]*2)):
                for k3 in np.arange(-0.5, 0.5, 1./(super_cell[2]*2)):
    
                    q_point = [np.around(k1,decimals=5), np.around(k2,decimals=5), np.around(k3,decimals=5)]
    
                    q_point_unit_cell = np.dot(q_point, np.linalg.inv(primitive_matrix))
                    q_point_unit_cell = np.multiply(q_point_unit_cell, super_cell)
    
                    if np.all(np.equal(np.mod(q_point_unit_cell, 1), 0)):
                        commensurate_points.append(q_point)
                        
        return commensurate_points




atom_data = [
    [  0, "X", "X", 0], # 0
    [  1, "H", "Hydrogen", 1.00794], # 1
    [  2, "He", "Helium", 4.002602], # 2
    [  3, "Li", "Lithium", 6.941], # 3
    [  4, "Be", "Beryllium", 9.012182], # 4
    [  5, "B", "Boron", 10.811], # 5
    [  6, "C", "Carbon", 12.0107], # 6
    [  7, "N", "Nitrogen", 14.0067], # 7
    [  8, "O", "Oxygen", 15.9994], # 8
    [  9, "F", "Fluorine", 18.9984032], # 9
    [ 10, "Ne", "Neon", 20.1797], # 10
    [ 11, "Na", "Sodium", 22.98976928], # 11
    [ 12, "Mg", "Magnesium", 24.3050], # 12
    [ 13, "Al", "Aluminium", 26.9815386], # 13
    [ 14, "Si", "Silicon", 28.0855], # 14
    [ 15, "P", "Phosphorus", 30.973762], # 15
    [ 16, "S", "Sulfur", 32.065], # 16
    [ 17, "Cl", "Chlorine", 35.453], # 17
    [ 18, "Ar", "Argon", 39.948], # 18
    [ 19, "K", "Potassium", 39.0983], # 19
    [ 20, "Ca", "Calcium", 40.078], # 20
    [ 21, "Sc", "Scandium", 44.955912], # 21
    [ 22, "Ti", "Titanium", 47.867], # 22
    [ 23, "V", "Vanadium", 50.9415], # 23
    [ 24, "Cr", "Chromium", 51.9961], # 24
    [ 25, "Mn", "Manganese", 54.938045], # 25
    [ 26, "Fe", "Iron", 55.845], # 26
    [ 27, "Co", "Cobalt", 58.933195], # 27
    [ 28, "Ni", "Nickel", 58.6934], # 28
    [ 29, "Cu", "Copper", 63.546], # 29
    [ 30, "Zn", "Zinc", 65.38], # 30
    [ 31, "Ga", "Gallium", 69.723], # 31
    [ 32, "Ge", "Germanium", 72.64], # 32
    [ 33, "As", "Arsenic", 74.92160], # 33
    [ 34, "Se", "Selenium", 78.96], # 34
    [ 35, "Br", "Bromine", 79.904], # 35
    [ 36, "Kr", "Krypton", 83.798], # 36
    [ 37, "Rb", "Rubidium", 85.4678], # 37
    [ 38, "Sr", "Strontium", 87.62], # 38
    [ 39, "Y", "Yttrium", 88.90585], # 39
    [ 40, "Zr", "Zirconium", 91.224], # 40
    [ 41, "Nb", "Niobium", 92.90638], # 41
    [ 42, "Mo", "Molybdenum", 95.96], # 42
    [ 43, "Tc", "Technetium", 0], # 43
    [ 44, "Ru", "Ruthenium", 101.07], # 44
    [ 45, "Rh", "Rhodium", 102.90550], # 45
    [ 46, "Pd", "Palladium", 106.42], # 46
    [ 47, "Ag", "Silver", 107.8682], # 47
    [ 48, "Cd", "Cadmium", 112.411], # 48
    [ 49, "In", "Indium", 114.818], # 49
    [ 50, "Sn", "Tin", 118.710], # 50
    [ 51, "Sb", "Antimony", 121.760], # 51
    [ 52, "Te", "Tellurium", 127.60], # 52
    [ 53, "I", "Iodine", 126.90447], # 53
    [ 54, "Xe", "Xenon", 131.293], # 54
    [ 55, "Cs", "Caesium", 132.9054519], # 55
    [ 56, "Ba", "Barium", 137.327], # 56
    [ 57, "La", "Lanthanum", 138.90547], # 57
    [ 58, "Ce", "Cerium", 140.116], # 58
    [ 59, "Pr", "Praseodymium", 140.90765], # 59
    [ 60, "Nd", "Neodymium", 144.242], # 60
    [ 61, "Pm", "Promethium", 0], # 61
    [ 62, "Sm", "Samarium", 150.36], # 62
    [ 63, "Eu", "Europium", 151.964], # 63
    [ 64, "Gd", "Gadolinium", 157.25], # 64
    [ 65, "Tb", "Terbium", 158.92535], # 65
    [ 66, "Dy", "Dysprosium", 162.500], # 66
    [ 67, "Ho", "Holmium", 164.93032], # 67
    [ 68, "Er", "Erbium", 167.259], # 68
    [ 69, "Tm", "Thulium", 168.93421], # 69
    [ 70, "Yb", "Ytterbium", 173.054], # 70
    [ 71, "Lu", "Lutetium", 174.9668], # 71
    [ 72, "Hf", "Hafnium", 178.49], # 72
    [ 73, "Ta", "Tantalum", 180.94788], # 73
    [ 74, "W", "Tungsten", 183.84], # 74
    [ 75, "Re", "Rhenium", 186.207], # 75
    [ 76, "Os", "Osmium", 190.23], # 76
    [ 77, "Ir", "Iridium", 192.217], # 77
    [ 78, "Pt", "Platinum", 195.084], # 78
    [ 79, "Au", "Gold", 196.966569], # 79
    [ 80, "Hg", "Mercury", 200.59], # 80
    [ 81, "Tl", "Thallium", 204.3833], # 81
    [ 82, "Pb", "Lead", 207.2], # 82
    [ 83, "Bi", "Bismuth", 208.98040], # 83
    [ 84, "Po", "Polonium", 0], # 84
    [ 85, "At", "Astatine", 0], # 85
    [ 86, "Rn", "Radon", 0], # 86
    [ 87, "Fr", "Francium", 0], # 87
    [ 88, "Ra", "Radium", 0], # 88
    [ 89, "Ac", "Actinium", 0], # 89
    [ 90, "Th", "Thorium", 232.03806], # 90
    [ 91, "Pa", "Protactinium", 231.03588], # 91
    [ 92, "U", "Uranium", 238.02891], # 92
    [ 93, "Np", "Neptunium", 0], # 93
    [ 94, "Pu", "Plutonium", 0], # 94
    [ 95, "Am", "Americium", 0], # 95
    [ 96, "Cm", "Curium", 0], # 96
    [ 97, "Bk", "Berkelium", 0], # 97
    [ 98, "Cf", "Californium", 0], # 98
    [ 99, "Es", "Einsteinium", 0], # 99
    [100, "Fm", "Fermium", 0], # 100
    [101, "Md", "Mendelevium", 0], # 101
    [102, "No", "Nobelium", 0], # 102
    [103, "Lr", "Lawrencium", 0], # 103
    [104, "Rf", "Rutherfordium", 0], # 104
    [105, "Db", "Dubnium", 0], # 105
    [106, "Sg", "Seaborgium", 0], # 106
    [107, "Bh", "Bohrium", 0], # 107
    [108, "Hs", "Hassium", 0], # 108
    [109, "Mt", "Meitnerium", 0], # 109
    [110, "Ds", "Darmstadtium", 0], # 110
    [111, "Rg", "Roentgenium", 0], # 111
    [112, "Cn", "Copernicium", 0], # 112
    [113, "Uut", "Ununtrium", 0], # 113
    [114, "Uuq", "Ununquadium", 0], # 114
    [115, "Uup", "Ununpentium", 0], # 115
    [116, "Uuh", "Ununhexium", 0], # 116
    [117, "Uus", "Ununseptium", 0], # 117
    [118, "Uuo", "Ununoctium", 0], # 118
    ]

symbol_map = {
    "H":1,
    "He":2,
    "Li":3,
    "Be":4,
    "B":5,
    "C":6,
    "N":7,
    "O":8,
    "F":9,
    "Ne":10,
    "Na":11,
    "Mg":12,
    "Al":13,
    "Si":14,
    "P":15,
    "S":16,
    "Cl":17,
    "Ar":18,
    "K":19,
    "Ca":20,
    "Sc":21,
    "Ti":22,
    "V":23,
    "Cr":24,
    "Mn":25,
    "Fe":26,
    "Co":27,
    "Ni":28,
    "Cu":29,
    "Zn":30,
    "Ga":31,
    "Ge":32,
    "As":33,
    "Se":34,
    "Br":35,
    "Kr":36,
    "Rb":37,
    "Sr":38,
    "Y":39,
    "Zr":40,
    "Nb":41,
    "Mo":42,
    "Tc":43,
    "Ru":44,
    "Rh":45,
    "Pd":46,
    "Ag":47,
    "Cd":48,
    "In":49,
    "Sn":50,
    "Sb":51,
    "Te":52,
    "I":53,
    "Xe":54,
    "Cs":55,
    "Ba":56,
    "La":57,
    "Ce":58,
    "Pr":59,
    "Nd":60,
    "Pm":61,
    "Sm":62,
    "Eu":63,
    "Gd":64,
    "Tb":65,
    "Dy":66,
    "Ho":67,
    "Er":68,
    "Tm":69,
    "Yb":70,
    "Lu":71,
    "Hf":72,
    "Ta":73,
    "W":74,
    "Re":75,
    "Os":76,
    "Ir":77,
    "Pt":78,
    "Au":79,
    "Hg":80,
    "Tl":81,
    "Pb":82,
    "Bi":83,
    "Po":84,
    "At":85,
    "Rn":86,
    "Fr":87,
    "Ra":88,
    "Ac":89,
    "Th":90,
    "Pa":91,
    "U":92,
    "Np":93,
    "Pu":94,
    "Am":95,
    "Cm":96,
    "Bk":97,
    "Cf":98,
    "Es":99,
    "Fm":100,
    "Md":101,
    "No":102,
    "Lr":103,
    "Rf":104,
    "Db":105,
    "Sg":106,
    "Bh":107,
    "Hs":108,
    "Mt":109,
    "Ds":110,
    "Rg":111,
    "Cn":112,
    "Uut":113,
    "Uuq":114,
    "Uup":115,
    "Uuh":116,
    "Uus":117,
    "Uuo":118,
    }
