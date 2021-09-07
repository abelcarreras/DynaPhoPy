from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import lines
from mpl_toolkits.mplot3d import proj3d
import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0.0, 0.0), (0.0, 0.0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_phonon_modes(structure, eigenvectors, q_vector,
                      supercell=(1, 1, 1),
                      draw_primitive=False,
                      vectors_scale=10,
                      by_element=True):

    atom_type = structure.get_atom_type_index(supercell=supercell)
    positions = structure.get_positions(supercell=supercell)
    masses = structure.get_masses(supercell=supercell)
    elements = structure.get_atomic_elements(supercell=supercell)
    np.set_printoptions(precision=8, suppress=True)

    cell_t = structure.get_cell()
    if draw_primitive:
        cell_t = structure.get_primitive_cell()

    for i_phonon in range(eigenvectors.shape[0]):

        fig = plt.figure(i_phonon+1)
        if draw_primitive:
            fig.suptitle('Primitive cell')
        else:
            fig.suptitle('Unit cell')

        ax = fig.add_subplot(111, projection='3d')

        color_atom=['g','b','m', 'c', 'y', 'k', 'w', 'g', 'b', 'm', 'c', 'y', 'k', 'w','g','b','m', 'c', 'y',
                    'k', 'w', 'g', 'b', 'm', 'c', 'y', 'k', 'w','g','b','m', 'c', 'y', 'k', 'w', 'g', 'b']

        if by_element:
            elements_unique = np.unique(elements, return_inverse=True)[1]
        else:
            elements_unique = atom_type

        # Atom positions
        for i, atom in enumerate(positions):
            ax.plot(atom[0][None], atom[1][None], atom[2][None], 'o', markersize=atom_radius[elements[i]]*30, color=color_atom[elements_unique[i]], alpha=0.8)

        # Cell frame
        for i in range(3):
            cell_side = [(0, cell_t[i, 0]), (0, cell_t[i, 1]), (0, cell_t[i, 2])]
            ax.plot3D(*cell_side, color='b')
            for j in range(3):
                if i != j:
                    cell_side = [(cell_t[i, l],
                                  cell_t[i, l]+cell_t[j, l]) for l in range(3)]

                    ax.plot3D(*cell_side, color='b')
                    for k in range(3):
                        if k != i and k != j and j > i:
                            cell_side = [(cell_t[i, l]+cell_t[j, l],
                                          cell_t[i, l]+cell_t[j, l]+cell_t[k, l]) for l in range(3)]

                            ax.plot3D(*cell_side, color='b')

        # Atom positions
        for i, position in enumerate(positions):
            eigenvector_atom = np.array(eigenvectors[i_phonon, atom_type[i], :])
            phase = 1.j * np.dot(position, q_vector)
            vector = (eigenvector_atom / np.sqrt(masses[atom_type[i]]) * np.exp(phase) * vectors_scale).real
 #           vector = np.dot(vector, np.linalg.inv(structure.get_primitive_matrix().T))
            a = Arrow3D([position[0], position[0]+vector[0]], [position[1], position[1]+vector[1]],
                        [position[2], position[2]+vector[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
            ax.add_artist(a)

        # Legend
        atom_type_index_unique = np.unique(atom_type, return_index=True)[0]



        if by_element:
            atomic_types_unique = np.unique(elements, return_inverse=True)[0]
        else:
            atomic_types_unique = [elements[i] for i in atom_type_index_unique]

        legend_atoms =  [ lines.Line2D([0],[0], linestyle='none', c=color_atom[i], marker='o') for i, element in enumerate(atomic_types_unique)]
        ax.legend(legend_atoms, atomic_types_unique, numpoints = 1)

        # ax.set_axis_off()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title('Phonon {0}'.format(i_phonon+1))
        plt.axis('auto')
    plt.show()

    return


atom_radius = {
    'H':0.32,
    'He':0.93,
    'Li':1.23,
    'Be':0.9,
    'B':0.82,
    'C':0.77,
    'N':0.75,
    'O':0.73,
    'F':0.72,
    'Ne':0.71,
    'Na':1.54,
    'Mg':1.36,
    'Al':1.18,
    'Si':1.11,
    'P':1.06,
    'S':1.02,
    'Cl':0.99,
    'Ar':0.98,
    'K':2.03,
    'Ca':1.74,
    'Sc':1.44,
    'Ti':1.32,
    'V':1.22,
    'Cr':1.18,
    'Mn':1.17,
    'Fe':1.17,
    'Co':1.16,
    'Ni':1.15,
    'Cu':1.17,
    'Zn':1.25,
    'Ga':1.26,
    'Ge':1.22,
    'As':1.2,
    'Se':1.16,
    'Br':1.14,
    'Kr':1.12,
    'Rb':2.16,
    'Sr':1.91,
    'Y':1.62,
    'Zr':1.45,
    'Nb':1.34,
    'Mo':1.3,
    'Tc':1.27,
    'Ru':1.25,
    'Rh':1.25,
    'Pd':1.28,
    'Ag':1.34,
    'Cd':1.48,
    'In':1.44,
    'Sn':1.41,
    'Sb':1.4,
    'Te':1.36,
    'I':1.33,
    'Xe':1.31,
    'Cs':2.35,
    'Ba':1.98,
    'La':1.69,
    'Ce':1.65,
    'Pr':1.65,
    'Nd':1.64,
    'Pm':1.63,
    'Sm':1.62,
    'Eu':1.85,
    'Gd':1.61,
    'Tb':1.59,
    'Dy':1.59,
    'Ho':1.58,
    'Er':1.57,
    'Tm':1.56,
    'Yb':1.74,
    'Lu':1.56,
    'Hf':1.44,
    'Ta':1.34,
    'W':1.3,
    'Re':1.28,
    'Os':1.26,
    'Ir':1.27,
    'Pt':1.3,
    'Au':1.34,
    'Hg':1.49,
    'Tl':1.48,
    'Pb':1.47,
    'Bi':1.46,
    'Po':1.46,
    'At':1.45,
    'Rn':0.0,
    'Fr':0.0,
    'Ra':0.0,
    'Ac':0.0,
    'Th':1.65,
    'Pa':0.0,
    'U':1.42,
    'Np':0.0,
    'Pu':0.0,
    'Am':0.0,
    'Cm':0.0,
    'Bk':0.0,
    'Cf':0.0,
    'Es':0.0,
    'Fm':0.0,
    'Md':0.0,
    'No':0.0,
    'Lr':0.0,
    'Rf':0.0,
    'Db':0.0,
    'Sg':0.0,
    'Bh':0.0,
    'Hs':0.0,
    'Mt':0.0,
}