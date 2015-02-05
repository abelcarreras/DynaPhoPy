from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_phonon_modes(structure, eigenvectors,
                      super_cell=(1, 1, 1),
                      draw_primitive=False,
                      vectors_scale=3):

    atom_type = structure.get_atom_type_index(super_cell=super_cell)
    positions = structure.get_positions(super_cell=super_cell)
    cell = structure.get_cell()

    if draw_primitive:
        cell = structure.get_primitive_cell()

    for i_phonon in range(eigenvectors.shape[0]):

        fig = plt.figure(i_phonon+1)
        ax = fig.add_subplot(111, projection='3d')

        #Atom positions
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o', markersize=30, color='green', alpha=0.8)

        #Cell frame
        for i in range(3):
            cell_side = [(0,cell[i,0]),(0, cell[i,1]),(0,cell[i,2])]
            ax.plot3D(*cell_side, color="b")
            for j in range(3):
                if i != j:
                    cell_side = [(cell[i,l],
                                  cell[i,l]+cell[j,l]) for l in range(3)]

                    ax.plot3D(*cell_side, color="b")
                    for k in range(3):
                        if k != i and k != j and j > i:
                            cell_side = [(cell[i,l]+cell[j,l],
                                          cell[i,l]+cell[j,l]+cell[k,l]) for l in range(3)]

                            ax.plot3D(*cell_side, color="b")


        #Atom positions
        for i, position in enumerate(positions):
            vector = np.array(eigenvectors[i_phonon, atom_type[i], :].real) * vectors_scale
            a = Arrow3D([position[0],position[0]+ vector[0]], [position[1], position[1]+vector[1]],
                        [position[2], position[2]+ vector[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
            ax.add_artist(a)

 #       ax.set_axis_off()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title('Phonon '+str(i_phonon))
        plt.axis("equal")
    plt.show()

    return