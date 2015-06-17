from matplotlib import pyplot as plt
import numpy as np

def coordinates_distribution(coordinates, cell):

    cell = np.array([[1, 0.5],[0.5, 1]])
    position = np.array([0.0, 0.0])
    coordinate = np.array([1.0, 0])



    IniSep = coordinate-position
    print(IniSep)

    print("Initial:",np.linalg.norm(IniSep))


    Separacio = np.array(np.dot(np.linalg.inv(cell),(coordinate-position)),dtype=int)
    Separacio = np.around(np.dot(np.linalg.inv(cell), (coordinate - position)), decimals=0)
    print(IniSep-np.dot(Separacio, cell))

    print("final:",np.linalg.norm(IniSep-np.dot(Separacio, cell)))



    return 0

"""
		Separacio = matrix_multiplication(Cell_i, Point_diff);

			for (int k = 0; k < NumberOfDimensions; k++)
            Separacio[k][0] = (double )(int)Separacio[k][0];


//		for (int k =0; k < NumberOfDimensions; k++)
            Separacio[0][k]= (double _Complex)(int)(Diferencia[k][0] / NormalizedCellVector[k]);

		double  ** SeparacioProjectada = matrix_multiplication(Cell_c, Separacio, NumberOfDimensions, NumberOfDimensions, 1);
//		printf("SepProj: %f %f %f\n",SeparacioProjectada[0][0],SeparacioProjectada[1][0],SeparacioProjectada[2][0]);

		for (int k = 0; k < NumberOfDimensions; k++) Point_final[k]= Point_final[k]-SeparacioProjectada[k][0];
//		printf("Proper: %f %f %f\n",Point_final[0],Point_final[1],Point_final[2]);

		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[i][j] = (Point_final[j]-Point_initial[j])/ (Time[i+Order]-Time[i-Order]);
		}
	}
"""


if __name__ == '__main__':
    coordinates_distribution(2,3)
