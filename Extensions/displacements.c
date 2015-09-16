#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>

#define min(a,b) ((a) < (b) ? (a) : (b))

//  Functions declaration
//static double   **matrix_inverse_3x3    (double  **a);
static double   **matrix_multiplication (double  **a, double  **b, int n, int l, int m);
static void       matrix_multiplication2 (double  **a, double  **b, double  **c, int n, int l, int m);

static int       TwotoOne              (int Row, int Column, int NumColumns);
static double   **pymatrix_to_c_array_real   (PyArrayObject *array);

static double _Complex  **pymatrix_to_c_array   (PyArrayObject *array);

static double  **matrix_inverse ( double ** a ,int n);
static double  Determinant(double  **a,int n);
static double  ** CoFactor(double  **a,int n);

static double *FiniteDifferenceCoefficients(int DerivativeOrder, int PrecisionOrder);
static int Position(int i);

//  Derivate calculation (centered differencing)
static PyObject* relative_trajectory (PyObject* self, PyObject *arg) {

//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj, *Positions_obj;

    if (!PyArg_ParseTuple(arg, "OOO", &Cell_obj, &Trajectory_obj, &Positions_obj))  return NULL;

    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Positions_array = PyArray_FROM_OTF(Positions_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL || Positions_array == NULL) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         Py_XDECREF(Positions_array);
         return NULL;
    }

//    double _Complex *Cell           = (double _Complex*)PyArray_DATA(Cell_array);
    double  *Positions     = (double *)PyArray_DATA(Positions_array);
    double  *Trajectory    = (double *)PyArray_DATA(Trajectory_array);
    int NumberOfData       = (int)PyArray_DIM(Trajectory_array, 0);
    int NumberOfDimensions = (int)PyArray_DIM(Trajectory_array, 1);


//  Create new Numpy array to store the result

    double **NormalizedTrajectory;
    PyArrayObject *NormalizedTrajectory_object;

    int dims[2]={NumberOfData,NumberOfDimensions};

    NormalizedTrajectory_object=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    NormalizedTrajectory=pymatrix_to_c_array_real( NormalizedTrajectory_object);

//  Create a pointer array for cell matrix (to be improved)
    double  **Cell_c = pymatrix_to_c_array_real((PyArrayObject *) Cell_array);


/*
	printf("\nCell Matrix");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_c[i][j]);
	}
	printf("\n\n");

*/

//	Matrix inversion
//	double _Complex **Cell_i = matrix_inverse_3x3(Cell_c);
	double  **Cell_i = matrix_inverse(Cell_c,NumberOfDimensions);

/*
	printf("\nMatrix Inverse");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_i[i][j]);
	}
    printf("\n\n");
*/

    double ** Difference          = malloc(sizeof(double *));
    for (int k = 0; k < NumberOfDimensions; k++) Difference[k] = (double *) malloc(sizeof(double));


    double ** DifferenceMatrix          = malloc(sizeof(double *));
    for (int k = 0; k < NumberOfDimensions; k++) DifferenceMatrix[k] = (double *) malloc(sizeof(double));


 //   printf ("Dim:%d Data:%d\n", NumberOfDimensions, NumberOfData);

    # pragma omp parallel for default(shared), firstprivate(Difference, DifferenceMatrix), schedule(dynamic, 20000)
	for (int i=0; i<NumberOfData; i++) {


     		for (int k = 0; k < NumberOfDimensions; k++) Difference[k][0] = Trajectory[TwotoOne(i, k, NumberOfDimensions)] - Positions[k];

		    DifferenceMatrix = matrix_multiplication(Cell_i, Difference, NumberOfDimensions, NumberOfDimensions, 1);

           // for (int k = 0; k < NumberOfDimensions; k++) printf("%d %d %f\n",i,k,DifferenceMatrix[k][0]);


		    for (int k = 0; k < NumberOfDimensions; k++) DifferenceMatrix[k][0] = round(DifferenceMatrix[k][0]);

	//	    for (int k = 0; k < NumberOfDimensions; k++) printf("%f ",DifferenceMatrix[k][0]);
      //      printf("\n");

		    //for (int k = 0; k < NumberOfDimensions; k++) printf("%d %d %f\n",i,k,DifferenceMatrix[0][k]);

		    double  ** NormalizedVector = matrix_multiplication(Cell_c, DifferenceMatrix, NumberOfDimensions, NumberOfDimensions, 1);
//		    for (int k = 0; k < NumberOfDimensions; k++) printf("%d %d %f\n",i,k,NormalizedVector[k][0]);

         //   for (int k = 0; k < NumberOfDimensions; k++) NormalizedTrajectory[i][k] = Trajectory[TwotoOne(i, k, NumberOfDimensions)] - NormalizedVector[k][0] - Positions[k];
            for (int k = 0; k < NumberOfDimensions; k++) NormalizedTrajectory[i][k] = Trajectory[TwotoOne(i, k, NumberOfDimensions)] - NormalizedVector[k][0] - Positions[k];

    }

    free (Difference);
    free (DifferenceMatrix);
 /*

    for j in range(number_of_atoms):
        for i in range(0, normalized_trajectory.shape[0]):

            difference = normalized_trajectory[i, j, :] - position[j]

            difference_matrix = np.around(np.dot(np.linalg.inv(cell), difference), decimals=0)

            normalized_trajectory[i, j, :] -= np.dot(difference_matrix, cell) + position[j]

        progress_bar(float(j+1)/number_of_atoms)

    return normalized_trajectory

*/


//  Returning python array
    return(PyArray_Return(NormalizedTrajectory_object));

};


//  Derivate calculation (centered differencing) [real]

static double _Complex **pymatrix_to_c_array(PyArrayObject *array)  {

      int n=(*array).dimensions[0];
      int m=(*array).dimensions[1];
      double _Complex ** c = malloc(n*sizeof(double _Complex));

      double _Complex *a = (double _Complex *) (*array).data;  /* pointer to array data as double _Complex */
      for ( int i=0; i<n; i++)  {
          c[i]=a+i*m;
      }

      return c;
};


static double  **pymatrix_to_c_array_real(PyArrayObject *array)  {

      int n=(*array).dimensions[0];
      int m=(*array).dimensions[1];
      double  ** c = malloc(n*sizeof(double));

      double  *a = (double  *) (*array).data;  /* pointer to array data as double  */
      for ( int i=0; i<n; i++)  {
          c[i]=a+i*m;
      }

      return c;
};

static double  **matrix_inverse ( double ** a ,int n){


	double ** b = malloc(n*sizeof(double *));
    for (int i = 0; i < n; i++) b[i] = (double *) malloc(n*sizeof(double ));

    double  **cof = CoFactor(a,n);

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++) {
			b[i][j]=cof[j][i]/Determinant(a,n);
		}
	}

	return b;

};

//  Recursive definition of determinate using expansion by minors.

static double  Determinant(double  **a,int n)
{
   int i,j,j1,j2;
   double  det = 0;
   double  **m = NULL;

   if (n < 1) { /* Error */

   } else if (n == 1) { /* Shouldn't get used */
      det = a[0][0];
   } else if (n == 2) {
      det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
   } else {
      det = 0;
      for (j1=0;j1<n;j1++) {
         m = malloc((n-1)*sizeof(double  *));
         for (i=0;i<n-1;i++)
            m[i] = malloc((n-1)*sizeof(double ));
         for (i=1;i<n;i++) {
            j2 = 0;
            for (j=0;j<n;j++) {
               if (j == j1)
                  continue;
               m[i-1][j2] = a[i][j];
               j2++;
            }
         }
         det += pow(-1.0,j1+2.0) * a[0][j1] * Determinant(m,n-1);
         for (i=0;i<n-1;i++)
            free(m[i]);
         free(m);
      }
   }
   return(det);
};

//   Find the cofactor matrix of a square matrix

static double **CoFactor(double  **a,int n)
{
   int i,j,ii,jj,i1,j1;
   double  det;
   double  **c;

   c = malloc((n-1)*sizeof(double  *));
   for (i=0;i<n-1;i++)
     c[i] = malloc((n-1)*sizeof(double ));

	double ** b = malloc(n*sizeof(double *));
    for (int i = 0; i < n; i++) b[i] = (double *) malloc(n*sizeof(double ));


   for (j=0;j<n;j++) {
      for (i=0;i<n;i++) {

         /* Form the adjoint a_ij */
         i1 = 0;
         for (ii=0;ii<n;ii++) {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) {
               if (jj == j)
                  continue;
               c[i1][j1] = a[ii][jj];
               j1++;
            }
            i1++;
         }
         /* Calculate the determinate */
         det = Determinant(c,n-1);

         /* Fill in the elements of the cofactor */
         b[i][j] = pow(-1.0,i+j+2.0) * det;
      }
   }

   for (i=0;i<n-1;i++)
      free(c[i]);
   free(c);
   return b;

};

//  Calculate the matrix multiplication creating new memory allocation
static double  **matrix_multiplication ( double   **a, double   **b, int n, int l, int m ){

	double **c = malloc(n*sizeof(double *));
    for (int i = 0; i < n; i++)
		c[i] = (double *) malloc(m*sizeof(double ));

	for (int i = 0 ; i< n ;i++) {
		for (int j  = 0; j<m ; j++) {
			c[i][j] = 0;
			for (int k = 0; k<l; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return c;
};


// Alternative matrix multiplication function using preexisting allocated memory
static void  matrix_multiplication2 (double  **a, double  **b, double  **c, int n, int l, int m){

	for (int i = 0 ; i< n ;i++) {
		for (int j  = 0; j<m ; j++) {
			c[i][j] = 0;
			for (int k = 0; k<l; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
};


static int TwotoOne(int Row, int Column, int NumColumns) {
	return Row*NumColumns + Column;
};

static int Position(int i) {
    return (i+1)/2*pow(-1,i+1);
};


static double *FiniteDifferenceCoefficients(int M, int N) {
    double *df = malloc((N+1)*sizeof(double *));
    double d[M+1][N+1][N+1];
    double a[N+1];

    double x0;
    int c1, c2, c3;
    int m,n,v, i;

    for (i=0; i <= N ; i++) {
        a[i] = Position(i);
    }

    x0 = 0;
    d[0][0][0] = 1.0;
    c1 = 1;
    for ( n = 1; n<=N; n++) {
        c2 = 1;
        for ( v=0; v < n; v++){
            c3 = a[n] - a[v];
            c2 = c2 * c3;
            if (n < M)  d[n][n-1][v]=0.0;
            for ( m=0; m <= min(n,M); m++) {
                d[m][n][v] = ((a[n]-x0)*d[m][n-1][v]-m*d[m-1][n-1][v])/c3;
            }
        }
        for (m=0; m <= min(n,M); m++) {
            d[m][n][v] = c1*(m*d[m-1][n-1][n-1]-(a[n-1]-x0)*d[m][n-1][n-1])/c2;
        }
        c1 = c2;
    }

    for (i=0; i<=N ; i++) {
        df[i] = d[M][N][i];
    }

    return df;
};

//  --------------- Interface functions ---------------- //

static char extension_docs[] =
    "relative_trajectory(cell, trajectory, positions )\n";


static PyMethodDef extension_funcs[] = {
      {"relative_trajectory", (PyCFunction)relative_trajectory, METH_VARARGS, extension_docs},
      {NULL}
};

void initdisplacements(void)
{
//  Importing numpy array types
    import_array();

    Py_InitModule3("displacements", extension_funcs,
                   "Calculate the trajectory relative to atoms position");
};