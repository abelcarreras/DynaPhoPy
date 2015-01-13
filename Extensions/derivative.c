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
static PyObject* method1 (PyObject* self, PyObject *arg) {

//  Declaring basic variables (default)
	int Order = 1;

//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj, *Time_obj;

    if (!PyArg_ParseTuple(arg, "OOO|i", &Cell_obj, &Trajectory_obj, &Time_obj, &Order))  return NULL;

    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_CDOUBLE, NPY_IN_ARRAY);
    PyObject *Time_array = PyArray_FROM_OTF(Time_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL || Time_array == NULL) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         Py_XDECREF(Time_array);
         return NULL;
    }

//    double _Complex *Cell           = (double _Complex*)PyArray_DATA(Cell_array);
    double  *Time           = (double *)PyArray_DATA(Time_array);
    double _Complex *Trajectory     = (double _Complex*)PyArray_DATA(Trajectory_array);
    int NumberOfData       = (int)PyArray_DIM(Trajectory_array, 0);
    int NumberOfDimensions = (int)PyArray_DIM(Cell_array, 0);


//  Create new Numpy array to store the result

    double _Complex **Derivative;
    PyArrayObject *Derivative_object;

    int dims[2]={NumberOfData,NumberOfDimensions};

    Derivative_object=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_CDOUBLE);
    Derivative=pymatrix_to_c_array( Derivative_object);


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
*/


//  Pointers definition
	double _Complex* Point_initial        = malloc(NumberOfDimensions*sizeof(double _Complex));
	double _Complex* Point_final          = malloc(NumberOfDimensions*sizeof(double _Complex));

	double ** Separacio           = malloc(NumberOfDimensions*sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Separacio[k] = (double *) malloc(sizeof(double ));

	double ** Point_diff          = malloc(NumberOfDimensions*sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Point_diff[k] = (double *) malloc(sizeof(double ));


//	Derivation algorithm
	for (int i=Order; i<NumberOfData-Order; i++) {

		for (int k = 0; k < NumberOfDimensions; k++) {
			Point_initial[k]    = Trajectory[TwotoOne(i-Order, k, NumberOfDimensions)];
			Point_final  [k]    = Trajectory[TwotoOne(i+Order, k, NumberOfDimensions)];
			Point_diff   [k][0] = (Point_final[k] - Point_initial[k]) / 0.5;
		}

//		printf("PointIni: %i %f %f %f\n",i,Point_initial[0],Point_initial[1],Point_initial[2]);
//		printf("PointFin: %i %f %f %f\n",i,Point_final[0],Point_final[1],Point_final[2]);
//		printf("Pointdif: %i %f %f %f\n",i,Point_diff[0][0],Point_diff[1][0],Point_diff[2][0]);

		Separacio = matrix_multiplication(Cell_i, Point_diff, NumberOfDimensions, NumberOfDimensions, 1);
		for (int k = 0; k < NumberOfDimensions; k++) Separacio[k][0] = (double )(int)Separacio[k][0];


//		for (int k =0; k < NumberOfDimensions; k++) Separacio[0][k]= (double _Complex)(int)(Diferencia[k][0] / NormalizedCellVector[k]);
//		printf("Sep: %f %f %f\n",Separacio[0][0],Separacio[1][0],Separacio[2][0]);

		double  ** SeparacioProjectada = matrix_multiplication(Cell_c, Separacio, NumberOfDimensions, NumberOfDimensions, 1);
//		printf("SepProj: %f %f %f\n",SeparacioProjectada[0][0],SeparacioProjectada[1][0],SeparacioProjectada[2][0]);

		for (int k = 0; k < NumberOfDimensions; k++) Point_final[k]= Point_final[k]-SeparacioProjectada[k][0];
//		printf("Proper: %f %f %f\n",Point_final[0],Point_final[1],Point_final[2]);

		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[i][j] = (Point_final[j]-Point_initial[j])/ (Time[i+Order]-Time[i-Order]);
		}
	}



//  Side limits extrapolation
	for (int k = Order; k > 0; k--) {
		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[k-1][j] = ((Derivative[2+k-1][j] - Derivative[1+k-1][j]) / (Time[2+k-1] - Time[1+k-1])) * (Time[0+k-1] - Time[1+k-1]) + Derivative[1+k-1][j];
			Derivative[NumberOfData-k][j] = ((Derivative[NumberOfData-2-k][j] - Derivative[NumberOfData-1-k][j]) / (Time[NumberOfData-2-k] - Time[NumberOfData-1-k]))
			*(Time[NumberOfData-k] - Time[NumberOfData-1-k]) + Derivative[NumberOfData-1-k][j];
		}
	 }


//  Returning python array
    return(PyArray_Return(Derivative_object));

};


//  Derivate calculation (centered differencing) [real]
static PyObject* method2 (PyObject* self, PyObject *arg) {

//  Declaring basic variables (default)
	int Order = 1;

//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj, *Time_obj;

    if (!PyArg_ParseTuple(arg, "OOO|i", &Cell_obj, &Trajectory_obj, &Time_obj, &Order))  return NULL;

    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Time_array = PyArray_FROM_OTF(Time_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL || Time_array == NULL) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         Py_XDECREF(Time_array);
         return NULL;
    }

//    double  *Cell           = (double *)PyArray_DATA(Cell_array);
    double  *Time           = (double *)PyArray_DATA(Time_array);
    double  *Trajectory     = (double *)PyArray_DATA(Trajectory_array);
    int NumberOfData       = (int)PyArray_DIM(Trajectory_array, 0);
    int NumberOfDimensions = (int)PyArray_DIM(Cell_array, 0);


//    printf("Cell: %f %f %f %f %f %f\n", Cell[0],Cell[1],Cell[2],Cell[3],Cell[4],Cell[5]);
//    printf("Time: %f %f %f %f", Time[0],Time[1],Time[2],Time[3]);

//  Create new Numpy array to store the result

    double  **Derivative;
    PyArrayObject *Derivative_object;

    int dims[2]={NumberOfData,NumberOfDimensions};

    Derivative_object=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    Derivative=pymatrix_to_c_array_real(Derivative_object);


//  Create a pointer array for cell matrix (to be improved)
    double  **Cell_c = pymatrix_to_c_array_real((PyArrayObject *)Cell_array);

/*
	printf("\nCell Matrix");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_c[i][j]);
	}

	printf("\n\n");

*/

//	Matrix inversion
//	double  **Cell_i = matrix_inverse_3x3(Cell_c);
	double  **Cell_i = matrix_inverse(Cell_c,NumberOfDimensions);

/*
	printf("\nMatrix Inverse");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_i[i][j]);
	}
*/


//  Pointers definition
	double * Point_initial        = malloc(NumberOfDimensions*sizeof(double ));
	double * Point_final          = malloc(NumberOfDimensions*sizeof(double ));

	double ** Separacio           = malloc(NumberOfDimensions*sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Separacio[k] = (double *) malloc(sizeof(double ));

	double ** Point_diff          = malloc(NumberOfDimensions*sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Point_diff[k] = (double *) malloc(sizeof(double ));


//	Derivation algorithm
	for (int i=Order; i<NumberOfData-Order; i++) {

		for (int k = 0; k < NumberOfDimensions; k++) {
			Point_initial[k]    = Trajectory[TwotoOne(i-Order, k, NumberOfDimensions)];
			Point_final  [k]    = Trajectory[TwotoOne(i+Order, k, NumberOfDimensions)];
			Point_diff   [k][0] = (Point_final[k] - Point_initial[k]) / 0.5;
		}

//		printf("PointIni: %i %f %f %f\n",i,Point_initial[0],Point_initial[1],Point_initial[2]);
//		printf("PointFin: %i %f %f %f\n",i,Point_final[0],Point_final[1],Point_final[2]);
//		printf("Pointdif: %i %f %f %f\n",i,Point_diff[0][0],Point_diff[1][0],Point_diff[2][0]);

		Separacio = matrix_multiplication(Cell_i, Point_diff, NumberOfDimensions, NumberOfDimensions, 1);
		for (int k = 0; k < NumberOfDimensions; k++) Separacio[k][0] = (double )(int)Separacio[k][0];


//		for (int k =0; k < NumberOfDimensions; k++) Separacio[0][k]= (double )(int)(Diferencia[k][0] / NormalizedCellVector[k]);
//		printf("Sep: %f %f %f\n",Separacio[0][0],Separacio[1][0],Separacio[2][0]);

		double  ** SeparacioProjectada = matrix_multiplication(Cell_c, Separacio, NumberOfDimensions, NumberOfDimensions, 1);
//		printf("SepProj: %f %f %f\n",SeparacioProjectada[0][0],SeparacioProjectada[1][0],SeparacioProjectada[2][0]);

		for (int k = 0; k < NumberOfDimensions; k++) Point_final[k]= Point_final[k]-SeparacioProjectada[k][0];
//		printf("Proper: %f %f %f\n",Point_final[0],Point_final[1],Point_final[2]);

		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[i][j] = (Point_final[j]-Point_initial[j])/ (Time[i+Order]-Time[i-Order]);
		}
	}



//  Side limits extrapolation
	for (int k = Order; k > 0; k--) {
		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[k-1][j] = ((Derivative[2+k-1][j] - Derivative[1+k-1][j]) / (Time[2+k-1] - Time[1+k-1])) * (Time[0+k-1] - Time[1+k-1]) + Derivative[1+k-1][j];
			Derivative[NumberOfData-k][j] = ((Derivative[NumberOfData-2-k][j] - Derivative[NumberOfData-1-k][j]) / (Time[NumberOfData-2-k] - Time[NumberOfData-1-k]))
			*(Time[NumberOfData-k] - Time[NumberOfData-1-k]) + Derivative[NumberOfData-1-k][j];
		}
	 }


//  Returning python array
    return(PyArray_Return(Derivative_object));

};


//  Derivate calculation (centered differencing with arbitrary order of precision)
static PyObject* method3 (PyObject* self, PyObject *arg, PyObject *keywords) {

//  Declaring basic variables (default)
	int Order = 2;
    double  TimeStep;

//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj;
    static char *kwlist[] = {"cell", "trajectory", "time_step", "precision_order", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOd|i", kwlist, &Cell_obj, &Trajectory_obj, &TimeStep, &Order))  return NULL;

    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_CDOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL ) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         return NULL;
    }

//    double _Complex *Cell           = (double _Complex*)PyArray_DATA(Cell_array);
    double _Complex *Trajectory = (double _Complex*)PyArray_DATA(Trajectory_array);
    int    NumberOfData         = (int)PyArray_DIM(Trajectory_array, 0);
    int    NumberOfDimensions   = (int)PyArray_DIM(Cell_array, 0);


//  Create new Numpy array to store the result
    double _Complex **Derivative;
    PyArrayObject *Derivative_object;

    int dims[2]={NumberOfData,NumberOfDimensions};

    Derivative_object=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_CDOUBLE);
    Derivative=pymatrix_to_c_array( Derivative_object);


//  Create a pointer array for cell matrix (to be improved)
    double  **Cell_c = pymatrix_to_c_array_real((PyArrayObject *) Cell_array);


//	Matrix inversion
	double  **Cell_i = matrix_inverse(Cell_c,NumberOfDimensions);


//  Pointers definition
	double _Complex* Point_initial        = malloc(NumberOfDimensions*sizeof(double _Complex));
	double _Complex* Point_final          = malloc(NumberOfDimensions*sizeof(double _Complex));

    double * Coefficients                 = malloc((Order+1)*sizeof(double));

	double ** Separation           = malloc(sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Separation[k] = (double *) malloc(sizeof(double ));


	double ** ProjectedSeparation           = malloc(sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) ProjectedSeparation[k] = (double *) malloc(sizeof(double ));


	double ** Point_diff          = malloc(sizeof(double *));
	for (int k = 0; k < NumberOfDimensions; k++) Point_diff[k] = (double *) malloc(sizeof(double ));



//	Derivation algorithm
    Coefficients = FiniteDifferenceCoefficients(1, Order);

	for (int i=Order; i<(NumberOfData-Order); i++) {

		for (int k = 0; k < NumberOfDimensions; k++) {
			Point_initial[k] = Trajectory[TwotoOne(i, k, NumberOfDimensions)];
			Derivative[i][k] = 0;
		}

		for (int j = 0; j <= Order; j++) {


            for (int k = 0; k < NumberOfDimensions; k++) {
                Point_final  [k]    = Trajectory[TwotoOne(i+Position(j), k, NumberOfDimensions)];
                Point_diff   [k][0] = (Point_final[k] - Point_initial[k]) / 0.5;
            }
            matrix_multiplication2(Cell_i, Point_diff, Separation, NumberOfDimensions, NumberOfDimensions, 1);

            for (int k = 0; k < NumberOfDimensions; k++) Separation[k][0] = (double)(int)Separation[k][0];
            matrix_multiplication2(Cell_c, Separation, ProjectedSeparation, NumberOfDimensions, NumberOfDimensions, 1);

            for (int k = 0; k < NumberOfDimensions; k++) Point_final[k]= Point_final[k]-ProjectedSeparation[k][0];

        	for (int k = 0; k < NumberOfDimensions; k++) Derivative[i][k] += (Point_final[k]*Coefficients[j])/ pow(TimeStep,1);

        }

	}

//  Side limits extrapolation
	for (int k = Order; k > 0; k--) {
		for (int j = 0; j < NumberOfDimensions; j++) {
			Derivative[k-1][j] = 2.0 * Derivative[1+k-1][j] - Derivative[2+k-1][j];
			Derivative[NumberOfData-k][j] = 2.0 * Derivative[NumberOfData-1-k][j] - Derivative[NumberOfData-2-k][j] ;
		}
	 }


//  Returning python array
    return(PyArray_Return(Derivative_object));

};


//  ---------------   Support functions ----------------  //


/* ==== Create Carray from PyArray ======================
     Assumes PyArray is contiguous in memory.
     Memory is allocated!                                    */

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

//	Calculate the matrix inversion of a 3x3 matrix
/*
static double  **matrix_inverse_3x3 ( double ** a ){


	double ** b = malloc(3*sizeof(double *));
    for (int i = 0; i < 3; i++) b[i] = (double *) malloc(3*sizeof(double ));


	float determinant=0;

	for(int i=0;i<3;i++)
		determinant = determinant + (a[i][0]*(a[(i+1)%3][1]*a[(i+2)%3][2] - a[(i+2)%3][1]*a[(i+1)%3][2]));

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++) {
			b[i][j]=((a[(j+1)%3][(i+1)%3] * a[(j+2)%3][(i+2)%3]) - (a[(j+2)%3][(i+1)%3]*a[(j+1)%3][(i+2)%3]))/ determinant;
		}
	}

	return b;

};


*/

static double  **matrix_inverse ( double ** a ,int n){


	double ** b = malloc(n*sizeof(double *));
    for (int i = 0; i < n; i++) b[i] = (double *) malloc(n*sizeof(double ));

//	double _Complex** cof = malloc(n*sizeof(double _Complex*));
//    for (int i = 0; i < n; i++) cof[i] = (double _Complex*) malloc(n*sizeof(double _Complex));

    double  **cof = CoFactor(a,n);

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++) {
			b[i][j]=cof[j][i]/Determinant(a,n);
		}
	}

	return b;

}
/*
   Recursive definition of determinate using expansion by minors.
*/


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
}

/*
   Find the cofactor matrix of a square matrix
*/
static double ** CoFactor(double  **a,int n)
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

}


//  Calculate the matrix multiplication
static double  **matrix_multiplication ( double   **a, double   **b, int n, int l, int m ){

	double ** c = malloc(n*sizeof(double *));
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

static void  matrix_multiplication2 (double  **a, double  **b, double  **c, int n, int l, int m){

	for (int i = 0 ; i< n ;i++) {
		for (int j  = 0; j<m ; j++) {
			c[i][j] = 0;
			for (int k = 0; k<l; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}


static int TwotoOne(int Row, int Column, int NumColumns) {
	return Row*NumColumns + Column;
};

static int Position(int i) {
    return (i+1)/2*pow(-1,i+1);
    }


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
}


//  --------------- Interface functions ---------------- //

static char extension_docs_method1[] =
    "derivative(cell, trajectory, time, order=1 )\n\n Calculation of the derivative [centered differencing]\n";

static char extension_docs_method2[] =
    "derivative_real(cell, trajectory, time, order=1 )\n\n Calculation of the derivative [centered differencing] [real]\n";

static char extension_docs_method3[] =
    "derivative_real(cell, trajectory, time, precision_order=2 )\n\n Calculation of the derivative [centered differencing] [Any order]\n";

static PyMethodDef extension_funcs[] = {
    {"derivative", (PyCFunction)method1, METH_VARARGS, extension_docs_method1},
    {"derivative_real", (PyCFunction)method2, METH_VARARGS, extension_docs_method2},
    {"derivative_general", (PyCFunction)method3, METH_VARARGS|METH_KEYWORDS, extension_docs_method3},
    {NULL}
};

void initderivative(void)
{
//  Importing numpy array types
    import_array();

    Py_InitModule3("derivative", extension_funcs,
                   "Calculate the derivative of periodic cell");
};
