#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>

//#include "/Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/Python.framework/Versions/2.6/Extras/lib/python/numpy/core/include/numpy/arrayobject.h"

// Functions declaration
double **matrix_inverse_3x3 ( double **a );
double **matrix_multiplication ( double **a, double **b ,int n, int l, int m);
double vector_norm (double *vector, int length);
int TwotoOne(int Row, int Column, int NumColumns);
double **pymatrix_to_c_array(PyArrayObject *array);


// Derivate calculation (centered differencing)
static PyObject* method1 (PyObject* self, PyObject *arg) {

//  Declaring basic variables
	int Order = 1;

//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj, *Time_obj;

    if (!PyArg_ParseTuple(arg, "OOO|i", &Cell_obj,&Trajectory_obj,&Time_obj,&Order))  return NULL;

    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Time_array = PyArray_FROM_OTF(Time_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL || Time_array == NULL) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         Py_XDECREF(Time_array);
         return NULL;
    }

//    double *Cell           = (double*)PyArray_DATA(Cell_array);
    double *Time           = (double*)PyArray_DATA(Time_array);
    double *Trajectory     = (double*)PyArray_DATA(Trajectory_array);
    int NumberOfData       = (int)PyArray_DIM(Trajectory_array, 0);
    int NumberOfDimensions = (int)PyArray_DIM(Cell_array, 0);


//    printf("Cell: %f %f %f %f %f %f\n", Cell[0],Cell[1],Cell[2],Cell[3],Cell[4],Cell[5]);
//    printf("Time: %f %f %f %f", Time[0],Time[1],Time[2],Time[3]);

//  Create new Numpy array to store the result

    double **Derivative;
    PyArrayObject *Derivative_object;

    int dims[2]={NumberOfData,NumberOfDimensions};

    Derivative_object=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    Derivative=pymatrix_to_c_array(Derivative_object);


//  Create a pointer array for cell matrix (to be improved)
    double **Cell_c = pymatrix_to_c_array(Cell_array);

//	Matrix inversion
	double **Cell_i = matrix_inverse_3x3(Cell_c);

/*	printf("\nMatrix Inverse");
	for(int i = 0 ;i < 3 ; i++){
		printf("\n");
		for(int j = 0; j < 3; j++) printf("%f\t",Cell_i[i][j]);
	}

	printf("\n\n");
*/

//  Pointers definition
	double* Point_initial        = malloc(NumberOfDimensions*sizeof(double));
	double* Point_final          = malloc(NumberOfDimensions*sizeof(double));

	double** Separacio           = malloc(NumberOfDimensions*sizeof(double*));
	for (int k = 0; k < NumberOfDimensions; k++) Separacio[k] = (double*) malloc(sizeof(double));

	double** Point_diff          = malloc(NumberOfDimensions*sizeof(double*));
	for (int k = 0; k < NumberOfDimensions; k++) Point_diff[k] = (double*) malloc(sizeof(double));


//	Derivation algorithm
	for (int i=Order; i<NumberOfData-Order; i++) {

		for (int k = 0; k < NumberOfDimensions; k++) {
			Point_initial[k]    = Trajectory[TwotoOne(i-Order, k, NumberOfDimensions)];
			Point_final  [k]    = Trajectory[TwotoOne(i+Order, k, NumberOfDimensions)];
			Point_diff   [k][0] = (Point_final[k] - Point_initial[k])/0.5;
		}

//		printf("PointIni: %i %f %f %f\n",i,Point_initial[0],Point_initial[1],Point_initial[2]);
//		printf("PointFin: %i %f %f %f\n",i,Point_final[0],Point_final[1],Point_final[2]);
//		printf("Pointdif: %i %f %f %f\n",i,Point_diff[0][0],Point_diff[1][0],Point_diff[2][0]);

		Separacio = matrix_multiplication(Cell_i, Point_diff, 3, 3, 1);
		for (int k = 0; k < NumberOfDimensions; k++) Separacio[k][0] = (double)(int)Separacio[k][0];


//		for (int k =0; k < NumberOfDimensions; k++) Separacio[0][k]= (double)(int)(Diferencia[k][0] / NormalizedCellVector[k]);
//		printf("Sep: %f %f %f\n",Separacio[0][0],Separacio[1][0],Separacio[2][0]);

		double ** SeparacioProjectada = matrix_multiplication(Cell_c, Separacio, 3, 3, 1);
//		printf("SepProj: %f %f %f\n",SeparacioProjectada[0][0],SeparacioProjectada[1][0],SeparacioProjectada[2][0]);

		for (int k =0; k < NumberOfDimensions; k++) Point_final[k]= Point_final[k]-SeparacioProjectada[k][0];
//		printf("Proper: %f %f %f\n",Point_final[0],Point_final[1],Point_final[2]);

		for (int j=0; j<NumberOfDimensions; j++) {
			Derivative[i][j] = (Point_final[j]-Point_initial[j])/ (Time[i+Order]-Time[i-Order]);
		}
	}

//  Side limits extrapolation
	for (int k=Order; k>0; k--) {
		for (int j=0; j<NumberOfDimensions; j++) {
			Derivative[0+k-1][j]=((Derivative[2+k-1][j]-Derivative[1+k-1][j])/(Time[2+k-1]-Time[1+k-1]))*(Time[0+k-1]-Time[1+k-1])+Derivative[1+k-1][j];
			Derivative[NumberOfData-k][j]=((Derivative[NumberOfData-2-k][j]-Derivative[NumberOfData-1-k][j])/(Time[NumberOfData-2-k]-Time[NumberOfData-1-k]))*(Time[NumberOfData-k]-Time[NumberOfData-1-k])+Derivative[NumberOfData-1-k][j];
		}
	 }


//  Returning python array
    return(PyArray_Return(Derivative_object));

}

//  ---------------   Support functions ----------------  //


/* ==== Create Carray from PyArray ======================
     Assumes PyArray is contiguous in memory.
     Memory is allocated!                                    */

double **pymatrix_to_c_array(PyArrayObject *array)  {

      int n=(*array).dimensions[0];
      int m=(*array).dimensions[1];
      double ** c = malloc(n*sizeof(double));

      double *a = (double *) (*array).data;  /* pointer to array data as double */
      for ( int i=0; i<n; i++)  {
          c[i]=a+i*m;
      }

      return c;
}

//	Calculate the matrix inversion of a 3x3 matrix
double **matrix_inverse_3x3 ( double** a ){

	double** b = malloc(3*sizeof(double*));
    for (int i = 0; i < 3; i++) b[i] = (double*) malloc(3*sizeof(double));


	float determinant=0;

	for(int i=0;i<3;i++)
		determinant = determinant + (a[i][0]*(a[(i+1)%3][1]*a[(i+2)%3][2] - a[(i+2)%3][1]*a[(i+1)%3][2]));

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++) {
			b[i][j]=((a[(j+1)%3][(i+1)%3] * a[(j+2)%3][(i+2)%3]) - (a[(j+2)%3][(i+1)%3]*a[(j+1)%3][(i+2)%3]))/ determinant;
		}
	}

	return b;

}

//  Calculate the matrix multiplication
double **matrix_multiplication ( double  **a, double  **b, int n, int l, int m ){

	double** c = malloc(n*sizeof(double*));
    for (int i = 0; i < n; i++)
		c[i] = (double*) malloc(m*sizeof(double));

	for (int i = 0 ; i< n ;i++) {
		for (int j  = 0; j<m ; j++) {
			c[i][j] = 0;
			for (int k = 0; k<l; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return c;
}

//	Calculate the norm of a 1D vector
double vector_norm (double *vector, int length){
	double norm = 0;
		for (int i=0; i<length; i++) {
			norm += pow(vector[i],2);
		}
	return sqrt(norm);
}

int TwotoOne(int Row, int Column, int NumColumns) {
	return Row*NumColumns + Column;
}



//  --------------- Interface functions ---------------- //

static char extension_docs1[] =
    "derivative1( ): Calculation of the derivative (centered differencing)\n";


static PyMethodDef extension_funcs[] = {
    {"derivative1", (PyCFunction)method1, METH_VARARGS, extension_docs1},

    {NULL}
};

void initderivative(void)
{
//  Importing numpy array types
    import_array();

    Py_InitModule3("derivative", extension_funcs,
                   "Calculate the derivative of periodic cell");
}
