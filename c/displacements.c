#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>


// Cross compatibility mingw - gcc
#if !defined(_WIN32)

#define _Dcomplex double _Complex

_Dcomplex _Cbuild(double real, double imag){
    return real + imag* 1.j;
}
#endif

//  Functions declaration
static double   **matrix_multiplication (double **a, double **b, int n, int l, int m);

static int        TwotoOne              (int Row, int Column, int NumColumns);
static double   **pymatrix_to_c_array_real   (PyArrayObject *array);

static _Dcomplex  **pymatrix_to_c_array_complex   (PyArrayObject *array);

static double   **matrix_inverse ( double ** a ,int n);
static double     Determinant(double  **a,int n);
static double   **CoFactor(double  **a,int n);
static PyObject* atomic_displacements(PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring[] =
    "atomic_displacements(cell, trajectory, positions )";

static PyMethodDef extension_funcs[] = {
    {"atomic_displacements", (PyCFunction)atomic_displacements, METH_VARARGS|METH_KEYWORDS, function_docstring},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "displacements",
  "atomic displacements module",
  -1,
  extension_funcs,
  NULL,
  NULL,
  NULL,
  NULL,
};
#endif


static PyObject *
moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("displacements",
        extension_funcs, "This is a module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initdisplacements(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_displacements(void)
    {
        import_array();
        return moduleinit();
    }

#endif



static PyObject *atomic_displacements(PyObject *self, PyObject *arg, PyObject *keywords) {


//  Interface with python
    PyObject *Cell_obj, *Trajectory_obj, *Positions_obj;


    static char *kwlist[] = {"trajectory", "positions", "cell", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOO", kwlist,  &Trajectory_obj, &Positions_obj, &Cell_obj))  return NULL;

    PyObject *Trajectory_array = PyArray_FROM_OTF(Trajectory_obj, NPY_CDOUBLE, NPY_IN_ARRAY);
    PyObject *Positions_array = PyArray_FROM_OTF(Positions_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *Cell_array = PyArray_FROM_OTF(Cell_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (Cell_array == NULL || Trajectory_array == NULL) {
         Py_XDECREF(Cell_array);
         Py_XDECREF(Trajectory_array);
         Py_XDECREF(Positions_array);
         return NULL;
    }


//  double _Complex *Cell           = (double _Complex*)PyArray_DATA(Cell_array);
    _Dcomplex *Trajectory     = (_Dcomplex*)PyArray_DATA(Trajectory_array);
    double *Positions               = (double*)PyArray_DATA(Positions_array);

    int NumberOfData       = (int)PyArray_DIM(Trajectory_array, 0);
    int NumberOfDimensions = (int)PyArray_DIM(Cell_array, 0);

//  Create new Numpy array to store the result
    _Dcomplex **Displacement;
    PyArrayObject *Displacement_object;

    npy_intp dims[]={NumberOfData, NumberOfDimensions};
    Displacement_object=(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_CDOUBLE);
    Displacement=pymatrix_to_c_array_complex( Displacement_object);


//  Create a pointer array from cell matrix (to be improved)
    double  **Cell_c = pymatrix_to_c_array_real((PyArrayObject *) Cell_array);

/*
	printf("\nCell Matrix");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_c[i][j]);
	}
	printf("\n\n");

*/

//	Matrix inverse
	double  **Cell_i = matrix_inverse(Cell_c, NumberOfDimensions);

/*
	printf("\nMatrix Inverse");
	for(int i = 0 ;i < NumberOfDimensions ; i++){
		printf("\n");
		for(int j = 0; j < NumberOfDimensions; j++) printf("%f\t",Cell_i[i][j]);
	}
	printf("\n\n");
*/

	double ** Difference = malloc(NumberOfDimensions * sizeof(double *));
    for (int i = 0; i < NumberOfDimensions; i++) Difference[i] = (double *) malloc( sizeof(double ));

    for (int i = 0; i < NumberOfData; i++) {

        for (int k = 0; k < NumberOfDimensions; k++) {
            Difference[k][0] = creal(Trajectory[TwotoOne(i, k, NumberOfDimensions)]) - Positions[k];
        }

        double ** DifferenceMatrix = matrix_multiplication(Cell_i, Difference, NumberOfDimensions, NumberOfDimensions, 1);

        for (int k = 0; k < NumberOfDimensions; k++) {
                DifferenceMatrix[k][0] = round(DifferenceMatrix[k][0]);
        }

        double ** PeriodicDisplacement = matrix_multiplication(Cell_c, DifferenceMatrix, NumberOfDimensions, NumberOfDimensions, 1);

        for (int k = 0; k < NumberOfDimensions; k++) {
            Displacement[i][k] = _Cbuild(creal(Trajectory[TwotoOne(i, k, NumberOfDimensions)]) - PeriodicDisplacement[k][0] - Positions[k],0);
		}

        //Free memory
        for (int k=0 ; k<NumberOfDimensions; k++) free(DifferenceMatrix[k]); free(DifferenceMatrix);
        for (int k=0 ; k<NumberOfDimensions; k++) free(PeriodicDisplacement[k]); free(PeriodicDisplacement);

    }
    //Free memory
    for (int k=0 ; k<NumberOfDimensions; k++) free(Difference[k]); free(Difference);
    for (int k=0 ; k<NumberOfDimensions; k++) free(Cell_i[k]); free(Cell_i);
 	free(Cell_c);

    // Free python memory
    Py_DECREF(Trajectory_array);
    Py_DECREF(Positions_array);
    Py_DECREF(Cell_array);

//  Returning python array
    return(PyArray_Return(Displacement_object));
};



static _Dcomplex **pymatrix_to_c_array_complex(PyArrayObject *array)  {

      long n=(*array).dimensions[0];
      long m=(*array).dimensions[1];
      _Dcomplex ** c = malloc(n*sizeof(_Dcomplex));

      _Dcomplex *a = (_Dcomplex *) (*array).data;  /* pointer to array data as double _Complex */
      for ( int i=0; i<n; i++)  {
          c[i]=a+i*m;
      }

      return c;
};


static double  **pymatrix_to_c_array_real(PyArrayObject *array)  {

      long n=(*array).dimensions[0];
      long m=(*array).dimensions[1];
      //PyObject *transpose_array = PyArray_Transpose(array, dims);

      double  ** c = malloc(n*sizeof(double));
      double  *a = (double  *) (*array).data;

      for ( int i=0; i<n; i++)  {
          double  *b = malloc(m*sizeof(double));
          for ( int j=0; j<m; j++)  {
              b[j] = a[i+n*j];
          }
          c[i]=b;
      }

      //for ( int i=0; i<n; i++)  {
      //    c[i]=b+i*m;
      //}


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
         m = malloc((n-1)*sizeof(double *));
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

//   Find the co factor matrix of a square matrix
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

//  Calculate the matrix multiplication
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


static int TwotoOne(int Row, int Column, int NumColumns) {
	return Row*NumColumns + Column;
};
