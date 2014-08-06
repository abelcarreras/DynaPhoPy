#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>


#undef I


static PyObject* correlation1 (PyObject* self, PyObject *arg)
{

//  Declaring basic variables
    double  Frequency;
	int     Increment = 1;

//  Interface with python
    PyObject *VQ_obj, *Time_obj;

    if (!PyArg_ParseTuple(arg, "dOO|i", &Frequency, &VQ_obj, &Time_obj, &Increment))  return NULL;

    PyObject *VQ_array = PyArray_FROM_OTF(VQ_obj, NPY_CDOUBLE, NPY_IN_ARRAY);
    PyObject *Time_array = PyArray_FROM_OTF(Time_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (VQ_array == NULL || Time_array == NULL ) {
         Py_XDECREF(VQ_array);
         Py_XDECREF(Time_array);
         return NULL;
    }



    double _Complex  *VQ = (double _Complex*)PyArray_DATA(VQ_array);
    double *Time         = (double*)PyArray_DATA(Time_array);
    int     NumberOfData = (int)PyArray_DIM(VQ_array, 0);


//  Starting correlation calculation
	double _Complex Correl = 0;

	for (int i = 0; i< NumberOfData; i += Increment) {
		for (int j = 0; j< (NumberOfData-i-Increment); j++) {
			switch ('R') {
				case 'T': //	Trapezoid Integration
					Correl += (conj(VQ[j]) * VQ[j+i+Increment] * cexp(_Complex_I*Frequency                            * (Time[i+Increment] - Time[0]))
					       +   conj(VQ[j]) * VQ[j+i]           * cexp(_Complex_I*Frequency * (Time[i]-Time[0])) )/2.0 * (Time[i+Increment] - Time[i]);

					break;
				case 'R': //	Rectangular Integration
					Correl +=  conj(VQ[j]) * VQ[j+i]           * cexp(_Complex_I*Frequency * (Time[i]-Time[0]))       * (Time[i+Increment] - Time[i]);
					break;
				default:
				    puts ("No correlation method selected");
				    exit(0);
					break;
			}
		}
	}
    return Py_BuildValue ("d", creal(Correl)/NumberOfData);
};


static PyObject* correlation2 (PyObject* self, PyObject *arg )
{

//  Declaring basic variables
    double  Frequency;
    double  DTime;
 	int     Increment = 13;   //Default value for Increment

//  Interface with python
    PyObject *VQ_obj;

    if (!PyArg_ParseTuple(arg, "dOd|i", &Frequency, &VQ_obj, &DTime, &Increment))  return NULL;

    PyObject *VQ_array = PyArray_FROM_OTF(VQ_obj, NPY_CDOUBLE, NPY_IN_ARRAY);

    if (VQ_array == NULL ) {
         Py_XDECREF(VQ_array);
         return NULL;
    }

    double _Complex *VQ  = (double _Complex*)PyArray_DATA(VQ_array);
    int    NumberOfData = (int)PyArray_DIM(VQ_array, 0);


//     printf ("inc %f\n",DTime);
//  Starting correlation calculation
	double _Complex Correl = 0;
	for (int i = 0; i < NumberOfData; i += Increment) {
		for (int j = 0; j < (NumberOfData-i-Increment); j++) {
			Correl += conj(VQ[j]) * VQ[j+i] * cexp(_Complex_I*Frequency*(i*DTime));
		}
	}

    return Py_BuildValue("d", creal(Correl) * DTime/(NumberOfData/Increment));

};

static char extension_docs_1[] =
    "correlation ( ): Calculation of the correlation\n (No time restriction)";

static char extension_docs_2[] =
    "correlation2( ): Calculation of the correlation\n Constant time step method (faster)";


static PyMethodDef extension_funcs[] =
{
    {"correlation",  (PyCFunction)correlation1, METH_VARARGS, extension_docs_1},
    {"correlation2", (PyCFunction)correlation2, METH_VARARGS, extension_docs_2},
    {NULL}
};

void initcorrelation(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3("correlation", extension_funcs,
                   "Fast Correlation Functions ");
};