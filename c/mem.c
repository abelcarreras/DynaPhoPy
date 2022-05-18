#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

#undef I

// Cross compatibility mingw - gcc
#if !defined(_WIN32)

#define _Dcomplex double _Complex

_Dcomplex _Cbuild(double real, double imag){
    return real + imag* 1.j;
}

_Dcomplex _Cmulcc(_Dcomplex num1, _Dcomplex num2){
    return num1 * num2;
}
#endif


static double FrequencyEvaluation(double Delta, double  Coefficients[], int m, double xms);
static double GetCoefficients(double  *data, int n, int m, double  d[]);
static PyObject* MaximumEntropyMethod (PyObject* self, PyObject *arg, PyObject *keywords);


//  Python Interface
static char function_docstring[] =
    "mem(frequency, velocity, time_step, coefficients=100 )\n\n Maximum Entropy Method\n Constant time step";


static PyMethodDef extension_funcs[] = {
    {"mem",  (PyCFunction)MaximumEntropyMethod, METH_VARARGS|METH_KEYWORDS, function_docstring},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "mem",
  "Maximum entropy method module",
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
    m = Py_InitModule3("mem",
        extension_funcs, "Maximum entropy method module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initmem(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_mem(void)
    {
        import_array();
        return moduleinit();
    }

#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif



static PyObject* MaximumEntropyMethod (PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Declaring initial variables
    double  TimeStep;
    int     NumberOfCoefficients = 100;   //Default value for number of coeficients
    double  AngularFrequency;

    //  Interface with Python
    PyObject *velocity_obj, *frequency_obj;
    static char *kwlist[] = {"frequency", "velocity", "time_step", "coefficients", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOd|i", kwlist, &frequency_obj, &velocity_obj, &TimeStep, &NumberOfCoefficients))  return NULL;

    PyObject *velocity_array = PyArray_FROM_OTF(velocity_obj, NPY_CDOUBLE, NPY_IN_ARRAY);
    PyObject *frequency_array = PyArray_FROM_OTF(frequency_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (velocity_array == NULL || frequency_array == NULL ) {
        Py_XDECREF(velocity_array);
        Py_XDECREF(frequency_array);
        return NULL;
    }

    _Dcomplex  *Velocity = (_Dcomplex *)PyArray_DATA(velocity_array);
    double *Frequency    = (double*)PyArray_DATA(frequency_array);
    int    NumberOfData = (int)PyArray_DIM(velocity_array, 0);
    int     NumberOfFrequencies = (int)PyArray_DIM(frequency_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *PowerSpectrum_object;
    npy_intp dims[]={NumberOfFrequencies};
    PowerSpectrum_object = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);


    double *PowerSpectrum  = (double*)PyArray_DATA(PowerSpectrum_object);

    //Declare variables
    double *Coefficients_r =malloc((NumberOfCoefficients+1)*sizeof(double));
    double *Coefficients_i =malloc((NumberOfCoefficients+1)*sizeof(double));


    // Divide complex data to 2 double arrays
    double *Velocity_r = (double *)malloc(NumberOfData * sizeof(double));
    double *Velocity_i = (double *)malloc(NumberOfData * sizeof(double));

    for (int i=0; i < NumberOfData; i++)  {
        Velocity_r[i] = (double)creal(Velocity[i]);
        Velocity_i[i] = (double)cimag(Velocity[i]);
    }

    // Maximum Entropy Method Algorithm
    double  MeanSquareDiscrepancy_r = GetCoefficients(Velocity_r, NumberOfData, NumberOfCoefficients, Coefficients_r);
    double  MeanSquareDiscrepancy_i = GetCoefficients(Velocity_i, NumberOfData, NumberOfCoefficients, Coefficients_i);

    # pragma omp parallel for default(shared) private(AngularFrequency)
    for (int i=0; i < NumberOfFrequencies; i++) {
        AngularFrequency = Frequency[i]*2.0*M_PI;

        PowerSpectrum[i] = 0.0;
//        if (! isnan(MeanSquareDiscrepancy_r)) {
        if (MeanSquareDiscrepancy_r != 0.0) {
            PowerSpectrum[i] += FrequencyEvaluation(AngularFrequency*TimeStep, Coefficients_r, NumberOfCoefficients, MeanSquareDiscrepancy_r);
        }
        if (MeanSquareDiscrepancy_i != 0.0) {
            PowerSpectrum[i] += FrequencyEvaluation(AngularFrequency*TimeStep, Coefficients_i, NumberOfCoefficients, MeanSquareDiscrepancy_i);
        }
        PowerSpectrum[i] *= TimeStep;
    }

    // Free python memory
    Py_DECREF(velocity_array);
    Py_DECREF(frequency_array);

    // Free memory
    free(Velocity_r);
    free(Coefficients_r);
    free(Coefficients_i);

    //Returning Python array
    return(PyArray_Return(PowerSpectrum_object));
}

// Evaluate MEM function
static double FrequencyEvaluation(double Delta, double  Coefficients[], int NumberOfCoefficients, double MeanSquareDiscrepancy) {

    _Dcomplex z = cexp(_Cbuild(0, Delta));
    _Dcomplex sum = _Cbuild(1, 0);
    _Dcomplex prod = _Cbuild(0, 0);

    for (int i=1; i <= NumberOfCoefficients; i++) {
        //sum -= Coefficients[i] * cpow(z, i);
        prod = _Cmulcc(_Cbuild(Coefficients[i],0), cpow(z, _Cbuild(i,0)));
        sum = _Cbuild(creal(sum) - creal(prod), cimag(sum) - cimag(prod));
    }
    return MeanSquareDiscrepancy/creal(_Cmulcc(sum, conj(sum)));
}

// Get LP coefficients
static double  GetCoefficients(double  *Data, int NumberOfData, int NumberOfCoefficients, double  Coefficients[]) {

    int k, j, i;
    double  p=0.0;

    double  MeanSquareDiscrepancy;
    double  *wk1 = malloc(NumberOfData*sizeof(double));
    double  *wk2 = malloc(NumberOfData*sizeof(double));
    double  *wkm = malloc(NumberOfCoefficients*sizeof(double));


    for (j=1; j <= NumberOfData; j++) p += pow(Data[j], 2);
    MeanSquareDiscrepancy = p / NumberOfData;

    wk1[1] = Data[1];
    wk2[NumberOfData-1] = Data[NumberOfData];

    for (j=2; j <= NumberOfData-1; j++) {
        wk1[j]=Data[j];
        wk2[j-1]=Data[j];
    }

    for (k=1; k <= NumberOfCoefficients; k++) {
        double  Numerator = 0.0, Denominator = 0.0;

        for (j=1; j <= (NumberOfData - k); j++) {
            Numerator += wk1[j] * wk2[j];
            Denominator += pow(wk1[j],2) + pow(wk2[j], 2);
        }

        if (fabs(Denominator) < 1.0e-6) return 0.0;

        Coefficients[k] = 2.0 * Numerator / Denominator;

        MeanSquareDiscrepancy *= (1.0 - pow(Coefficients[k], 2));

        for (i=1; i <= (k-1); i++) Coefficients[i] = wkm[i] - Coefficients[k] * wkm[k-i];

        if (k == NumberOfCoefficients) continue;

        for (i=1; i<=k; i++) wkm[i] = Coefficients[i];

        for (j=1; j <= (NumberOfData-k-1); j++) {
            wk1[j] -= wkm[k] * wk2[j];
            wk2[j]  = wk2[j+1] - wkm[k] * wk1[j+1];
        }
    }
    // Free memory
    free(wk1);
    free(wk2);
    free(wkm);

    return MeanSquareDiscrepancy;
};
