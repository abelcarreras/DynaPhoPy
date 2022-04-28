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


static double EvaluateCorrelation (double AngularFrequency, _Dcomplex * Velocity, int NumberOfData, double TimeStep, int Increment, int IntMethod);
static PyObject* correlation_par (PyObject* self, PyObject *arg, PyObject *keywords);



//  Python Interface
static char function_docstring[] =
    "correlation_par(frequency, velocity, timestep, step=10, integration_method=1 )\n\n Calculation of the correlation\n Constant time step method (faster)\n OpenMP parallel";

static PyMethodDef extension_funcs[] = {
    {"correlation_par", (PyCFunction)correlation_par, METH_VARARGS|METH_KEYWORDS, function_docstring},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "correlation",
  "power spectrum direct method module ",
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
    m = Py_InitModule3("correlation",
        extension_funcs, "power spectrum direct method module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    initcorrelation(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    PyInit_correlation(void)
    {
        import_array();
        return moduleinit();
    }

#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


// Correlation method for constant time (TimeStep) step trajectories paralellized with OpenMP
static PyObject* correlation_par (PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Declaring initial variables
    double  TimeStep;
    double  AngularFrequency;
 	int     Increment = 10;   //Default value for Increment
 	int     IntMethod = 1;    //Define integration method

    //  Interface with Python
    PyObject *velocity_obj, *frequency_obj;
    static char *kwlist[] = {"frequency", "velocity", "time_step", "step", "integration_method", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "OOd|ii", kwlist, &frequency_obj, &velocity_obj, &TimeStep, &Increment, &IntMethod))  return NULL;

    PyObject *velocity_array = PyArray_FROM_OTF(velocity_obj, NPY_CDOUBLE, NPY_IN_ARRAY);
    PyObject *frequency_array = PyArray_FROM_OTF(frequency_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (velocity_array == NULL || frequency_array == NULL ) {
        Py_XDECREF(velocity_array);
        Py_XDECREF(frequency_array);
        return NULL;
    }

    _Dcomplex  *Velocity = (_Dcomplex *)PyArray_DATA(velocity_array);
    double *Frequency    = (double*)PyArray_DATA(frequency_array);
    int     NumberOfData = (int)PyArray_DIM(velocity_array, 0);
    int     NumberOfFrequencies = (int)PyArray_DIM(frequency_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *PowerSpectrum_object;
    int dims[1]={NumberOfFrequencies};
    PowerSpectrum_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_DOUBLE);
    double *PowerSpectrum  = (double*)PyArray_DATA(PowerSpectrum_object);

    // Maximum Entropy Method Algorithm
    if (IntMethod < 0 || IntMethod > 1) {
        puts ("\nIntegration method selected does not exist\n");
        return NULL;
    }

# pragma omp parallel for default(shared) private(AngularFrequency)
    for (int i=0;i<NumberOfFrequencies;i++) {
        AngularFrequency = Frequency[i]*2.0*M_PI;
        PowerSpectrum[i] = EvaluateCorrelation(AngularFrequency, Velocity, NumberOfData, TimeStep, Increment, IntMethod);
    }
    //Returning Python array
    return(PyArray_Return(PowerSpectrum_object));
}

// Averaged
double EvaluateCorrelation (double Frequency, _Dcomplex * Velocity, int NumberOfData, double TimeStep, int Increment, int IntMethod) {
    _Dcomplex Correl = _Cbuild(0.0, 0.0);
    _Dcomplex Correl_i, Correl_j;
    for (int i = 0; i < NumberOfData; i += Increment) {
        for (int j = 0; j < (NumberOfData-i-Increment); j++) {
            Correl_i = _Cmulcc(_Cmulcc(conj(Velocity[j]), Velocity[j+i]), cexp(_Cbuild(Frequency*(i*TimeStep), 1)));
            Correl = _Cbuild(creal(Correl) + creal(Correl_i), cimag(Correl) + cimag(Correl_i));
            switch (IntMethod) {
                case 0: //	Trapezoid Integration
                    Correl_i = _Cmulcc(_Cmulcc(conj(Velocity[j]), Velocity[j+i+Increment]), cexp(_Cbuild(Frequency * ((i+Increment)*TimeStep), 1)));
                    Correl_j = _Cmulcc(_Cmulcc(conj(Velocity[j]), Velocity[j+i]), cexp(_Cbuild(Frequency * (i+Increment), 1)));
                    Correl = _Cbuild(creal(Correl) + creal(Correl_i) + creal(Correl_j), cimag(Correl) + cimag(Correl_i) + cimag(Correl_j));

                    //Correl += (conj(Velocity[j]) * Velocity[j+i+Increment] * cexp(_Cbuild(Frequency * ((i+Increment)*TimeStep), 1))
                    //           +   conj(Velocity[j]) * Velocity[j+i]           * cexp(_Complex_I*Frequency * (i*TimeStep) ))/2.0 ;
                    break;
                case 1: //	Rectangular Integration
                    Correl_i = _Cmulcc(_Cmulcc(conj(Velocity[j]), Velocity[j+i]), cexp(_Cbuild(Frequency*(i*TimeStep), 1)));
                    Correl = _Cbuild(creal(Correl) + creal(Correl_i), cimag(Correl) + cimag(Correl_i));
                    // Correl +=  conj(Velocity[j]) * Velocity[j+i] * cexp(_Complex_I*Frequency * (i*TimeStep));
                    break;
            }
        }
    }
    return  creal(Correl) * TimeStep/(NumberOfData/Increment);
}

/*
// Original (simple)
double EvaluateCorrelation (double AngularFrequency, double _Complex Velocity[], int NumberOfData, double TimeStep, int Increment, int IntMethod) {

    double _Complex Correl;
    double _Complex Integral = 0;
    for (int i = 0; i < NumberOfData-Increment-1; i += Increment) {
        Correl = 0;
        for (int j = 0; j < (NumberOfData-i-Increment); j++) {


            switch (IntMethod) {
                    case 0: //	Trapezoid Integration
                    Correl += (conj(Velocity[j]) * Velocity[j+i+Increment] * cexp(_Complex_I*AngularFrequency * ((i+Increment)*TimeStep))
                               +   conj(Velocity[j]) * Velocity[j+i]           * cexp(_Complex_I*AngularFrequency * (i*TimeStep) ))/2.0 ;
                    break;
                    case 1: //	Rectangular Integration
                    Correl +=  conj(Velocity[j]) * Velocity[j+i]          * cexp(_Complex_I*AngularFrequency * (i*TimeStep));
                    break;
            }
         //   printf("\nDins %f",creal(Correl));

        }
        Integral += Correl/(NumberOfData -i -Increment);
      //  printf("\n%i %f",i, creal(Correl/(NumberOfData -i -Increment)));

    }


    return  creal(Integral)* TimeStep * Increment;
}
*/
