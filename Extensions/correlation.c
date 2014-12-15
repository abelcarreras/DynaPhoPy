#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#undef I

static double EvaluateCorrelation (double Frequency, double _Complex Velocity[], int NumberOfData, double TimeStep, int Increment, int IntMethod);



// General correlation method for non-constant time step trajectories
static PyObject* correlation1 (PyObject* self, PyObject *arg, PyObject *keywords)
{

//  Declaring basic variables
    double  Frequency;
	int     Increment = 10;   //Default value for Increment
 	int     IntMethod = 1;    //Define integration method

//  Interface with python
    PyObject *VQ_obj, *Time_obj;
    static char *kwlist[] = {"frequency","vq","time","step","integration_method",NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "dOO|ii", kwlist, &Frequency, &VQ_obj, &Time_obj, &Increment, &IntMethod))  return NULL;

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
			switch (IntMethod) {
				case 0: //	Trapezoid Integration
					Correl += (conj(VQ[j]) * VQ[j+i+Increment] * cexp(_Complex_I*Frequency                            * (Time[i+Increment] - Time[0]))
					       +   conj(VQ[j]) * VQ[j+i]           * cexp(_Complex_I*Frequency * (Time[i]-Time[0])) )/2.0 * (Time[i+Increment] - Time[i]);
					break;
				case 1: //	Rectangular Integration
					Correl +=  conj(VQ[j]) * VQ[j+i]           * cexp(_Complex_I*Frequency * (Time[i]-Time[0]))       * (Time[i+Increment] - Time[i]);
					break;
				default:
				    puts ("\nIntegration method selected does not exist\n");
				    return NULL;
					break;
			}
		}
	}
    return Py_BuildValue ("d", creal(Correl)/NumberOfData);
};

// Correlation method for constant time (DTime) step trajectories
static PyObject* correlation2 (PyObject* self, PyObject *arg, PyObject *keywords )
{

//  Declaring basic variables
    double  Frequency;
    double  DTime;
 	int     Increment = 10;   //Default value for Increment
 	int     IntMethod = 1;    //Define integration method


//  Interface with python
    PyObject *VQ_obj;
    static char *kwlist[] = {"frequency","vq","dtime","step","integration_method",NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "dOd|ii", kwlist, &Frequency, &VQ_obj, &DTime, &Increment, &IntMethod))  return NULL;

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

            switch (IntMethod) {
                case 0: //	Trapezoid Integration
                    Correl += (conj(VQ[j]) * VQ[j+i+Increment] * cexp(_Complex_I*Frequency * ((i+Increment)*DTime))
                           +   conj(VQ[j]) * VQ[j+i]           * cexp(_Complex_I*Frequency * (i*DTime) ))/2.0 ;
                    break;
                case 1: //	Rectangular Integration
                     Correl +=  conj(VQ[j]) * VQ[j+i]          * cexp(_Complex_I*Frequency * (i*DTime));
                    break;
                default:
                    puts ("\nIntegration method selected does not exist\n");
                    return NULL;
                    break;
            }
		}
	}

    return Py_BuildValue("d", creal(Correl) * DTime/(NumberOfData/Increment));

};

// Correlation method for constant time (DTime) step trajectories paralellized with OpenMP
static PyObject* correlation2_par (PyObject* self, PyObject *arg, PyObject *keywords)
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

    double _Complex  *Velocity = (double _Complex*)PyArray_DATA(velocity_array);
    double *Frequency    = (double*)PyArray_DATA(frequency_array);
    int     NumberOfData = (int)PyArray_DIM(velocity_array, 0);
    int     NumberOfFrequencies = (int)PyArray_DIM(frequency_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *PowerSpectrum_object;
    int dims[1]={NumberOfFrequencies};
    PowerSpectrum_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT);
    float *PowerSpectrum  = (float*)PyArray_DATA(PowerSpectrum_object);

    // Maximum Entropy Method Algorithm
    if (IntMethod < 0 || IntMethod > 1) {
        puts ("\nIntegration method selected does not exist\n");
        return NULL;
    }

    int i;
# pragma omp parallel for default(shared) private(i,AngularFrequency)
    for (i=0;i<NumberOfFrequencies;i++) {
        AngularFrequency = Frequency[i]*2.0*M_PI;
        PowerSpectrum[i] = EvaluateCorrelation(AngularFrequency, Velocity, NumberOfData, TimeStep, Increment, IntMethod);
    }
    //Returning Python array
    return(PyArray_Return(PowerSpectrum_object));
}


double EvaluateCorrelation (double Frequency, double _Complex Velocity[], int NumberOfData, double TimeStep, int Increment, int IntMethod) {

	double _Complex Correl = 0;
	for (int i = 0; i < NumberOfData; i += Increment) {
		for (int j = 0; j < (NumberOfData-i-Increment); j++) {
			Correl += conj(Velocity[j]) * Velocity[j+i] * cexp(_Complex_I*Frequency*(i*TimeStep));

            switch (IntMethod) {
                case 0: //	Trapezoid Integration
                    Correl += (conj(Velocity[j]) * Velocity[j+i+Increment] * cexp(_Complex_I*Frequency * ((i+Increment)*TimeStep))
                           +   conj(Velocity[j]) * Velocity[j+i]           * cexp(_Complex_I*Frequency * (i*TimeStep) ))/2.0 ;
                    break;
                case 1: //	Rectangular Integration
                     Correl +=  conj(Velocity[j]) * Velocity[j+i]          * cexp(_Complex_I*Frequency * (i*TimeStep));
                    break;
            }
		}
	}
    return  creal(Correl) * TimeStep/(NumberOfData/Increment);
}


static char extension_docs_1[] =
    "correlation ( ): Calculation of the correlation\n Non constant time step (slower)";

static char extension_docs_2[] =
    "correlation2( ): Calculation of the correlation\n Constant time step method (faster)";

static char extension_docs_3[] =
    "correlation3( ): Calculation of the correlation\n Constant time step method (faster)\n OpenMP paralellized";

static PyMethodDef extension_funcs[] =
{
    {"correlation",  (PyCFunction)correlation1, METH_VARARGS|METH_KEYWORDS, extension_docs_1},
    {"correlation2", (PyCFunction)correlation2, METH_VARARGS|METH_KEYWORDS, extension_docs_2},
    {"correlation2_par", (PyCFunction)correlation2_par, METH_VARARGS|METH_KEYWORDS, extension_docs_3},
    {NULL}
};

void initcorrelation(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3("correlation", extension_funcs,
                   "Fast Correlation Functions ");
};