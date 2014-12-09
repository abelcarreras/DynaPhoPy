#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <numpy/arrayobject.h>

#undef I

static float FrequencyEvaluation(float TimeStep, float Coeficients[], int m, float xms);
static double GetCoeficients(_Complex double data[], int n, int m, float d[]);


static PyObject* MaximumEntropyMethod (PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Declaring initial variables
    double  TimeStep;
    int     NumberOfCoefficients = 100;   //Default value for number of coeficients

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

    double _Complex  *Velocity = (double _Complex*)PyArray_DATA(velocity_array);
    double *Frequency    = (double*)PyArray_DATA(frequency_array);
    int     NumberOfData = (int)PyArray_DIM(velocity_array, 0);
    int     NumberOfFrequencies = (int)PyArray_DIM(frequency_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *PowerSpectrum_object;
    int dims[1]={NumberOfFrequencies};
    PowerSpectrum_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT);
    float *PowerSpectrum  = (float*)PyArray_DATA(PowerSpectrum_object);

    //Declare internal variables
    float coefficients[NumberOfCoefficients];

    // Maximum Entropy Method Algorithm
    double MeanSquareDiscrepancy = GetCoeficients(Velocity, NumberOfData, NumberOfCoefficients, coefficients);
    for (int i=0;i<NumberOfFrequencies;i++)
        PowerSpectrum[i] = FrequencyEvaluation(Frequency[i]*TimeStep, coefficients, NumberOfCoefficients, MeanSquareDiscrepancy);

    //Returning Python array
    return(PyArray_Return(PowerSpectrum_object));
}


static float FrequencyEvaluation(float TimeStep, float Coefficients[], int NumberOfCoefficients, float MeanSquareDiscrepancy) {

    double _Complex z = cexp(2*M_PI*_Complex_I *TimeStep);
    double _Complex sum = 1.0 + 0.0 * _Complex_I;

    for (int i=1;i<=NumberOfCoefficients;i++) {
        sum -= Coefficients[i]*cpow(z,i);
    }
    return MeanSquareDiscrepancy/(sum*conj(sum));
}


static double GetCoeficients(_Complex double Data[], int NumberOfData, int NumberOfCoefficients, float Coefficients[]) {

    int k,j,i;
    float p=0.0;

    double MeanSquareDiscrepany;
    double wk1 [NumberOfData];
    double wk2 [NumberOfData];
    double wkm [NumberOfCoefficients];

    for (j=1;j<=NumberOfData;j++) p += pow(Data[j],2);
    MeanSquareDiscrepany=p/NumberOfData;

    wk1[1]=Data[1];
    wk2[NumberOfData-1]=Data[NumberOfData];


    for (j=2;j<=NumberOfData-1;j++) {
        wk1[j]=Data[j];
        wk2[j-1]=Data[j];
    }


    for (k=1;k<=NumberOfCoefficients;k++) {

        float Numerator=0.0,Denominator=0.0;

        for (j=1;j<=(NumberOfData-k);j++) {
            Numerator += wk1[j]*wk2[j];
            Denominator += pow(wk1[j],2)+pow(wk2[j],2);
        }

        Coefficients[k]=2.0*Numerator/Denominator;

        MeanSquareDiscrepany *= (1.0-pow(Coefficients[k],2));

        for (i=1;i<=(k-1);i++) Coefficients[i]=wkm[i]-Coefficients[k]*wkm[k-i];

        if (k == NumberOfCoefficients) return MeanSquareDiscrepany;

        for (i=1;i<=k;i++) wkm[i]=Coefficients[i];

        for (j=1;j<=(NumberOfData-k-1);j++) {
            wk1[j] -= wkm[k]*wk2[j];
            wk2[j]  = wk2[j+1]-wkm[k]*wk1[j+1];
        }
    }
    return 0;
};



//static char extension_docs[] =
//"MEM ( ): Calculation of Power spectra with <Maximum Entropy Method";

static char extension_docs[] =
    "MeM ( ): Maximum Entropy Method\n Constant time step ";


static PyMethodDef extension_funcs[] =
{
    {"mem",  (PyCFunction)MaximumEntropyMethod, METH_VARARGS|METH_KEYWORDS, extension_docs},
    {NULL}
};


void initmem(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3("mem", extension_funcs,
                   "Fast Correlation Functions ");
};