#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

//includes numpy
#include <Python.h>
#include <numpy/arrayobject.h>

// includes CUDA
#include <cufft.h>
#include <cublas_v2.h>

// Complex data type
typedef float2 Complex;
typedef double2 DoubleComplex;


///////////////////////////////////////////////
//                                           //
// Single precision FFT (Complex to complex) //
//                                           //
///////////////////////////////////////////////

static PyObject* fft(PyObject* self, PyObject *arg, PyObject *keywords)
{

    // Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CFLOAT, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    Complex *h_signal = (Complex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);


    //Create new numpy array to store result
    PyArrayObject *return_object;
    int dims[1]={signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CFLOAT);
    Complex *return_data  = (Complex *)PyArray_DATA(return_object);


    int mem_size = sizeof(Complex) * signal_size;

    // Allocate device memory for signal
    Complex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);


    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);

    // Fourier transform using CUFFT_FORWARD
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);


    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    cudaFree(d_signal);

    /* Clean up. */
    Py_DECREF(h_signal_array);

    //Returning Python array
    return(PyArray_Return(return_object));

}


////////////////////////////////////////////////
//                                            //
// Single precision iFFT (Complex to complex) //
//                                            //
////////////////////////////////////////////////

static PyObject* ifft(PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CFLOAT, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    Complex *h_signal = (Complex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *return_object;
    int dims[1]={signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CFLOAT);
    Complex *return_data  = (Complex *)PyArray_DATA(return_object);


    int mem_size = sizeof(Complex) * signal_size;

    // Allocate device memory for signal
    Complex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);
    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);


    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);


    // Inverse Fourier transform using CUFFT_INVERSE
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    float alpha = 1.0 / signal_size;
    cublasCsscal(handle, signal_size,
                 &alpha,
                 d_signal, 1);

    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    // Finish cublas
    cublasDestroy(handle);

    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    cudaFree(d_signal);

    /* Clean up. */
    Py_DECREF(h_signal_array);

    //Returning Python array
    return(PyArray_Return(return_object));

}


///////////////////////////////////////////////
//                                           //
// Double precision FFT (Complex to complex) //
//                                           //
///////////////////////////////////////////////

static PyObject* dfft(PyObject* self, PyObject *arg, PyObject *keywords)
{

    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CDOUBLE, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    DoubleComplex *h_signal = (DoubleComplex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *return_object;
    int dims[1]={signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CDOUBLE);
    DoubleComplex *return_data  = (DoubleComplex *)PyArray_DATA(return_object);


    int mem_size = sizeof(DoubleComplex) * signal_size;

    // Allocate device memory for signal
    DoubleComplex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);


    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_Z2Z, 1);

    // Fourier transform using CUFFT_FORWARD
    cufftExecZ2Z(plan, (cufftDoubleComplex *)d_signal, (cufftDoubleComplex *)d_signal, CUFFT_FORWARD);


    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    cudaFree(d_signal);

    /* Clean up. */
    Py_DECREF(h_signal_array);

    //Returning Python array
    return(PyArray_Return(return_object));

}


////////////////////////////////////////////////
//                                            //
// Double precision iFFT (Complex to complex) //
//                                            //
////////////////////////////////////////////////

static PyObject* difft(PyObject* self, PyObject *arg, PyObject *keywords)
{

    // Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CDOUBLE, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    DoubleComplex *h_signal = (DoubleComplex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *return_object;
    int dims[1]={signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CDOUBLE);
    DoubleComplex *return_data  = (DoubleComplex *)PyArray_DATA(return_object);


    int mem_size = sizeof(DoubleComplex) * signal_size;

    // Allocate device memory for signal
    DoubleComplex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);


    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_Z2Z, 1);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);


    // Inverse Fourier transform using CUFFT_INVERSE
    cufftExecZ2Z(plan, (cufftDoubleComplex *)d_signal, (cufftDoubleComplex *)d_signal, CUFFT_INVERSE);

    double alpha = 1.0 / signal_size;
    cublasZdscal(handle, signal_size,
                 &alpha,
                 d_signal, 1);

    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    // Finish cublas
    cublasDestroy(handle);

    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    cudaFree(d_signal);

    /* Clean up. */
    Py_DECREF(h_signal_array);

    //Returning Python array
    return(PyArray_Return(return_object));
}


static char extension_docs_fft[] =
    "fft(signal)\nFast Fourier Transform implemented in CUDA\n using cuFFT (Single precision)\n  ";

static char extension_docs_ifft[] =
    "ifft(signal)\nInverse Fast Fourier Transform implemented in CUDA\n using cuFFT (Single precision)\n  ";

static char extension_docs_dfft[] =
    "dfft(signal)\nFast Fourier Transform implemented in CUDA\n using cuFFT (Double precision)\n  ";

static char extension_docs_difft[] =
    "difft(signal)\nInverse nFast Fourier Transform implemented in CUDA\n using cuFFT (Double precision)\n  ";


static PyMethodDef extension_funcs[] =
{
    {"fft", (PyCFunction) fft, METH_VARARGS|METH_KEYWORDS, extension_docs_fft},
    {"ifft", (PyCFunction) ifft, METH_VARARGS|METH_KEYWORDS, extension_docs_ifft},
    {"dfft", (PyCFunction) dfft, METH_VARARGS|METH_KEYWORDS, extension_docs_dfft},
    {"difft", (PyCFunction) difft, METH_VARARGS|METH_KEYWORDS, extension_docs_difft},
    {NULL}
};


PyMODINIT_FUNC initgpu_fft(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3("gpu_fft", extension_funcs,
                   "Fast Fourier Tranform functions (CUDA)");
};