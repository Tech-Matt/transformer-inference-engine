#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#define DEFAULT_NDIM 2
#define DEFAULT_DIM_SIZE 2

// Empty Constructor - This initializes the tensor to DEFAULT VALUES
Tensor * Tensor_empty() {
    Tensor * t;
    t = malloc(sizeof(Tensor));
    t->ndim = DEFAULT_NDIM;

    int *dim = malloc(sizeof(int) * DEFAULT_NDIM);
    for(int i = 0; i < t->ndim; i++) {
        dim[i] = DEFAULT_DIM_SIZE;
    }

    t->dim = dim;

    // Compute the strides
    // TODO: strides computation is shared with Tensor_new() and can be turned into a static helper function  
    // strides[i] = product of all dimensions to the right of i
    // e.g. shape [3,4,5] → strides [20, 5, 1] 
    int *strides = malloc(sizeof(int) * t->ndim);
    int stride_partial = 1;
    for (int i = t->ndim -1; i >= 0; i--) {
        strides[i] = stride_partial;
        stride_partial *= t->dim[i];
    }

    // Strides
    t->strides = strides;

    // Data Buffer Allocation
    int data_buffer_size = strides[0] * t->dim[0];
    // calloc() zero-initializes — no memset needed
    float *data = calloc(sizeof(float), data_buffer_size);

    t->data = data;

    return t;
}


// Full constructor
Tensor * Tensor_new(int ndim, int * dim, float *data) {
    Tensor *t = malloc(sizeof(Tensor));
    t->ndim = ndim;

    // Copy dim array
    int *dim_copy = malloc(sizeof(int)*ndim);
    memcpy(dim_copy, dim, sizeof(int)*ndim);
    t->dim = dim_copy;

    // Strides
    // strides[i] = product of all dimensions to the right of i
    // e.g. shape [3,4,5] → strides [20, 5, 1] 
    int *strides = malloc(sizeof(int) * t->ndim);
    int stride_partial = 1;
    for (int i = t->ndim -1; i >= 0; i--) {
        strides[i] = stride_partial;
        stride_partial *= t->dim[i];
    }
    t->strides = strides;

    // Data Buffer allocation
    int data_buffer_size = strides[0] * t->dim[0];
    float *data_copy = malloc(sizeof(float)*data_buffer_size);
    memcpy(data_copy, data, sizeof(float)*data_buffer_size);

    t->data = data_copy;

    return t;
}



// Destructor
void Tensor_free(Tensor *t) {
    free(t->data);
    free(t->strides);
    free(t->dim);
    free(t);
}
