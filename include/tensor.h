/* Definition of Tensor data structure and structural operations:
 * - tensor_add()
 * - tensor_mul()
 * - tensor_matmul()
 * - tensor_transpose()
 * - etc.
*/

#ifndef TENSOR_H
#define TENSOR_H


// A tensor is a multidimensional array, which can have as many dimensions as you'd like
// Since we don't know how many dimensions it will have a startup, we can store it as a flat array.
// As an example the Tensor(2, 3, 4) with dimensions 2x3x4 will have 24 total elements which can
// be stored as a flat array.
//
// Parameters
// - ndim: Number of dimensions
// - dim: Set size of each dim (e.g. Tensor(3, 2, 1))
// - strides: strides[i] = product of dim[i+1..ndim-1]; strides[ndim-1] = 1
// - data: the flat tensor array
typedef struct Tensor {
    int ndim;
    int *dim;
    int *strides;
    float *data;
} Tensor;


// Empty constructor
Tensor * Tensor_empty(); 

// Full constructor
Tensor * Tensor_new(int ndim, int * dim, float *data);

// Destructor
void Tensor_free(Tensor *t);

// Element-wise ops - tensor + tensor
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_div(const Tensor *a, const Tensor *b, Tensor *out);

// tensor + scalar
void tensor_add_scalar(const Tensor *a, float s, Tensor *out);
void tensor_sub_scalar(const Tensor *a, float s, Tensor *out);
void tensor_mul_scalar(const Tensor *a, float s, Tensor *out);
void tensor_div_scalar(const Tensor *a, float s, Tensor *out);

// Matrix ops
void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_transpose(const Tensor *a, Tensor *out);


#endif
