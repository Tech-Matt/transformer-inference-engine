#include <alloca.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tensor.h"

#define DEFAULT_NDIM 2
#define DEFAULT_DIM_SIZE 2

/**
 * @brief Construct a zero-initialized tensor with default shape [2, 2].
 *
 * Allocates the Tensor struct, dim array, strides array, and data buffer.
 * Data is zero-initialized via calloc. Caller is responsible for calling
 * Tensor_free() when done.
 *
 * @return Pointer to newly allocated Tensor.
 */
Tensor * Tensor_empty(Arena *a) {
    // Allocate the struct from the arena
    Tensor * t = arena_alloc(a, sizeof(Tensor));
    t->ndim = DEFAULT_NDIM;

    // Initialize dim
    for(int i = 0; i < t->ndim; i++) {
        t->dim[i] = DEFAULT_DIM_SIZE;
    }

    // Initializes strides
    int stride_partial = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = stride_partial;
        stride_partial *= t->dim[i];
    }

    // Allocate data from the arena and ZERO it out
    int data_buffer_size = t->strides[0] * t->dim[0];
    t->data = arena_alloc(a, data_buffer_size * sizeof(float));
    memset(t->data, 0, data_buffer_size * sizeof(float));

    return t;
}


/**
 * @brief Construct a tensor with a given shape and copy data into it.
 *
 * Allocates the Tensor struct, dim array, strides array, and data buffer.
 * Both `dim` and `data` are copied — the caller retains ownership of the
 * original arrays. Caller is responsible for calling Tensor_free() when done.
 *
 * @param ndim  Number of dimensions.
 * @param dim   Array of length `ndim` specifying the size of each dimension.
 * @param data  Flat array of (product of all dims) floats to copy in.
 * @return Pointer to newly allocated Tensor.
 */
Tensor * Tensor_new(Arena *a, int ndim, int *dim, float *data) {
    // Allocate tensor on the arena
    Tensor *t = arena_alloc(a, sizeof(Tensor));
    t->ndim = ndim;
    memcpy(t->dim, dim, sizeof(int) * ndim);

    /* strides[i] = product of dim[i+1..ndim-1]; strides[ndim-1] = 1
     * e.g. shape [3,4,5] → strides [20, 5, 1] */
    int stride_partial = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = stride_partial;
        stride_partial *= t->dim[i];
    }

    // Allocate data on the arena and copy values
    int data_buffer_size = t->strides[0] * t->dim[0];
    t->data = arena_alloc(a, sizeof(float) * data_buffer_size);
    memcpy(t->data, data, sizeof(float) * data_buffer_size);

    return t;
}


/**
 * @brief Element-wise addition: out[i] = a[i] + b[i]
 *
 * All three tensors must have identical shape. `out` may alias `a` or `b`
 * for in-place operation (safe because each element is read before it is
 * written). Asserts fire on shape mismatch; stripped by -DNDEBUG in release.
 *
 * @param a   Left operand (read-only).
 * @param b   Right operand (read-only).
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out) {
    assert(a->ndim == b->ndim && "tensor_add: ndim mismatch between a and b");
    assert(a->ndim == out->ndim && "tensor_add: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == b->dim[i] && "tensor_add: dim mismatch between a and b");
        assert(a->dim[i] == out->dim[i] && "tensor_add: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}


/**
 * @brief Element-wise subtraction: out[i] = a[i] - b[i]
 *
 * See tensor_add for shape, aliasing, and assert behaviour.
 *
 * @param a   Left operand (read-only).
 * @param b   Right operand (read-only).
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_sub(const Tensor *a, const Tensor *b, Tensor *out) {
    assert(a->ndim == b->ndim && "tensor_sub: ndim mismatch between a and b");
    assert(a->ndim == out->ndim && "tensor_sub: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == b->dim[i] && "tensor_sub: dim mismatch between a and b");
        assert(a->dim[i] == out->dim[i] && "tensor_sub: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}


/**
 * @brief Element-wise multiplication: out[i] = a[i] * b[i]
 *
 * See tensor_add for shape, aliasing, and assert behaviour.
 *
 * @param a   Left operand (read-only).
 * @param b   Right operand (read-only).
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_mul(const Tensor *a, const Tensor *b, Tensor *out) {
    assert(a->ndim == b->ndim && "tensor_mul: ndim mismatch between a and b");
    assert(a->ndim == out->ndim && "tensor_mul: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == b->dim[i] && "tensor_mul: dim mismatch between a and b");
        assert(a->dim[i] == out->dim[i] && "tensor_mul: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}


/**
 * @brief Element-wise division: out[i] = a[i] / b[i]
 *
 * See tensor_add for shape, aliasing, and assert behaviour. Additionally
 * asserts that no element of `b` is zero. In production code (layer norm,
 * softmax), callers should add an epsilon to the denominator before calling
 * this function rather than relying on the assert.
 *
 * @param a   Numerator tensor (read-only).
 * @param b   Denominator tensor (read-only); no element may be zero.
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_div(const Tensor *a, const Tensor *b, Tensor *out) {
    assert(a->ndim == b->ndim && "tensor_div: ndim mismatch between a and b");
    assert(a->ndim == out->ndim && "tensor_div: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == b->dim[i] && "tensor_div: dim mismatch between a and b");
        assert(a->dim[i] == out->dim[i] && "tensor_div: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        assert(b->data[i] != 0.0f && "tensor_div: division by zero in b");
        out->data[i] = a->data[i] / b->data[i];
    }
}



/**
 * @brief Element-wise scalar addition: out[i] = a[i] + s
 *
 * `a` and `out` must have identical shape. `out` may alias `a` for in-place
 * operation. Asserts fire on shape mismatch;
 *
 * @param a   Input tensor (read-only).
 * @param s   Scalar addend.
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_add_scalar(const Tensor *a, float s, Tensor *out) {
    assert(a->ndim == out->ndim && "tensor_add_scalar: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == out->dim[i] && "tensor_add_scalar: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] + s;
    }
}


/**
 * @brief Element-wise scalar subtraction: out[i] = a[i] - s
 *
 * See tensor_add_scalar for shape, aliasing, and assert behaviour.
 *
 * @param a   Input tensor (read-only).
 * @param s   Scalar subtrahend.
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_sub_scalar(const Tensor *a, float s, Tensor *out) {
    assert(a->ndim == out->ndim && "tensor_sub_scalar: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == out->dim[i] && "tensor_sub_scalar: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] - s;
    }
}


/**
 * @brief Element-wise scalar multiplication: out[i] = a[i] * s
 *
 * See tensor_add_scalar for shape, aliasing, and assert behaviour.
 *
 * @param a   Input tensor (read-only).
 * @param s   Scalar multiplier.
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_mul_scalar(const Tensor *a, float s, Tensor *out) {
    assert(a->ndim == out->ndim && "tensor_mul_scalar: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == out->dim[i] && "tensor_mul_scalar: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] * s;
    }
}


/**
 * @brief Element-wise scalar division: out[i] = a[i] / s
 *
 * See tensor_add_scalar for shape, aliasing, and assert behaviour. `s` must
 * not be zero. In production code, callers should ensure s != 0.0f (e.g. by
 * adding an epsilon) rather than relying on the assert.
 *
 * @param a   Input tensor (read-only).
 * @param s   Scalar divisor; must not be zero.
 * @param out Pre-allocated output tensor; caller owns allocation.
 */
void tensor_div_scalar(const Tensor *a, float s, Tensor *out) {
    assert(s != 0.0f && "tensor_div_scalar: division by zero");
    assert(a->ndim == out->ndim && "tensor_div_scalar: ndim mismatch between a and out");
    for (int i = 0; i < a->ndim; i++) {
        assert(a->dim[i] == out->dim[i] && "tensor_div_scalar: dim mismatch between a and out");
    }
    int n_elem = a->strides[0] * a->dim[0];
    for (int i = 0; i < n_elem; i++) {
        out->data[i] = a->data[i] / s;
    }
}


/**
 * @brief Matrix multiplication: out = a @ b
 *
 * Both inputs must be 2D. `a` has shape [M, K], `b` has shape [K, N],
 * `out` must be pre-allocated with shape [M, N]. `out` must not alias
 * `a` or `b` — in-place matmul corrupts intermediate results.
 *
 * Uses i-j-k loop ordering. Consider switching to i-k-j for better
 * cache locality on large matrices (makes inner loop contiguous in b).
 *
 * @param a   Left matrix [M, K] (read-only).
 * @param b   Right matrix [K, N] (read-only).
 * @param out Pre-allocated output matrix [M, N]; caller owns allocation.
 */
void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out) {
    assert(a->ndim == 2 && "tensor_matmul: a must be 2D");
    assert(b->ndim == 2 && "tensor_matmul: b must be 2D");
    assert(out->ndim == 2 && "tensor_matmul: out must be 2D");
    assert(a->dim[1] == b->dim[0] && "tensor_matmul: inner dimensions must match (a cols != b rows)");
    assert(out->dim[0] == a->dim[0] && out->dim[1] == b->dim[1] && "tensor_matmul: out shape mismatch");

    for (int i = 0; i < a->dim[0]; i++) {
        for (int j = 0; j < b->dim[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->dim[1]; k++) {
                sum += a->data[i * a->strides[0] + k * a->strides[1]]
                     * b->data[k * b->strides[0] + j * b->strides[1]];
            }
            out->data[i * out->strides[0] + j * out->strides[1]] = sum;
        }
    }
}

/**
 * @brief Transpose a 2D matrix: out[j][i] = a[i][j]
 *
 * `a` has shape [M, N]; `out` must be pre-allocated with shape [N, M].
 * Physically copies data into the transposed layout. A zero-copy version
 * (swapping strides only) is possible but requires view semantics not yet
 * supported by the Tensor struct.
 *
 * @param a   Input matrix [M, N] (read-only).
 * @param out Pre-allocated output matrix [N, M]; caller owns allocation.
 */
void tensor_transpose(const Tensor *a, Tensor *out) {
    assert(a->ndim == 2 && "tensor_transpose: a must be 2D");
    assert(out->ndim == 2 && "tensor_transpose: out must be 2D");
    assert(a->dim[0] == out->dim[1] && "tensor_transpose: out shape must be [N, M] for a [M, N] input");
    assert(a->dim[1] == out->dim[0] && "tensor_transpose: out shape must be [N, M] for a [M, N] input");

    for (int i = 0; i < a->dim[0]; i++) {
        for (int j = 0; j < a->dim[1]; j++) {
            out->data[j * out->strides[0] + i * out->strides[1]]
                = a->data[i * a->strides[0] + j * a->strides[1]];
        }
    }
}
