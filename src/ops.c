#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ops.h"

/**
 * @brief Rectified Linear Unit function
 *
 * Performs ReLU on the `in` tensor (without modifying it).
 * Output is going to be set on the `out` tensor, which need to be
 * properly allocated by the caller.
 *
 * @param in Input Tensor.
 * @param out Output Tensor
 */
void relu(const Tensor *in, Tensor *out) {
    // Check tensors sizes match
    assert(in->ndim == out->ndim && "Tensors ndim do not match.");
    for (int i = 0; i < in->ndim; i++) {
        assert(in->dim[i] == out->dim[i] && "Tensors dim do not match.");
    }
    
    int tensor_size = in->strides[0] * in->dim[0];

    // Perform RELU
    for (int i = 0; i < tensor_size; i++) {
        out->data[i] = fmaxf(0.0f, in->data[i]);
    }
}


/**
 * @brief Softmax function
 *
 * Performs the softmax function on the `in` tensor (without modifying it).
 * Output is going to be set on the `out` tensor, which need to be
 * properly allocated by the caller.
 *
 * @param in Input Tensor.
 * @param out Output Tensor
 */ 
void softmax(const Tensor *in, Tensor *out) {
    // Check tensors sizes match
    assert(in->ndim == out->ndim && "Tensors ndim do not match.");
    for (int i = 0; i < in->ndim; i++) {
        assert(in->dim[i] == out->dim[i] && "Tensors dim do not match.");
    }

    int tensor_size = in->strides[0] * in->dim[0];

    // Find max value in tensor
    float max = in->data[0];
    for (int i = 0; i < tensor_size; i++) {
        max = fmaxf(max, in->data[i]);
    }

    // Compute exp(x - max) for each elem and store it
    // accumulate total sum in exp_sum
    float exp_sum = 0.0f;
    for (int i = 0; i < tensor_size; i++) {
        float temp_exp = expf(in->data[i] - max);
        out->data[i] = temp_exp;
        exp_sum += temp_exp;
    }

    // Divide each element by exp_sum
    for (int i = 0; i < tensor_size; i++) {
        out->data[i] /= exp_sum;
    }

}

/**
 * @brief Log Softmax Function
 *
 * Performs Log Softmax on an input tensor. Output goes into output
 * tensor, which needs to be previously allocated by the caller.
 *
 * @param in Input Tensor.
 * @param out Output Tensor
 */ 
void log_softmax(const Tensor *in, Tensor *out) {
    // Check tensors sizes match
    assert(in->ndim == out->ndim && "Tensors ndim do not match.");
    for (int i = 0; i < in->ndim; i++) {
        assert(in->dim[i] == out->dim[i] && "Tensors dim do not match.");
    }

    int tensor_size = in->strides[0] * in->dim[0];

    // Find max value in tensor
    float max = in->data[0];
    for (int i = 0; i < tensor_size; i++) {
        max = fmaxf(max, in->data[i]);
    }

    // Compute exp(x - max) and accumulate in exp_sum
    float exp_sum = 0.0f;
    for (int i = 0; i < tensor_size; i++) {
        exp_sum += expf(in->data[i] - max);
    }

    // Apply (x - max) - log(exp_sum) for each element
    for (int i = 0; i < tensor_size; i++) {
        out->data[i] = (in->data[i] - max) - logf(exp_sum);
    }
}


/**
 * @brief Mean reduction over one axis.
 *
 * Reduces `in` along `axis` and stores the result into `out`.
 * `out` must keep the same ndim as `in`, with out->dim[axis] = 1,
 * and all other dimensions unchanged.
 *
 * @param in Input Tensor.
 * @param out Output Tensor.
 * @param axis Axis to reduce.
 */
void mean(const Tensor *in, Tensor *out, int axis) {
    // Check ndim and axis validity
    assert(in->ndim == out->ndim && "mean: ndim mismatch");
    assert(axis >= 0 && axis < in->ndim && "mean: axis out of bounds");
    // Check shapes
    for (int i = 0; i < in->ndim; i++) {
        if (i == axis) {
            assert(out->dim[i] == 1 && "mean: output dim at axis must be 1");
        } else {
            assert(in->dim[i] == out->dim[i] && "mean: dim mismatch at non-axis index");
        }
    }

    // Zero-initialize output
    int out_buf_size = out->strides[0] * out->dim[0];
    for (int i = 0; i < out_buf_size; i++) {
        out->data[i] = 0.0f;
    }

    // Loop through input elements
    int in_buf_size = in->strides[0] * in->dim[0];
    for (int flat_idx = 0; flat_idx < in_buf_size; flat_idx++) {
        // Convert flat-index indecing to multi-dim coordinates
        // e.g flat_idx=5 in [3, 4] tensor -> coords=[1, 1]
        int coords[in->ndim];
        int rem = flat_idx;
        for(int d = 0; d < in->ndim; d++) {
            coords[d] = rem / in->strides[d]; 
            rem %= in->strides[d]; 
        }
        // at the end of this cycle coords=[1, 1] for a tensor like
        // the one above with dim[3, 4]

        // Build output coordinats by DROPPING the reduction axis
        // e.g. coords=[1, 2, 3], axis=1 (drop axis 1)
        // -> output coords=[1, 0, 3]
        int out_coords[out->ndim];
        for (int d = 0; d < in->ndim; d++) {
            if (d != axis) {
                out_coords[d] = coords[d]; // Keep this coordinate
            } else {
                out_coords[d] = 0;  // Reduced dimension
            }
        }

        // Convert output coordinates back to flat index
        // E.g. out_coords=[1,0,3] with strides=[4,3,1]
        // -> out_flat = 1*4 + 0*3 + 3*1 = 7
        int out_flat = 0;
        for(int d = 0; d < out->ndim; d++) {
            out_flat += out_coords[d] * out->strides[d];
        }

        // Add the input value to the corresponding output bucket
        out->data[out_flat] += in->data[flat_idx];
        // what happens in the above step is that elements of the
        // reduced axis are summed over in the output bucket
    }
    
    // Divide all sums by the number of elems summed
    float inv_n = 1.0f / (float)in->dim[axis];
    for (int i = 0; i < out_buf_size; i++) {
        out->data[i] *= inv_n;
    }
}


/**
 * @brief Variance reduction over one axis.
 *
 * Computes population variance along `axis` and stores it into `out`.
 * Shape rules are identical to mean():
 * out->dim[axis] = 1 and all other dimensions unchanged.
 *
 * @param in Input Tensor.
 * @param out Output Tensor.
 * @param axis Axis to reduce.
 */
void var(const Tensor* in, Tensor *out, int axis) {
    // Check ndim, axis, and output shape
    assert(in->ndim == out->ndim && "var: ndim mismatch");
    assert(axis >= 0 && axis < in->ndim && "var: axis out of bounds");
 
    for (int d = 0; d < in->ndim; d++) {
        if (d == axis) {
            assert(out->dim[d] == 1 && "var: reduced axis dim must be 1");
        } else {
            assert(out->dim[d] == in->dim[d] && "var: non-axis dim mismatch");
        }
    }

    int out_elems = out->strides[0] * out->dim[0];
    float *mean_buf = malloc(sizeof(float) * out_elems);
    assert(mean_buf != NULL && "var: mean buffer allocation failed");

    // Pass 1: compute mean per output bucket
    mean(in, out, axis);

    for (int i = 0; i < out_elems; i++) {
        mean_buf[i] = out->data[i]; // save the mean
        out->data[i] = 0.0f; // Reset out for pass 2
    }

    // Pass 2: accumulate squared distance to bucket mean
    int in_elems = in->strides[0] * in->dim[0];
    for (int flat = 0; flat < in_elems; flat++) {
        int rem = flat;
        int out_flat = 0;

        for (int d = 0; d < in->ndim; d++) {
            int coord = rem / in->strides[d];
            rem %= in->strides[d];

            if (d != axis) {
                out_flat += coord * out->strides[d];
            }
        }

        float diff = in->data[flat] - mean_buf[out_flat];
        // accumulate (x - mean)^2
        out->data[out_flat] += diff * diff;
    }

    // Convert squared-diff sums to population variance
    float inv_n = 1.0f / (float)in->dim[axis];
    for (int i = 0; i < out_elems; i++) {
        out->data[i] *= inv_n;
    }

    free(mean_buf);
}


void layer_norm(const Tensor *x, Tensor *y, const Tensor *gamma, const Tensor *beta, float epsilon, int axis) {
    // Check ndim, axis and output shape
    assert(x->ndim == y->ndim && "layer_norm: ndim mismatch");
    assert(gamma->ndim == 1 && beta->ndim == 1);
    assert(axis >= 0 && axis < x->ndim && "var: axis out of bounds");
    for (int i = 0; i < x->ndim; i++) {
        assert(x->dim == y->dim && "layer_norm: dim mismatch");
    }
    
    
    // Initialize temp Tensors for mean() and var()
    float *data_mean = malloc(sizeof(float) * x->dim[0] * x->strides[0]);
    int *dim_mean = malloc(sizeof(int) * x->ndim);
    memcpy(dim_mean, x->dim, sizeof(int) * x->ndim);
    dim_mean[axis] = 1;
    Tensor *t_mean = Tensor_new(x->ndim, dim_mean, data_mean); 
    float *data_var = malloc(sizeof(float) * x->dim[0] * x->strides[0]);
    int *dim_var = malloc(sizeof(int) * x->ndim);
    memcpy(dim_var, x->dim, sizeof(int) * x->ndim);
    dim_var[axis] = 1;
    Tensor * t_var = Tensor_new(x->ndim, dim_var, data_var); 

    // Compute mean(x), var(x)
    mean(x, t_mean, axis);
    var(x, t_var, axis);

    // Let's traverse the entire input using flat index
    // and compute y = (x-u) / sqrt(var + eps) 
    int total_elems = x->dim[0] * x->strides[0];
    for (int i = 0; i < total_elems; i++) {
        int rem = i;
        int out_flat = 0;
        int axis_coord = 0; // We need to save this to look up gamma / beta

        for (int d = 0; d < x->ndim; d++) {
            // Figure out the coordinate for dimension "d"
            int coord = rem / x->strides[d];

            // Strip that dimension out of the remainder
            rem = rem % x->strides[d];

            if (d == axis) {
                // If we are at the reduction axis, save this coordinate
                axis_coord = coord;
            } else {
                // If we are not at the reduction axis, this coordinate helps
                // build the flat index for mean/var buckets
                out_flat += coord * t_mean->strides[d];
            }
        }

    }

    tensor_sub(x, mean, y);
    tensor_add_scalar(var, epsilon, var);
    // TODO: I need an sqrt() tensor operator here
    tensor_div(y, var, y);

    // TODO: multiply(but which one? matmul?) by gamma and sum beta


    // TODO: Free every malloc()
}
