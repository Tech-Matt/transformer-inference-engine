#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ops.h"
#include "tensor.h"

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
        int coords[MAX_NDIM];
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
        int out_coords[MAX_NDIM];
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
void var(Arena *scratch, const Tensor* in, Tensor *out, int axis) {
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

    // Save the Arena state!
    size_t bookmark = arena_get_bookmark(scratch);
    float *mean_buf = arena_alloc(scratch, sizeof(float) * out_elems);
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

    // Restore the arena state
    arena_free_to_bookmark(scratch, bookmark);
}


/**
 * @brief Layer Normalization function
 *
 * Performs layer normalization on the input tensor `x` and stores the result in `y`.
 * The normalization is performed along the specified `axis` using the learned
 * scale (`gamma`) and shift (`beta`) parameters.
 *
 * Formula: y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
 *
 * @param x Input tensor to be normalized.
 * @param y Output tensor (must have same shape as x).
 * @param gamma Scale parameter tensor (1D).
 * @param beta Shift parameter tensor (1D).
 * @param epsilon Small constant for numerical stability.
 * @param axis Axis along which to normalize.
 */
void layer_norm(Arena *scratch, const Tensor *x, Tensor *y, const Tensor *gamma, const Tensor *beta, float epsilon, int axis) {
     // --- Argument validation ---
     assert(x->ndim == y->ndim && "layer_norm: ndim mismatch");
     assert(gamma->ndim == 1 && beta->ndim == 1);
     assert(axis >= 0 && axis < x->ndim && "layer_norm: axis out of bounds");
     for (int i = 0; i < x->ndim; i++) {
         assert(x->dim[i] == y->dim[i] && "layer_norm: dim mismatch");
     }

     // Save arena state before allocation
     size_t bookmark = arena_get_bookmark(scratch);

     /* --- Allocate temporary tensors for mean and variance ---
      * Shape: same as input but with the reduction axis compressed to size 1.
      * For input shape [batch, seq, hidden] and axis=2 (hidden),
      * t_mean and t_var will have shape [batch, seq, 1].
      */
     float *data_mean = arena_alloc(scratch, sizeof(float) * x->dim[0] * x->strides[0]);
     int dim_mean[MAX_NDIM];
     memcpy(dim_mean, x->dim, sizeof(int) * x->ndim);
     dim_mean[axis] = 1;
     Tensor *t_mean = Tensor_new(scratch, x->ndim, dim_mean, data_mean);

     float *data_var = arena_alloc(scratch, sizeof(float) * x->dim[0] * x->strides[0]);
     int dim_var[MAX_NDIM];
     memcpy(dim_var, x->dim, sizeof(int) * x->ndim);
     dim_var[axis] = 1;
     Tensor *t_var = Tensor_new(scratch, x->ndim, dim_var, data_var);

     /* --- Step 1: Compute mean and variance per token ---
      * After this, for each token (position along non-axis dimensions),
      * there is exactly ONE mean value and ONE variance value stored at
      * t_mean->data[out_flat] and t_var->data[out_flat].
      */
     mean(x, t_mean, axis);
     var(scratch, x, t_var, axis);

     /* --- Step 2: Normalize every element and apply affine transform --- */
     int total_elems = x->dim[0] * x->strides[0];

     // Iterate over every element in the input tensor x
     for (int i = 0; i < total_elems; i++) {
         int rem = i;
         int out_flat = 0;       // will become the flat index into t_mean/t_var
         int axis_coord = 0;     // will become the index along the normalization axis
                                 // (used to index gamma/beta)

         /* --- Coordinate decomposition ---
          * Convert flat index i into multi-dimensional coordinates.
          * Example for shape [2,3,4] with strides [12,4,1]:
          *   i = 23  ->  batch=1, seq=2, hidden=3
          * Then:
          *   - For dimensions NOT equal to axis (batch, seq), build out_flat
          *   - For dimension == axis (hidden), save axis_coord
          */
         for (int d = 0; d < x->ndim; d++) {
             int coord = rem / x->strides[d];  // Coordinate in dimension d
             rem = rem % x->strides[d];        // Remainder for next dimensions

             if (d == axis) {
                 axis_coord = coord;            // Hidden feature index (to access gamma and beta)
             } else {
                 out_flat += coord * t_mean->strides[d];  // Build mean/var index
             }
         }

         /* --- Layer Norm formula for this element ---
          * 1. Normalize: (x - μ) / √(σ² + ε)
          * 2. Apply affine: γ * normalized + β
          *
          * Access patterns:
          *   x->data[i]        -> input value at flat index i
          *   t_mean->data[out_flat]  -> mean for this token
          *   t_var->data[out_flat]   -> variance for this token
          *   gamma->data[axis_coord] -> scale for this hidden feature
          *   beta->data[axis_coord]  -> shift for this hidden feature
          */
         float x_val = x->data[i];
         float mean_val = t_mean->data[out_flat];
         float var_val = t_var->data[out_flat];

         float normalized = (x_val - mean_val) / sqrtf(var_val + epsilon);
         float y_val = gamma->data[axis_coord] * normalized + beta->data[axis_coord];

         y->data[i] = y_val;
     }

     // Free data structures from Arena
     arena_free_to_bookmark(scratch, bookmark);
 }
