#include <assert.h>
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


void mean(const Tensor *in, Tensor *out, int axis) {
    // Check ndim
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

    // TODO: Zero-initialize the output data buffer? (do I really need this?)


    // TODO: Reduction loop
    // Loop over every element in the input tensor
    // Write unchanged dimensions back to out
    // accumulate sum of reduced dimension from in and write sum
    // to out.

    // TODO: Final division by in->dim[axis]

}


void var(const Tensor* in, Tensor *out, int axis) {
    // TODO: Assertions

    // TODO: Calculate mean first
    // You can't compute variance without mean!
    
    // TODO: Calculate squared differences
    
    // TODO: Final division
}
