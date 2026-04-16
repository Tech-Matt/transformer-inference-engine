// ops.h
// common math operations performed in transformer neural networks
// (For non-matrix operations like sqrt(), and other operations related to scalars, i am just going to use math.h for now)

// List of required operations to be implemented
// - Mean() on Tensors 
// - Std() on Tensors
// - Relu() on Tensors
// - softmax() on Tensors
// - log_softmax()

#ifndef OPS_H
#define OPS_H

#include <math.h>
#include "tensor.h"


void relu(const Tensor *in, Tensor *out);
void softmax(const Tensor *in, Tensor *out);
void log_softmax(const Tensor *in, Tensor *out);
void mean(const Tensor *in, Tensor * out, int axis);
void var(const Tensor *in, Tensor * out, int axis);

#endif
