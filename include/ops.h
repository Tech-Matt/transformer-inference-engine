/**
 * @file    ops.h
 * @author  Mattia Rizzo (Tech-Matt)
 * @brief   Common math operations performed in transformer neural networks
 * @license MIT License
 *
 * This file contains function prototypes for non-matrix mathematical
 * operations required during inference, such as activation functions
 * (ReLU, Softmax) and statistical reductions (Mean, Variance).
 */

#ifndef OPS_H
#define OPS_H

#include <math.h>
#include "tensor.h"


void relu(const Tensor *in, Tensor *out);
void softmax(const Tensor *in, Tensor *out);
void log_softmax(const Tensor *in, Tensor *out);
void mean(const Tensor *in, Tensor * out, int axis);
void var(Arena *scratch, const Tensor *in, Tensor * out, int axis);
void layer_norm(Arena *scratch, const Tensor *in, Tensor *out, const Tensor *gamma, const Tensor *beta, float epsilon, int axis);

#endif
