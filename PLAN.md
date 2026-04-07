# Transformer Inference Engine — Project Plan

## Context
Building a pure-C transformer inference engine with the long-term goal of running on ARM Cortex-M embedded hardware. Inference only (no training). General-purpose: eventually load models from HuggingFace. MVP first for professor, then optimize incrementally.

Architecture approach: compile-time modularity via `#ifdef` flags. Start with shared primitives, add block-level composition later. Full encoder-decoder as baseline, then configure for decoder-only or encoder-only via flags.

This file is the living project roadmap — update it as phases are completed.

---

## Phase 0: Build System (DONE)
- [x] Create a `Makefile` in project root
  - Targets: `all`, `clean`
  - Compile `src/*.c` with `include/` on the include path
  - Use `-Wall -Wextra -g` for development
  - Add a placeholder for future `-DDECODER_ONLY` style flags

---

## Phase 1: Tensor Primitive (DONE)
Critical foundation — everything else depends on this.

### Files
- `include/tensor.h`
- `src/tensor.c`

### Tasks
- [x] Fix `Tensor_empty()` — `data_buffer_size` is wrong (should be product of all dims, not `ndim * DEFAULT_DIM`)
- [x] Implement strides computation in `Tensor_empty()` (strides[i] = product of dim[i+1..ndim-1])
- [x] Implement `Tensor_new(int ndim, int *dim, float *data)` — full constructor
- [x] Add `Tensor_free()` — free `data`, `dim`, `strides`, then the struct itself
- [x] Declare `Tensor_free()` in `tensor.h`
- [x] Add `#include <stdlib.h>` to `tensor.c` (needed for malloc/free)

### Key concept: Strides
For a tensor of shape [d0, d1, d2]:
- strides[2] = 1
- strides[1] = d2
- strides[0] = d2 * d1
Element at [i][j][k] → data[i*strides[0] + j*strides[1] + k*strides[2]]

---

## Phase 2: Tensor Operations
Files: `include/tensor.h`, `src/tensor.c`
Pure tensor mechanics — no ML semantics, just array arithmetic.

- [ ] Element-wise: `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div` (tensor-tensor and tensor-scalar)
- [ ] Matrix multiply: `tensor_matmul` (the core operation of attention)
- [ ] Tensor transpose: `tensor_transpose`

---

## Phase 3: Activation Functions & Math Ops
Files: `include/ops.h`, `src/ops.c`
Neural network math operating on Tensors. Separated from tensor mechanics by design — changes here never require touching `tensor.h`.

- [ ] `relu(Tensor *)` — element-wise max(0, x)
- [ ] `softmax(Tensor *)` — numerically stable version (subtract max before exp)
- [ ] `log_softmax(Tensor *)` — for output layer
- [ ] `mean(Tensor *)` and `std(Tensor *)` — needed for layer normalization

---

## Phase 4: Transformer Building Blocks
Each block maps cleanly to the "Attention is All You Need" paper (Vaswani et al. 2017).

- [ ] Layer Normalization
- [ ] Input Embeddings (`include/embedding.h`)
- [ ] Positional Encoding (`include/pos_embedding.h`) — sinusoidal, uses math.h sin/cos
- [ ] Scaled Dot-Product Self-Attention
- [ ] Multi-Head Attention
- [ ] Feed-Forward Network (Linear → ReLU → Linear)
- [ ] Residual Connections (just tensor_add, but wired correctly)

---

## Phase 5: Model Composition & Compile-Time Config
- [ ] Encoder block (`encoder_block.h/.c`)
- [ ] Decoder block (`decoder_block.h/.c`) — adds masked self-attention + cross-attention
- [ ] Compile-time flags: `#ifdef ENCODER_ONLY`, `#ifdef DECODER_ONLY`, `#ifdef ENCODER_DECODER`
- [ ] Model config struct (for passing dimensions: d_model, n_heads, n_layers, d_ff, vocab_size)

---

## Phase 6: Weight Loading
- [ ] Define a binary weight file format
- [ ] Implement weight loader from file into Tensor structs
- [ ] (Later) HuggingFace safetensors compatibility

---

## Phase 7: Embedded Optimization (Post-MVP)
- [ ] Replace heap allocation with static/stack allocation
- [ ] Evaluate fixed-point / quantization (INT8) for Cortex-M targets without FPU
- [ ] Leverage ARM CMSIS-DSP for SIMD-accelerated matmul on Cortex-M4/M7
- [ ] Minimize memory footprint — in-place operations where possible

---

## Phase 0.5: Testing Infrastructure
C has no built-in test runner, but there are good options. Recommendation: **Unity** — a pure C, single-file testing framework designed for embedded systems. Used industry-wide on Cortex-M projects.

- [ ] Add Unity as a dependency (`tests/unity/`) — single `.c` + `.h` file, no build system magic
- [ ] Create `tests/` directory with a `test_tensor.c` to start
- [ ] Add a `test` target to the Makefile
- [ ] Pattern: each test file has a `main()` that calls `UNITY_BEGIN()`, runs test functions via `RUN_TEST()`, then `UNITY_END()`

### What to test
- After Phase 1: `test_tensor.c` — construct tensors, verify strides, index elements correctly
- After Phase 2: `test_ops.c` — matmul against known outputs, element-wise ops
- After Phase 3: `test_activations.c` — softmax sums to 1.0, relu clamps correctly
- After Phase 4+: `test_attention.c` — forward pass of a tiny hand-initialized model vs PyTorch reference values

### Reference values strategy
For numerical correctness: write a small Python/PyTorch script that computes expected outputs for known inputs, hardcode those as expected values in C tests. This is the standard approach for validating inference engines.

---

## Verification Strategy
- After Phase 1: manually construct a known tensor, index into it using strides, verify correctness
- After Phase 2: matmul of known matrices, compare output against Python/numpy reference
- After Phase 3: softmax of known vector, verify sums to 1.0
- After Phase 4+: run a forward pass of a tiny hand-initialized model, compare layer outputs to PyTorch reference
