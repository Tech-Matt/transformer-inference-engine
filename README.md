# transformer-inference-engine

A pure C implementation of a transformer inference engine targeting embedded systems (Cortex-M).

<img width="421" height="518" alt="image" src="https://github.com/user-attachments/assets/8918d85a-dca7-4289-b14e-a69fa2664374" />

[Reference: Attention is All You Need (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762)

## Current Status

### Completed
- **Phase 0: Build System** — Makefile with `-Wall -Wextra -Werror`
- **Phase 1: Tensor Primitive** — N-dimensional tensors with stride-based indexing
  - Constructors: `Tensor_empty()`, `Tensor_new()`
  - Strides computation for arbitrary shapes
  - Memory management: `Tensor_free()`
- **Phase 2: Tensor Operations** — Core array arithmetic
  - Element-wise: add, sub, mul, div (tensor-tensor and tensor-scalar)
  - Matrix multiply: `tensor_matmul()`
  - Transpose: `tensor_transpose()`
- **Phase 3: Activation Functions & Math Ops** — Neural network primitives
  - ReLU, softmax, log_softmax
  - Mean reduction (`mean(Tensor *, int axis)`)
  - Variance reduction (`var(Tensor *, int axis)`)

### In Progress
- **Phase 3.5: Test Infrastructure** — Golden-value tests using NumPy/PyTorch reference outputs
- **Phase 3 Final: Layer Normalization** — Depends on test suite completion

### Planned
- **Phase 4: Transformer Building Blocks** — Embeddings, positional encoding, attention, FFN
- **Phase 5: Model Composition & Config** — Encoder/decoder blocks with compile-time modularity
- **Phase 6: Weight Loading** — Binary format and HuggingFace safetensors compatibility
- **Phase 7: Embedded Optimization** — Static allocation, INT8 quantization, ARM CMSIS-NN integration

## How to Build

```bash
make          # Build engine executable
make clean    # Remove binary
make test     # Build and run all tests (once Phase 3.5 is complete)
```

## Project Notes

- **Pure C**: No C++, no external ML frameworks. Self-contained inference.
- **Inference only**: No training. Focus on numerical correctness and memory efficiency.
- **Embedded target**: Currently uses heap allocation for simplicity; will migrate to static/stack allocation for Cortex-M deployment.
- **Architecture**: Compile-time modularity via `#ifdef` flags (e.g., `ENCODER_ONLY`, `DECODER_ONLY`).

For detailed roadmap, see [PLAN.md](PLAN.md).
