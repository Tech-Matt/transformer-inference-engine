# Project Plan

Building a pure-C transformer inference engine. Inference only. Target: ARM Cortex-M.

## Completed
- **Build System:** Strict `Makefile`.
- **Tensor Primitives:** N-dimensional tensors, stride-based indexing, matrix multiplication, element-wise math.
- **Activations:** `relu`, `softmax`, `log_softmax`, `mean`, `var`.
- **Testing:** Unity framework integration. Full coverage of tensor math and edge cases.
- **Doxygen**: Documentation framework added to the repo and configured.

## IN PROGRESS
- [ ] **Build an Arena Allocator:** 
  - Route tensor creations through the arena instead of `malloc`.

## Future Phases
- [ ] **Documentation**: Add or correct documentation in missing headers and files.
- [ ] **Correct variable sizes**: Port `int` and `float` over the codebase into definitions in `<stdint.h>`
- [ ] **Transformer Blocks:** Self-Attention, Positional Encoding, Feed-Forward Networks.
- [ ] **Model Composition:** Config structs and block routing for Encoder/Decoder architectures.
- [ ] **Weight Loading:** Parse and load binary weights from disk into the Arena.
- [ ] **Different Quantizations** Introduce A DataType enum and put it into the tensor struct. Once you have that, we need to change the implemented functions to use an opaque void * pointer instead of a hardcoded float *, so that we can implement operation for int8, int16, float32, etc.
- [ ] **Embedded Optimizations:** CMSIS-DSP SIMD acceleration, INT8 quantization footprint reduction.
