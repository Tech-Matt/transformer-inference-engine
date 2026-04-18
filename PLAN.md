# Project Plan

Building a pure-C transformer inference engine. Inference only. Target: ARM Cortex-M.

## Completed
- **Build System:** Strict `Makefile`.
- **Tensor Primitives:** N-dimensional tensors, stride-based indexing, matrix multiplication, element-wise math.
- **Activations:** `relu`, `softmax`, `log_softmax`, `mean`, `var`.
- **Testing:** Unity framework integration. Full coverage of tensor math and edge cases.

## Next Up: Memory Arena
- [ ] **Build an Arena Allocator:** 
  - Create a pre-allocated memory pool.
  - Route tensor creations through the arena instead of `malloc`.
  - Reset the arena at the end of every inference pass (zero garbage collection overhead).

## Future Phases
- [ ] **Transformer Operations:** Layer Normalization.
- [ ] **Transformer Blocks:** Self-Attention, Positional Encoding, Feed-Forward Networks.
- [ ] **Model Composition:** Config structs and block routing for Encoder/Decoder architectures.
- [ ] **Weight Loading:** Parse and load binary weights from disk into the Arena.
- [ ] **Embedded Optimizations:** CMSIS-DSP SIMD acceleration, INT8 quantization footprint reduction.
