# transformer-inference-engine

A pure C implementation of a transformer inference engine, built from scratch with embedded systems in mind.

[Reference: Attention is All You Need (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762)

## What's working?
- Core N-dimensional tensor arithmetic and stride manipulations.
- Neural network primitives (Matrix multiplication, ReLU, stable Softmax, log-softmax, reductions).
- Unity-based testing harness 

## What's next?
- **Memory Arena Allocator:** Replacing dynamic `malloc`/`free` calls with a static memory pool for zero-allocation forward passes.
- **Transformer Building Blocks:** Layer Normalization, Attention, and Feed-Forward networks.

## Building & Testing

```bash
make          # Build engine
make test     # Run the test suite
make clean    # Clean binaries
```

See [PLAN.md](PLAN.md) for the full architectural roadmap.
