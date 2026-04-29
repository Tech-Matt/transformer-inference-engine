# transformer-inference-engine

A pure C implementation of a transformer inference engine, built from scratch with embedded systems in mind.

[Reference: Attention is All You Need (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762)


## Building & Testing

```bash
make          # Build engine
make test     # Run the test suite
make clean    # Clean binaries
```

## Results
- Performance Plots with some tiny Transformers

## Possible Future Optimizations
- Introduce different levels of quantization
- Approximate [LayerNorm](https://ieeexplore.ieee.org/abstract/document/11373553)
- Alignment is also important. Apparently aligning the arena allocation at every 16 bytes, makes it possible to use SIMD instructions (to be verified)
- The arena allocator can be used in a sort of "wrapping" mode. Once the memory is full with the activations of the next layer, the previous layer activations are useless and can be freed from the memory block, thus releasing memory for the next layer. This makes it possible to execute models with a dynamic size larger than the memory of the device RAM. The max usage of RAM can be statically analyzed on an "offline" planning phase, like frameworks like Tensorflow are already doing.

## Useful Papers to be read
- https://arxiv.org/abs/1802.04799
- https://people.csail.mit.edu/jrk/halide-pldi13.pdf
- [MCUnet Tiny DeepL](https://arxiv.org/abs/2007.10319)
- [TensorFlow Lite Micro](https://arxiv.org/abs/2010.08678)
- https://proceedings.mlr.press/v97/gural19a/gural19a.pdf
- [Internal Org of Numpy arrays](https://numpy.org/doc/stable/dev/internals.html)

## Documentation
- To be created in the future
- Right now I am making every file "Doxygen" compliant to easily create an HTML or PDF documentation


## Archive (Papers and Resources)