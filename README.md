# transformer-inference-engine

A pure C implementation of a transformer inference engine, built from scratch with embedded systems in mind.

[Reference: Attention is All You Need (Vaswani et al. 2017)](https://arxiv.org/pdf/1706.03762)


## Building & Testing

```bash
make          # Build engine
make test     # Run the test suite
make clean    # Clean binaries
```

## Possible Future Optimizations
- Introduce different levels of quantization
- Aproximate [LayerNorm](https://ieeexplore.ieee.org/abstract/document/11373553)

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