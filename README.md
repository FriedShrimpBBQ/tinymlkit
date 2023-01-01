# TinyMLKit

Inspired by TinyML libraries like [river](https://github.com/online-ml/river), this is an experimental library to bring tabular machine learning to WebAssembly via [TinyGo](tinygo.org/), and as a side benefit maybe embedded systems.

This library is purely just for fun - not expecting it to actually be competitive with other machine learning libraries in performance or latency.

**Goals**

- Readable TinyGo compatible code
- Target WASM by default
- Support [online machine learning](https://www.wikiwand.com/en/Online_machine_learning)

**Non-Goals**

- Support deep learning
- Support offline or batch machine learning algorithms

## Design Notes

To keep things simple, we will target CLI only. The interface would resemble:

```sh
# training
tinymlkit <model-name> -d <data.txt> -f <model.tmk>
# prediction
tinymlkit <model-name> -d <data.txt> -i <model.tmk> -p <output.txt>
``` 

Because of the nature of TinyGo, it may end up that each model is its own "thing" that is built. We'll deal with that later, and perhaps use a code generator to switch between Go and TinyGo. 

The format of `data.txt` should simply be sparse arrays in format:

```
<label> | <index>:value
```

E.g. if our input `X` is `[1, 20, -3, 4]` and `y` is [2], then the input is:

```
2 | 0:1 1:20 2:-3 3:4
```

This naturally supports sparse formats where elements with the value 0 can be dropped.

**Federated Learning**

As we are using Nelder-Mead method for choosing the weights, and keeping a history of the set weights, we can offer Byzantine resilience when performing federated learning. 

Possible algorithms to support aggregated weights include:

- Coordinate-Wide Median
- Geometric Mean
- Mean around median

Proposed interface:

```sh
tinymlkit <model-name> -m <federated_model.tmk> -m <federated_model.tmk> -m <federated_model.tmk> -m <federated_model.tmk> -f <model.tmk>
```

Note that the `federated_model.tmk` file may contain summarised information about the underlying model. If `model.tmk` does not exist, it will be created through aggregation. 


## Roadmap

1. Initial release targetting Go (with one algorithm?)
2. Target TinyGo with `wasm` support
3. Add algorithms

Later stage: export to `onnx` if this becomes a thing.

**Algorithms**

The purpose of this project isn't machine learning, its for me to learn Go and get acquainted with WASM. As such, we're going to concentrate on a small set of algorithms and see how far we go:

- Derivative free optimization approaches - [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)) which can be applied to a variety of contexts
- Non-parametric bounds - Hoeffding inequality for drawing inferrences on streams

## Inspirations

- [Vowpal Wabbit](https://vowpalwabbit.org/)
- [River](https://github.com/online-ml/river)