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
tinymlkit <model-name> -json_data /path/to/json/data <model-options>
# prediction
tinymlkit <model-name> -json_data /path/to/json/data <model-loading-options>
``` 

Because of the nature of TinyGo, it may end up that each model is its own "thing" that is built. We'll deal with that later, and perhaps use a code generator to switch between Go and TinyGo. 

The format of the data is in `json` so that it can be serialised/deserialised in go easily.

```
{
	"Features": [[1,2,3],[4,5,6]],
	"Label": [7,8]
}
```

## Roadmap

1. Initial release targetting Go (with one algorithm?)
2. Target TinyGo with `wasm` support
3. Add algorithms

Later stage: export to `onnx` if this becomes a thing.

**Algorithms**

The purpose of this project isn't machine learning, its for me to learn Go and get acquainted with WASM. As such, we're going to concentrate on a small set of algorithms and see how far we go:

- Randomised Optimisation
- Non-parametric bounds - Hoeffding inequality for drawing inferrences on streams

... other optimisation algorithms as we go, perhaps a lightweight auto-diff implementation for example

## Inspirations

- [Vowpal Wabbit](https://vowpalwabbit.org/)
- [River](https://github.com/online-ml/river)