# fast_math

A fast and accurate replacement for NumPy's `sum` and `cumsum`.

## Floating-Point Arithmetic

Floating-point arithmetic is prone to errors, especially when multiple operations are performed in a row. Each individual operation can introduce a bit of rounding error, which accumulates over the course of the calculation. Usually, floating-point operations are good enough, and the rounding error is acceptably small. Unfortunately, not every use case can tolerate rounding error. For real-world examples of rounding error problems, check out [Julia Evan's *Examples of floating point problems*](https://jvns.ca/blog/2023/01/13/examples-of-floating-point-problems/).

```python
>>> import numpy as np
>>> np.float32(2/3) * 3 == 2
False
```

### Summation

Summing a list of numbers is one of the most common calculations using many floating point operations in a row. Each addition can introduce a bit of rounding error along the way, especially when the numbers in the list vary by magnitude. Even when the numbers have similar magnitudes, summing a long list of numbers can end up adding a large accumulated sum to a much smaller value, rounding it mostly away.

```python
>>> import numpy as np
>>> array = np.array([2**24, 1, -2**24], dtype=np.float32)
>>> np.sum(array) # Should be 1
0.0
```

Notably, NumPy's summation does not guarantee that its result is the closest floating-point number to the actual sum (See `numpy.sum`'s [Notes](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)). This package aims to provide a replacement for `sum` and `cumsum` that calculate the accurate sum, and return the sum rounded to the closest floating-point number.

### `math.fsum` Performance

The Python standard library provides [`math.fsum`](https://docs.python.org/3/library/math.html#math.fsum), which mostly calculates the sum of floating-point numbers accurately (up to possible error in the last digit). Unfortunately, the accuracy of `fsum` comes at a significant performance price, running 100-200 times slower than NumPy's `sum`. That big slowdown makes a difference when summing a large array, the exact case when floating-point error is the worst!

## Installation

`fast_math` can be installed with Rust and [maturin](https://www.maturin.rs) by running [pip](https://pip.pypa.io/en/stable/) in the root directory.

## Usage

```python
>>> import numpy as np
>>> import fast_math as fm
>>> array = np.array([2**24, 1, -2**24], dtype=np.float32)
>>> fm.sum(array)
1.0
```

## Status

`fast_math` only implements two functions, `sum` and `cumsum`, for 32-bit floats.
- `fast_math.cumsum` runs 1-3x slower than the `numpy.cumsum`.
- `fast_math.sum` runs about five times slower than `numpy.sum` on average. It outperforms NumPy on small arrays and when summing over short axes, but runs much slower when summing over long axes.

`fast_math.sum` does not support summing over multiple axes at once.

## References

Summation is based on the work of:
1. Yong-Kang Zhu and [Wayne B. Hayes](https://www.cs.toronto.edu/~wayne/), particularly "Algorithm 908: Online Exact Summation of Floating-Point Streams" (2010).
2. [Jonathan Shewchunk](https://people.eecs.berkeley.edu/~jrs/), particularly "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (1997).
