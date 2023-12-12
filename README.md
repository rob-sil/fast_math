# fast_math

An accurate replacement for NumPy's `sum`.

## Floating-Point Rounding Errors

Floating-point addition is subject to **rounding error**, where the result loses some precision when stored in the fixed bits of a `np.float32` or `np.float64`. For a stylized example:

```python
>>> import numpy as np
>>> array = np.array([2**24, 1, -2**24], dtype=np.float32)
>>> np.sum(array)
0.0
```

The correct answer is $2^{24} + 1 - 2^{24} = 1$, but the sum gets it wrong because 32-bit floats can't represent the number $2^{24} + 1$ (under IEEE 754). A 32-bit float only has 23 bits for significant digits, so a 32-bit float with a one in the $2^{24}$ place doesn't have the precision to store anything in a $2^0$ place. When adding one, floating-point math has to round it away: $2^{24} + 1 = 2^{24}$. These rounding errors accumulate in `numpy.sum`, which may return a value different from the 32-bit representation of the sum (see `numpy.sum`'s [Notes](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)).

### Rounding Errors in Big Data

Rounding error is a particular problem when working with a large number of floating-point numbers, as we do in modern scientific computing or machine learning. While the previous example was quite stylized, the same problem arises when working with a large number of values.

```python
>>> array = np.arange(1_000_000, dtype=np.float32)
>>> np.sum(array)
499999800000.0
>>> np.float32(np.sum(array, dtype=np.float64))
499999500000.0
```

*Note*: In some cases, `numpy.sum` will use [pairwise summation](https://en.wikipedia.org/wiki/Pairwise_summation), which reduces the amount of rounding error. For example, summing $2^{25}$ ones will be accurate and summing $2^{25} + 1$ ones will only be off by one. However, when when Numpy doesn't use pairwise summation, summing $2^{25}$ ones gets stuck at $2^{24}$.

### `math.fsum` and Performance

The Python standard library implements [`math.fsum`](https://docs.python.org/3/library/math.html#math.fsum), which calculates the sum of floating-point numbers accurately. However, `fsum` is significantly slower than `numpy.sum`,  possibly 100-200 times slower, which may not be practical for scientific computing on large arrays.

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

`fast_math.sum` runs about five times slower than `numpy.sum`. It outperforms NumPy on small arrays and when summing over short axes, however it runs much slower when summing over long axes.

Currently, `fast_math.sum` only supports summing over a single axis.

## References

Summation is based on the work of:
1. Yong-Kang Zhu and [Wayne B. Hayes](https://www.cs.toronto.edu/~wayne/), particularly "Algorithm 908: Online Exact Summation of Floating-Point Streams" (2010).
2. [Jonathan Shewchunk](https://people.eecs.berkeley.edu/~jrs/), particularly "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (1997).
