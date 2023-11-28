from math import fsum
from timeit import timeit

import numpy as np

import fast_math as fm

if __name__ == "__main__":
    arrays = {
        "ones": np.ones(1_000_000, dtype=np.float32),
        "ascending": np.arange(1_000_000, dtype=np.float32),
        "descending": np.arange(1_000_000, dtype=np.float32)[::-1],
        "rounding": np.array([2**24, 1, -(2**24)] * 300_000, dtype=np.float32),
    }

    num_runs = 100
    for name, array in arrays.items():
        print(f"Benchmark: {name}...")
        fast_time = timeit(
            "func(array)", number=num_runs, globals={"func": fm.sum, "array": array}
        )
        print(f"\tfast_math: {fast_time:.3f}s")
        numpy_time = timeit(
            "func(array)", number=num_runs, globals={"func": np.sum, "array": array}
        )
        print(f"\t    numpy: {numpy_time:.3f}s ({fast_time / numpy_time:.1f}x slower)")
        fsum_time = timeit(
            "func(array)", number=num_runs, globals={"func": fsum, "array": array}
        )
        print(f"\t     fsum: {fsum_time:.3f}s ({fsum_time / fast_time:.1f}x faster)")
