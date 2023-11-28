from math import fsum
from timeit import timeit

import numpy as np

import fast_math as fm

if __name__ == "__main__":
    array_constructors = {
        "ones": lambda N: np.ones(N, dtype=np.float32),
        "ascending": lambda N: np.arange(N, dtype=np.float32),
        "descending": lambda N: np.arange(N, dtype=np.float32)[::-1],
        "rounding": lambda N: np.array(
            [2**24, 1, -(2**24)] * (N // 3), dtype=np.float32
        ),
    }

    fast_total = 0
    numpy_total = 0
    fsum_total = 0

    num_runs = 100
    for name, constructor in array_constructors.items():
        print(f"Benchmark: {name}...")
        for N in [100, 10_000, 100_000, 1_000_000]:
            print(f"\t|array|={N}")
            array = constructor(N)
            fast_time = timeit(
                "func(array)", number=num_runs, globals={"func": fm.sum, "array": array}
            )
            fast_total += fast_time
            print(f"\tfast_math: {fast_time:.3f}s")

            numpy_time = timeit(
                "func(array)", number=num_runs, globals={"func": np.sum, "array": array}
            )
            numpy_total += numpy_time
            print(
                f"\t    numpy: {numpy_time:.3f}s ({fast_time / numpy_time:.1f}x slower)"
            )

            fsum_time = timeit(
                "func(array)", number=num_runs, globals={"func": fsum, "array": array}
            )
            fsum_total += fsum_time
            print(
                f"\t     fsum: {fsum_time:.3f}s ({fsum_time / fast_time:.1f}x faster)"
            )

    with open("benchmark.md", "w") as f:
        f.write(
            "# Benchmark Results\n"
            f"`fast_math` is {fast_total / numpy_total:.1f} times slower than `numpy`.\n\n"
            f"`fast_math` is {fsum_total / fast_total:.1f} times faster than `math.fsum`."
        )
