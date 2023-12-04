from timeit import timeit
from typing import List, Tuple

import numpy as np

import fast_math as fm

benchmarks: List[Tuple[str, str]] = [
    (
        "np.ones({N}, dtype=np.float32)",
        "sum(array)",
    ),
    (
        "np.arange({N}, dtype=np.float32)",
        "sum(array)",
    ),
    (
        "np.arange({N}, dtype=np.float32)[::-1]",
        "sum(array)",
    ),
    (
        "np.array([2**24, 1, -(2**24)] * ({N} // 3), dtype=np.float32)",
        "sum(array)",
    ),
    (
        "np.ones({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        "sum(array)",
    ),
    (
        "np.arange({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        "sum(array)",
    ),
    (
        "np.ones({N} // 25 * 25, dtype=np.float32).reshape((-1, 5, 5))",
        "sum(array)",
    ),
    (
        "np.arange({N} // 25 * 25, dtype=np.float32).reshape((-1, 5, 5))",
        "sum(array)",
    ),
]

sizes = (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)


def time_format(seconds) -> str:
    """Format a runtime."""
    if seconds < 1e-6:
        return f"{seconds / 1e-9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds / 1e-6:.2f} us"
    elif seconds < 1:
        return f"{seconds / 1e-3:.2f} ms"
    else:
        return f"{seconds:.2f}  s"


if __name__ == "__main__":
    results = []

    num_runs = 1_000
    for setup_format, statement in benchmarks:
        for N in sizes:
            setup = "array = " + setup_format.format(N=N)

            print(f"> {setup}\n> {statement}")

            fast_time = timeit(
                statement, setup, number=num_runs, globals={"sum": fm.sum, "np": np}
            )
            print(f"\tfast_math: {fast_time:.3f}s")

            numpy_time = timeit(
                statement, setup, number=num_runs, globals={"sum": np.sum, "np": np}
            )
            print(
                f"\t    numpy: {numpy_time:.3f}s ({fast_time / numpy_time:.1f}x slower)"
            )

            results.append(
                {
                    "N": N,
                    "fast_math": fast_time,
                    "numpy": numpy_time,
                }
            )

            print()

    markdown = "# Benchmark Results\n"

    markdown += "| Array Size | `fast_math` | NumPy | Slowdown |\n"
    markdown += "| -: | -: | -: | -: |\n"
    for N in sizes:
        fast_time = np.mean(
            [result["fast_math"] for result in results if result["N"] == N]
        )
        numpy_time = np.mean(
            [result["numpy"] for result in results if result["N"] == N]
        )

        markdown += (
            f"| {N:,} | {time_format(fast_time)} | {time_format(numpy_time)} "
            f"| {fast_time / numpy_time:.1f}x |\n"
        )

    with open("benchmark.md", "w") as f:
        f.write(markdown)
