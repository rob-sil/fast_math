from collections import defaultdict
from dataclasses import dataclass, field
from timeit import timeit
from typing import Dict, List

import numpy as np

import fast_math as fm


@dataclass
class Benchmark:
    setup: str
    """Format string to create an array of size `N`."""
    statement: str
    """Statement to run, taking in the array created by `setup`."""
    tags: Dict[str, str] = field(default_factory=dict)
    """Optional tags to print result breakdowns."""

    def timeit(self, N: int, sum, num_runs: int = 100) -> float:
        """Time the benchmark for sum function `sum` on an array of roughly
        `N` elements.
        """
        return (
            timeit(
                self.statement,
                "array = " + self.setup.format(N=N),
                number=num_runs,
                globals={"sum": sum, "np": np},
            )
            / num_runs
        )


benchmarks: List[Benchmark] = [
    Benchmark(
        setup="np.ones({N}, dtype=np.float32)",
        statement="sum(array)",
        tags={"Dimensions": "1", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N}, dtype=np.float32)",
        statement="sum(array)",
        tags={"Dimensions": "1", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N}, dtype=np.float32)[::-1]",
        statement="sum(array)",
        tags={"Dimensions": "1", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.array([2**24, 1, -(2**24)] * ({N} // 3), dtype=np.float32)",
        statement="sum(array)",
        tags={"Dimensions": "1", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N}, dtype=np.float32)[::-1]",
        statement="sum(array)",
        tags={"Dimensions": "1", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        statement="sum(array)",
        tags={"Dimensions": "2", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        statement="sum(array)",
        tags={"Dimensions": "2", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones({N} // 25 * 25, dtype=np.float32).reshape((-1, 5, 5))",
        statement="sum(array)",
        tags={"Dimensions": "3", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N} // 25 * 25, dtype=np.float32).reshape((-1, 5, 5))",
        statement="sum(array)",
        tags={"Dimensions": "3", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones({N} // 5 * 5, dtype=np.float32).reshape((5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "2", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N} // 5 * 5, dtype=np.float32).reshape((5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "2", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones({N} // 25 * 25, dtype=np.float32).reshape((5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "3", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange({N} // 25 * 25, dtype=np.float32).reshape((5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "3", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones(({N} // 125 + 1) * 125, dtype=np.float32).reshape((5, 5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "4", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange(({N} // 125 + 1) * 125, dtype=np.float32).reshape((5, 5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "4", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones(({N} // 625 + 1) * 625, dtype=np.float32).reshape((5, 5, 5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "5", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.arange(({N} // 625 + 1) * 625, dtype=np.float32).reshape((5, 5, 5, 5, -1))",
        statement="sum(array)",
        tags={"Dimensions": "5", "Axis (`axis`)": "Full Array"},
    ),
    Benchmark(
        setup="np.ones({N} // 5 * 5, dtype=np.float32).reshape((5, -1))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "2", "Axis (`axis`)": "By Axis (Short)"},
    ),
    Benchmark(
        setup="np.arange({N} // 5 * 5, dtype=np.float32).reshape((5, -1))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "2", "Axis (`axis`)": "By Axis (Short)"},
    ),
    Benchmark(
        setup="np.arange({N} // 25 * 25, dtype=np.float32).reshape((5, 5, -1))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "3", "Axis (`axis`)": "By Axis (Short)"},
    ),
    Benchmark(
        setup="np.ones({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "2", "Axis (`axis`)": "By Axis (Long)"},
    ),
    Benchmark(
        setup="np.arange({N} // 5 * 5, dtype=np.float32).reshape((-1, 5))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "2", "Axis (`axis`)": "By Axis (Long)"},
    ),
    Benchmark(
        setup="np.arange({N} // 25 * 25, dtype=np.float32).reshape((-1, 5, 5))",
        statement="sum(array, axis=0)",
        tags={"Dimensions": "3", "Axis (`axis`)": "By Axis (Long)"},
    ),
]

sizes = (10, 100, 1_000, 10_000, 100_000, 1_000_000)


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
    seen_tags = defaultdict(set)

    for benchmark in benchmarks:
        for N in sizes:
            print(f"> array = {benchmark.setup.format(N=N)}\n> {benchmark.statement}")

            fast_time = benchmark.timeit(N, fm.sum)
            print(f"\tfast_math: {time_format(fast_time)}")

            numpy_time = benchmark.timeit(N, np.sum)
            print(f"\t    numpy: {time_format(numpy_time)}")

            result = {
                "N": N,
                "fast_math": fast_time,
                "numpy": numpy_time,
            }
            for tag_name, tag_value in benchmark.tags.items():
                result[tag_name] = tag_value
                seen_tags[tag_name].add(tag_value)

            results.append(result)

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

    for tag_name, tag_values in seen_tags.items():
        markdown += "\n## Results: " + tag_name + "\n"

        markdown += f"| {tag_name} | `fast_math` | NumPy | Slowdown |\n"
        markdown += "| -: | -: | -: | -: |\n"

        for tag_value in sorted(tag_values):
            fast_time = np.mean(
                [
                    result["fast_math"]
                    for result in results
                    if result.get(tag_name, None) == tag_value
                ]
            )
            numpy_time = np.mean(
                [
                    result["numpy"]
                    for result in results
                    if result.get(tag_name, None) == tag_value
                ]
            )

            markdown += (
                f"| {tag_value} | {time_format(fast_time)} | {time_format(numpy_time)} "
                f"| {fast_time / numpy_time:.1f}x |\n"
            )

    with open("benchmark.md", "w") as f:
        f.write(markdown)
