use crate::expansion::Expansion;
use crate::online_sum::OnlineSumAlgorithm;

#[derive(Clone, Copy)]
struct Accumulator<const N: usize> {
    highs: [f32; N],
    lows: [f32; N],
    count: usize,
}

impl<const N: usize> Accumulator<N> {
    /// Create an accumulator representing zero.
    pub fn new() -> Accumulator<N> {
        Accumulator {
            highs: [0_f32; N],
            lows: [0_f32; N],
            count: 0,
        }
    }

    //. Add a floating-point value to the expansion without rounding error.
    #[inline(always)]
    fn add(&mut self, value: f32) {
        let j = ((value.to_bits() >> 23) & ((1 << 8) - 1)) as usize;

        let high = self.highs[j] + value;
        let low = value - (high - self.highs[j]);

        self.highs[j] = high;
        self.lows[j] += low;

        self.count += 1;
    }

    // Reset the accumulator to zero
    fn clear(&mut self) {
        self.highs.fill(0_f32);
        self.lows.fill(0_f32);
        self.count = 0;
    }

    /// Add the value of the accumulator to the sink.
    fn drain_into(&mut self, sink: &mut Accumulator<N>) {
        for value in self.highs {
            if value != 0_f32 {
                sink.add(value);
            }
        }

        for value in self.lows {
            if value != 0_f32 {
                sink.add(value);
            }
        }

        self.clear();
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// A summation algorithm that uses `A` accumulators in parallel
pub struct MultiAccumulator<const N: usize, const A: usize> {
    accumulators: [Accumulator<N>; A],
    swap: Accumulator<N>,
    swapped: bool,
}

impl<const N: usize, const A: usize> OnlineSumAlgorithm<A> for MultiAccumulator<N, A> {
    fn new() -> Self {
        MultiAccumulator {
            accumulators: [Accumulator::<N>::new(); A],
            swap: Accumulator::<N>::new(),
            swapped: true,
        }
    }

    fn add(&mut self, value: f32) {
        let (first, rest) = self.accumulators.split_at_mut(1);

        let mut active = &mut first[0];
        let mut backup = &mut self.swap;
        if self.swapped {
            active = &mut self.swap;
            backup = &mut first[0];
        }

        active.add(value);

        if active.count() > 2_usize.pow(22) {
            active.drain_into(backup);
            for accumulator in rest {
                accumulator.drain_into(backup)
            }
            self.swapped = !self.swapped;
        }
    }

    #[inline(always)]
    fn add_array(&mut self, values: &[f32; A]) {
        let (first, rest) = self.accumulators.split_at_mut(1);

        let (active, backup) = if self.swapped {
            (&mut self.swap, &mut first[0])
        } else {
            (&mut first[0], &mut self.swap)
        };

        active.add(values[0]);
        for i in 1..A {
            rest[i - 1].add(values[i])
        }

        if active.count() > 2_usize.pow(22) {
            active.drain_into(backup);
            for accumulator in rest {
                accumulator.drain_into(backup)
            }
            self.swapped = !self.swapped;
        }
    }

    fn finalize(mut self) -> f32 {
        let (first, rest) = self.accumulators.split_at_mut(1);

        let (active, backup) = if self.swapped {
            (&mut self.swap, &mut first[0])
        } else {
            (&mut first[0], &mut self.swap)
        };

        active.drain_into(backup);
        for accumulator in rest {
            accumulator.drain_into(backup)
        }

        let mut expansion = Expansion::new();
        for value in backup.highs {
            expansion.add(value);
        }
        for value in backup.lows {
            expansion.add(value);
        }

        expansion.finalize()
    }
}
