use std::convert::Into;

use crate::expansion::Expansion;

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
    fn add(&mut self, value: f32) {
        let j = ((value.to_bits() >> 23) & ((1 << 8) - 1)) as usize;

        let high = self.highs[j] + value;
        let low = value - (high - self.highs[j]);

        self.highs[j] = high;
        self.lows[j] += low;

        self.count += 1;
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

        self.highs.fill(0_f32);
        self.lows.fill(0_f32);
        self.count = 0;
    }

    fn count(&self) -> usize {
        self.count
    }
}

impl<const N: usize> From<&mut Accumulator<N>> for Expansion {
    fn from(accumulator: &mut Accumulator<N>) -> Expansion {
        let mut expansion = Expansion::new();
        for value in accumulator.highs {
            if value != 0_f32 {
                expansion.add(value);
            }
        }
        for value in accumulator.lows {
            if value != 0_f32 {
                expansion.add(value);
            }
        }

        expansion
    }
}

/// An online, accurate sum based on Zhu and Hayes' OnlineExactSum.
/// Uses N+1 accumulators to add faster
pub fn online_sum<'a, I, T, const A: usize, const N: usize>(values: I) -> f32
where
    I: IntoIterator,
    I::Item: Into<&'a T>,
    T: Into<f32> + 'a + Copy,
{
    // Pointers rotate the first accumulator with the swap accumulator
    let mut swap0 = Accumulator::<N>::new();
    let mut swap1 = Accumulator::<N>::new();
    let mut active = &mut swap0;
    let mut backup = &mut swap1;

    // Additional accumulators to speed up summation
    let mut accumulators = [Accumulator::<N>::new(); A];

    // We can only add so many values before needing a reset
    let max_active_count = 2_usize.pow(22);

    let mut values_iter = values.into_iter();
    let mut chunk = [0_f32; A];
    while let Some(value1) = values_iter.next() {
        for i in 0..chunk.len() {
            chunk[i] = match values_iter.next() {
                Some(value) => (*value.into()).into(),
                None => 0.,
            }
        }

        active.add((*value1.into()).into());
        for i in 0..A {
            accumulators[i].add(chunk[i])
        }

        if active.count() > max_active_count {
            active.drain_into(backup);
            for mut accumulator in accumulators {
                accumulator.drain_into(backup)
            }
            (active, backup) = (backup, active);
        }
    }

    active.drain_into(backup);
    for mut accumulator in accumulators {
        accumulator.drain_into(backup)
    }

    let expansion: Expansion = backup.into();
    expansion.into()
}
