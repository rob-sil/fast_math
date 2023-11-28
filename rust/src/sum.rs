use std::convert::Into;

use crate::expansion::Expansion;

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
pub fn online_sum<'a, I>(values: I) -> f32
where
    I: IntoIterator,
    I::Item: Into<&'a f32>,
{
    let mut accumulator0 = Accumulator::<{ (f32::MAX_EXP - f32::MIN_EXP + 1) as usize }>::new();
    let mut accumulator1 = Accumulator::<{ (f32::MAX_EXP - f32::MIN_EXP + 1) as usize }>::new();

    // Pointers rotate the first accumulator
    let mut backup = &mut accumulator0;
    let mut active = &mut accumulator1;

    // We can only add so many values before needing a reset
    let max_active_count = 2_usize.pow(22);

    for value in values {
        active.add(*value.into());

        if active.count() > max_active_count {
            active.drain_into(backup);

            (active, backup) = (backup, active);
        }
    }

    let expansion: Expansion = active.into();
    expansion.into()
}
