use crate::expansion::Expansion;
use crate::float::FloatingPoint;
use crate::online_sum::OnlineSumAlgorithm;

/// Calculate the size an accumulator needs to hold floating-point summation
const fn accumulator_array_size<T>() -> usize
where
    T: FloatingPoint,
{
    2_usize.pow(T::EXPONENT_BITS as u32)
}

#[derive(Clone, Copy)]
struct Accumulator {
    highs: [f32; accumulator_array_size::<f32>()],
    lows: [f32; accumulator_array_size::<f32>()],
    count: usize,
    special: f32,
}

impl Accumulator {
    /// Create an accumulator representing zero.
    pub fn new() -> Accumulator {
        Accumulator {
            highs: [0_f32; accumulator_array_size::<f32>()],
            lows: [0_f32; accumulator_array_size::<f32>()],
            count: 0,
            special: 0_f32,
        }
    }

    //. Add a floating-point value to the expansion without rounding error.
    #[inline(always)]
    fn add(&mut self, value: f32) {
        if value.is_finite() {
            let j = value.exponent();

            let high = self.highs[j] + value;
            let low = value - (high - self.highs[j]);

            self.highs[j] = high;
            self.lows[j] += low;

        // Haven't encountered NAN or an infinity yet
        } else if self.special == 0_f32 {
            self.special = value;

		// Two different non-finite values add to NAN
        } else if self.special != value {
            self.special = f32::NAN;
        }

        self.count += 1;
    }

    // Reset the accumulator to zero
    fn clear(&mut self) {
        self.highs.fill(0_f32);
        self.lows.fill(0_f32);
        self.count = 0;
		self.special = 0_f32;
    }

    /// Add the value of the accumulator to the sink.
    fn drain_into(&mut self, sink: &mut Accumulator) {
		if self.special != 0_f32 {
			sink.add(self.special);
		} else {
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
		}

        self.clear();
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// A summation algorithm that uses `A` accumulators in parallel
pub struct MultiAccumulator<const A: usize> {
    accumulators: [Accumulator; A],
    swap: Accumulator,
    swapped: bool,
}

impl<const A: usize> OnlineSumAlgorithm<A> for MultiAccumulator<A> {
    fn new() -> Self {
        MultiAccumulator {
            accumulators: [Accumulator::new(); A],
            swap: Accumulator::new(),
            swapped: true,
        }
    }

    fn add(&mut self, value: f32) {
        let (first, rest) = self.accumulators.split_at_mut(1);

        let (active, backup) = if self.swapped {
            (&mut self.swap, &mut first[0])
        } else {
            (&mut first[0], &mut self.swap)
        };

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

		if backup.special != 0_f32 {
			backup.special
		} else {
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
}
