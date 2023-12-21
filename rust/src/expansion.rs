use crate::online_sum::OnlineSumAlgorithm;

/// Add two numbers with the Fast2Sum algorithm
///
/// Returns the sum along with the rounding error.
#[inline(always)]
fn fast2sum(a: f32, b: f32) -> (f32, f32) {
    if a.abs() < b.abs() {
        let high = b + a;
        let low = a - (high - b);
        (high, low)
    } else {
        let high = a + b;
        let low = b - (high - a);
        (high, low)
    }
}

#[derive(Debug)]
/// A non-overlapping expansion of a number
///
/// Implementation of Shewchuk (1997)'s accurate floating-point methods.
pub struct Expansion {
	/// A list of non-overlapping floats representing the value
    components: Vec<f32>,
	/// NAN or an infinity if one has been encountered, otherwise zero
	special: f32,
}

impl Expansion {
    #[inline(always)]
    /// Round the represented value to the nearest floating-point value
    pub fn round(&self) -> f32 {
		if self.special != 0_f32 {
			self.special
		} else if self.components.len() > 0 {
            self.components[self.components.len() - 1]
        } else {
            0.
        }
    }
}

impl OnlineSumAlgorithm<1> for Expansion {
    fn new() -> Self {
        Expansion { components: vec![], special: 0_f32 }
    }

    fn add(&mut self, value: f32) {
		// Handle NAN and infinities
		if !value.is_finite() {
			if self.special == 0_f32 {
				self.special = value;
			} else if self.special != value {
				self.special = f32::NAN;
			}
			return;
		}

		// Sum finite values
        let mut current = value;
        let mut j = 0;
        for i in 0..self.components.len() {
            let (high, low) = fast2sum(current, self.components[i]);
            current = high;
            if low != 0_f32 {
                self.components[j] = low;
                j += 1;
            }
        }
        self.components.truncate(j);

		if current != 0_f32 {
            self.components.push(current);
        }
    }

    fn finalize(self) -> f32 {
        self.round()
    }
}

impl IntoIterator for Expansion {
    type Item = f32;

    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.components.into_iter()
    }
}
