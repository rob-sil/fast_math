use std::convert::{From, Into};

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

/// A non-overlapping expansion of a number
///
/// Implementation of Shewchuk (1997)'s accurate floating-point methods.
pub struct Expansion {
    components: Vec<f32>,
}

impl Expansion {
    /// Create an expansion representing zero.
    pub fn new() -> Expansion {
        Expansion { components: vec![] }
    }

    /// Add a floating-point value to the expansion without rounding error.
    pub fn add(&mut self, value: f32) {
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
}

impl From<f32> for Expansion {
    fn from(value: f32) -> Expansion {
        Expansion {
            components: vec![value],
        }
    }
}

impl Into<f32> for Expansion {
    fn into(self) -> f32 {
        if self.components.len() > 0 {
            self.components[self.components.len() - 1]
        } else {
            0.
        }
    }
}

pub fn online_sum<'a, I, T>(values: I) -> f32
where
    I: IntoIterator,
    I::Item: Into<&'a T>,
    T: Into<f32> + 'a + Copy,
{
	let mut expansion = Expansion::new();
	for value in values {
		let &value_t = value.into();
		let value_f32: f32 = value_t.into();
		if value_f32 != 0_f32 {
			expansion.add(value_f32);
		}
	}
	expansion.into()
}
