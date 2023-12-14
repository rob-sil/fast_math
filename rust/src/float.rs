pub trait FloatingPoint: Sized {
	const MANTISSA_BITS: usize;
	const EXPONENT_BITS: usize;

	fn exponent(&self) -> usize;
}

impl FloatingPoint for f32 {
	const MANTISSA_BITS: usize = 23;
	const EXPONENT_BITS: usize = 8;

	fn exponent(&self) -> usize {
		((self.to_bits() >> f32::MANTISSA_BITS) & ((1 << f32::EXPONENT_BITS) - 1)) as usize
	}
}