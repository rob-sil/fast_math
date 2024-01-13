/// Additional methods for floating point values.
pub trait FloatingPoint: Sized {
    const MANTISSA_BITS: usize;
    const EXPONENT_BITS: usize;

    fn exponent(&self) -> usize;
    fn mantissa(&self) -> usize;
}

impl FloatingPoint for f32 {
    const MANTISSA_BITS: usize = 23;
    const EXPONENT_BITS: usize = 8;

    /// The exponent bits of the floating-point representation.
    /// Zero denotes the smallest exponent available (excluding subnormals).
    /// 1=2^0 will have a positive exponent.
    fn exponent(&self) -> usize {
        ((self.to_bits() >> f32::MANTISSA_BITS) & ((1 << f32::EXPONENT_BITS) - 1)) as usize
    }

    /// The mantissa bits of the floating-point representation, excluding the
    /// implied one.
    fn mantissa(&self) -> usize {
        (self.to_bits() & ((1 << f32::MANTISSA_BITS) - 1)) as usize
    }
}
