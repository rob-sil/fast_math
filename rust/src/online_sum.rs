pub trait OnlineSumAlgorithm<const N: usize>: Sized {
    /// Create a new instance of this algorithm
    fn new() -> Self;

    /// Add a single float value
    fn add(&mut self, value: f32);

    /// Get the final value of the sum as a float
    fn finalize(self) -> f32;

    /// Add multiple elements at once
    fn add_array(&mut self, values: &[f32; N]) {
        for &value in values {
            self.add(value);
        }
    }

    /// Sum up values, using the algorithm
    fn online_sum<'a, I, T>(values: I) -> f32
    where
        I: IntoIterator,
        I::Item: Into<&'a T>,
        T: Into<f32> + 'a + Copy,
    {
        let mut algorithm = Self::new();

        let mut to_add = [0_f32; N];

        let mut values_iter = values.into_iter();
        while let Some(value) = values_iter.next() {
            to_add[0] = (*value.into()).into();
            for i in 1..N {
                to_add[i] = values_iter
                    .next()
                    .map_or(0., |value| (*value.into()).into());
            }

            algorithm.add_array(&to_add);
        }

        algorithm.finalize()
    }
}
