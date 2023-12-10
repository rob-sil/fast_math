use ndarray::{Array, ArrayView, ArrayView1, Axis, Dim, Ix, Ix1, RemoveAxis, Shape, ShapeBuilder};

pub fn map_axis<'a, A, B, D, F>(
    array: &'a ArrayView<A, D>,
    axis: Axis,
    mut mapping: F,
) -> Array<B, D::Smaller>
where
    D: RemoveAxis,
    F: FnMut(ArrayView1<'a, A>) -> B,
    A: 'a,
{
    let shape: Shape<_> = Dim::<Ix>(array.len_of(axis)).into();

    if array.len_of(axis) == 0 {
        let empty = unsafe { ArrayView1::<A>::from_shape_ptr(shape, array.as_ptr()) };
        Array::<B, D::Smaller>::from_shape_simple_fn(
            array.raw_dim().remove_axis(axis).into_shape(),
            || mapping(empty),
        )
    } else {
        let stride = shape.strides(Ix1(array.stride_of(axis) as usize));
        array
            .index_axis(axis, 0)
            .map(|ptr| unsafe { mapping(ArrayView1::from_shape_ptr(stride, ptr)) })
    }
}
