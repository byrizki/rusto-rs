use smallvec::SmallVec;

/// Stack-allocated shape representation up to 6 dimensions without heap allocation
pub type Shape = SmallVec<[usize; 4]>;

/// Stack-allocated strides representation
pub type Strides = SmallVec<[usize; 4]>;

/// Compute contiguous C-order (row-major) strides for a given shape
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = Strides::with_capacity(shape.len());
    if shape.is_empty() {
        return strides;
    }
    strides.resize(shape.len(), 1);
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute total element count of a shape
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_strides() {
        let shape: Shape = smallvec::smallvec![1, 3, 48, 320];
        assert_eq!(element_count(&shape), 3 * 48 * 320);
        assert_eq!(element_count(&[]), 1);

        let strides = contiguous_strides(&shape);
        assert_eq!(strides.as_slice(), &[3 * 48 * 320, 48 * 320, 320, 1]);
    }
}

