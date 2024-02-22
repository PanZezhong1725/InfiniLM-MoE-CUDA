﻿use crate::{
    idim,
    operator::{Broadcast, Slice, SliceDim, Split, Squeeze, SqueezeOp, Transpose},
    udim, DataType, Operator,
};
use nalgebra::{DMatrix, DVector, DVectorView};
use rayon::iter::*;
use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    data_type: DataType,
    shape: Shape,
    pattern: Pattern,
    physical: Physical,
}

impl<Physical> Tensor<Physical> {
    pub fn new(data_type: DataType, shape: &[udim], physical: Physical) -> Self {
        let shape = Shape::from_iter(shape.iter().map(|&d| d as udim));
        Self {
            data_type,
            pattern: Pattern::from_shape(&shape, 0),
            shape,
            physical,
        }
    }

    #[inline]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    #[inline]
    pub fn shape(&self) -> &[udim] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[idim] {
        self.pattern.strides()
    }

    #[inline]
    pub const fn physical(&self) -> &Physical {
        &self.physical
    }

    #[inline]
    pub fn physical_mut(&mut self) -> &mut Physical {
        &mut self.physical
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    #[inline]
    pub fn bytes_size(&self) -> usize {
        self.size() * self.data_type.size()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous_len() == self.shape.len()
    }

    /// 连续维度的数量。
    pub fn contiguous_len(&self) -> usize {
        self.pattern
            .strides()
            .iter()
            .enumerate()
            .rev()
            .scan(1 as idim, |mul, (i, &s)| {
                if s == *mul {
                    *mul *= self.shape[i] as idim;
                    Some(())
                } else {
                    None
                }
            })
            .count()
    }

    /// # Safety
    ///
    /// The caller must ensure that the `physical` matches shape, pattern of `self` and the new `dtype`.
    #[inline]
    pub unsafe fn set_physical<U>(&self, dtype: DataType, physical: U) -> Tensor<U> {
        Tensor {
            data_type: dtype,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical,
        }
    }

    #[inline]
    fn byte_offset(&self) -> usize {
        self.pattern.offset() as usize * self.data_type.size()
    }
}

impl<Physical: Clone> Tensor<Physical> {
    pub fn reshape(&self, shape: &[udim]) -> Self {
        assert!(self.is_contiguous());
        assert_eq!(
            self.shape.iter().product::<udim>(),
            shape.iter().product::<udim>(),
        );
        let shape = Shape::from_slice(shape);
        Self {
            data_type: self.data_type,
            pattern: Pattern::from_shape(&shape, self.pattern.offset()),
            shape,
            physical: self.physical.clone(),
        }
    }

    pub fn apply(&self, operator: &impl Operator) -> SmallVec<[Self; 1]> {
        operator
            .build(&self.shape)
            .into_iter()
            .map(|(shape, affine)| Self {
                data_type: self.data_type,
                shape,
                pattern: Pattern(affine * &self.pattern.0),
                physical: self.physical.clone(),
            })
            .collect()
    }

    pub fn broadcast(&self, shape: &[usize]) -> Self {
        self.apply(&Broadcast(Shape::from_iter(
            shape.iter().map(|&d| d as udim),
        )))
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn slice(&self, dims: &[SliceDim]) -> Self {
        self.apply(&Slice(dims.to_vec()))
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn split(&self, axis: usize, segments: &[usize]) -> SmallVec<[Self; 1]> {
        self.apply(&Split {
            axis: axis as udim,
            segments: Shape::from_iter(segments.iter().map(|&d| d as udim)),
        })
    }

    pub fn squeeze(&self, ops: &str) -> Self {
        self.apply(&Squeeze(
            ops.chars()
                .map(|s| match s {
                    '+' => SqueezeOp::Insert,
                    '-' => SqueezeOp::Remove,
                    '_' => SqueezeOp::Skip,
                    _ => unreachable!(),
                })
                .collect(),
        ))
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn transpose(&self, axes: &[usize]) -> Self {
        self.apply(&Transpose(SmallVec::from_iter(
            axes.iter().map(|&i| i as udim),
        )))
        .into_iter()
        .next()
        .unwrap()
    }
}

impl<Physical: Deref<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(self.is_contiguous());
        let off = self.byte_offset();
        let len = self.bytes_size();
        &self.physical[off..][..len]
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        let ptr = self.physical.as_ptr();
        let offset = self.byte_offset();
        unsafe { ptr.add(offset) }
    }

    pub fn reform_to(&self, dst: &mut [u8]) {
        let src = &self.physical[self.byte_offset()..];
        // 计算结尾连续维度数量
        let contiguous = self.contiguous_len();
        if contiguous == self.shape.len() {
            // 所有维度都连续，直接拷贝所有数据
            dst.copy_from_slice(&src[..dst.len()]);
        } else {
            let dt = self.data_type.size();
            // 一部分维度连续，迭代不连续的部分
            let (iter, contiguous) = self.shape.split_at(self.shape.len() - contiguous);
            let (n, idx_strides) = idx_strides(iter);
            let len = contiguous.iter().product::<udim>() as usize * dt;
            let pattern = self.pattern.0.view_range(..iter.len(), ..);
            let ptr = dst.as_mut_ptr() as usize;
            (0..n).into_par_iter().for_each(|i| {
                let j = pattern.dot(&expand_indices(i, &idx_strides, &[]));
                unsafe { std::slice::from_raw_parts_mut((ptr + i as usize * len) as *mut u8, len) }
                    .copy_from_slice(&src[j as usize * dt..][..len]);
            });
        }
    }
}

impl<Physical: DerefMut<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        debug_assert!(self.is_contiguous());
        let off = self.byte_offset();
        let len = self.bytes_size();
        &mut self.physical[off..][..len]
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        let ptr = self.physical.as_mut_ptr();
        let offset = self.byte_offset();
        unsafe { ptr.add(offset) }
    }

    pub fn get_mut_ptr(&mut self, indices: &DVectorView<idim>) -> Option<*mut u8> {
        let i = self.pattern.0.dot(indices) as usize * self.data_type.size();
        let physical = self.physical.as_mut();
        if i < physical.len() {
            Some(unsafe { physical.as_mut_ptr().add(i) })
        } else {
            None
        }
    }
}

pub type Shape = SmallVec<[udim; 4]>;
pub type Affine = DMatrix<idim>;

#[derive(Clone, Debug)]
struct Pattern(DVector<idim>);

impl Pattern {
    pub fn from_shape(shape: &[udim], offset: idim) -> Self {
        let n = shape.len();
        let mut strides = vec![0; n + 1];
        strides[n - 1] = 1;
        strides[n] = offset;
        for i in (1..n).rev() {
            strides[i - 1] = strides[i] * shape[i] as idim;
        }
        Self(DVector::from_vec(strides))
    }

    #[inline]
    pub fn strides(&self) -> &[idim] {
        &self.0.as_slice()[..self.0.len() - 1]
    }

    #[inline]
    pub fn offset(&self) -> idim {
        self.0[self.0.len() - 1]
    }
}

pub fn idx_strides(shape: &[udim]) -> (udim, Vec<udim>) {
    let mut idx_strides = vec![0; shape.len()];
    idx_strides[shape.len() - 1] = 1;
    for i in (1..shape.len()).rev() {
        idx_strides[i - 1] = idx_strides[i] * shape[i];
    }
    (shape[0] * idx_strides[0], idx_strides)
}

pub fn expand_indices(i: udim, idx_strides: &[udim], tail: &[idim]) -> DVector<idim> {
    let mut rem = i as idim;
    let mut ans = vec![0 as idim; idx_strides.len() + tail.len()];
    for (i, &s) in idx_strides.iter().enumerate() {
        ans[i] = rem / s as idim;
        rem %= s as idim;
    }
    ans[idx_strides.len()..].copy_from_slice(tail);
    DVector::from_vec(ans)
}

#[test]
fn test() {
    let t = Tensor::new(DataType::F32, &[2, 3, 4, 5], ());
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 5, 1, 0]);
    assert_eq!(t.contiguous_len(), 4);
    assert_eq!(t.is_contiguous(), true);

    let t = t.reshape(&[2, 3, 20]);
    assert_eq!(t.shape(), &[2, 3, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 1, 0]);
    assert_eq!(t.contiguous_len(), 3);
    assert_eq!(t.is_contiguous(), true);

    let t = t.transpose(&[1, 0, 2]);
    assert_eq!(t.shape(), &[3, 2, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 60, 1, 0]);
    assert_eq!(t.contiguous_len(), 1);
    assert_eq!(t.is_contiguous(), false);
}
