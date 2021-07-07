use std::ptr;

use ndarray::{ArrayBase, Ix1, Ix2, RawData};
use quadprogpp_sys as sys;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can happen in [`solve`]
#[derive(Debug, Error)]
pub enum Error {
    /// The problem has no feasible solution
    #[error("no feasible solution")]
    Infeasible,
    /// The given metrices and vectors have inconsistent dimentionalities.
    #[error("size mismatch")]
    SizeMismatch {
        term: &'static str,
        expected: usize,
        actual: usize,
    },
    /// FFI error
    #[error("{reason:?}")]
    Ffi { reason: String },
}

impl From<sys::Exception> for Error {
    fn from(exception: sys::Exception) -> Self {
        Self::Ffi {
            reason: format!("{}", exception),
        }
    }
}

macro_rules! assert_size {
    ($term:expr, $expected:expr, $actual:expr) => {
        if $expected != $actual {
            return Err(Error::SizeMismatch {
                term: stringify!($term),
                expected: $expected,
                actual: $actual,
            });
        }
    };
}

pub fn solve<G, G0, CE, CE0, CI, CI0>(
    (g, g0): (ArrayBase<G, Ix2>, ArrayBase<G0, Ix1>),
    ce: Option<(ArrayBase<CE, Ix2>, ArrayBase<CE0, Ix1>)>,
    ci: Option<(ArrayBase<CI, Ix2>, ArrayBase<CI0, Ix1>)>,
) -> Result<(Vec<f64>, f64)>
where
    G: RawData<Elem = f64>,
    G0: RawData<Elem = f64>,
    CE: RawData<Elem = f64>,
    CE0: RawData<Elem = f64>,
    CI: RawData<Elem = f64>,
    CI0: RawData<Elem = f64>,
{
    let (g_n, g_m) = g.dim();
    assert_size!(g, g_n, g_m);
    let mut g = unsafe { sys::new_matrix_from_ptr(g.as_ptr(), g_n as u32, g_m as u32) };
    let g0_n = g0.dim();
    assert_size!(g0.dim(), g_n, g0_n);
    let mut g0 = unsafe { sys::new_vector_from_ptr(g0.as_ptr(), g0_n as u32) };
    let (ce, ce0) = match ce {
        Some((ce, ce0)) => {
            let (ce_n, ce_m) = ce.dim();
            assert_size!(ce.dim(), g_n, ce_n);
            let ce = unsafe { sys::new_matrix_from_ptr(ce.as_ptr(), ce_n as u32, ce_m as u32) };
            let ce0_n = ce0.dim();
            assert_size!(ce0.dim(), ce0_n, ce_m);
            let ce0 = unsafe { sys::new_vector_from_ptr(ce0.as_ptr(), ce0_n as u32) };
            (ce, ce0)
        }
        None => {
            let ce = unsafe { sys::new_matrix_from_ptr(ptr::null(), g_n as u32, 0) };
            let ce0 = unsafe { sys::new_vector_from_ptr(ptr::null(), 0) };
            (ce, ce0)
        }
    };
    let (ci, ci0) = match ci {
        Some((ci, ci0)) => {
            let (ci_n, ci_m) = ci.dim();
            assert_size!(ci.dim(), g_n, ci_n);
            let ci = unsafe { sys::new_matrix_from_ptr(ci.as_ptr(), ci_n as u32, ci_m as u32) };
            let ci0_n = ci0.dim();
            assert_size!(ci0.dim(), ci0_n, ci_m);
            let ci0 = unsafe { sys::new_vector_from_ptr(ci0.as_ptr(), ci0_n as u32) };
            (ci, ci0)
        }
        None => {
            let ci = unsafe { sys::new_matrix_from_ptr(ptr::null(), g_n as u32, 0) };
            let ci0 = unsafe { sys::new_vector_from_ptr(ptr::null(), 0) };
            (ci, ci0)
        }
    };
    let mut x = unsafe { sys::new_vector(g_n as u32) };
    let best = sys::solve_quadprog(g.pin_mut(), g0.pin_mut(), &ce, &ce0, &ci, &ci0, x.pin_mut())?;
    if best.is_infinite() {
        return Err(Error::Infeasible);
    }
    let mut v = Vec::with_capacity(g_n);
    for i in 0..g_n {
        v.push(unsafe { sys::vector_index(&x, i as u32) });
    }
    assert_size!(v.len(), g_n, v.len());
    Ok((v, best))
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_ulps_eq};
    use ndarray::{array, Array, Array2, ArrayView1, ArrayView2};

    use super::*;

    #[test]
    fn quadprogpp_demo() -> Result<()> {
        #[rustfmt::skip]
        let g = array![
            [4.0, -2.0],
            [-2.0, 4.0],
        ];
        let g0 = array![6.0, 0.0];
        #[rustfmt::skip]
        let ce = array![
            [1.0],
            [1.0],
        ];
        let ce0 = array![-3.0];
        #[rustfmt::skip]
        let ci = array![
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ];
        let ci0 = array![0.0, -2.0, 0.0];
        let (x, best) = solve((g, g0), Some((ce, ce0)), Some((ci, ci0)))?;
        assert_ulps_eq!(best, 12.0);
        assert_ulps_eq!(x[0], 1.0);
        assert_ulps_eq!(x[1], 2.0);
        Ok(())
    }

    #[test]
    fn eiquadprog_demo() {
        let g = array![[2.1, 0.0, 1.0], [1.5, 2.2, 0.0], [1.2, 1.3, 3.1]];
        let g0 = array![6.0, 1.0, 1.0];
        let ce = array![[1.0], [2.0], [-1.0]];
        let ce0 = array![-4.0];
        #[rustfmt::skip]
        let ci = array![
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0,  0.0]
        ];
        let ci0 = array![0.0, 0.0, 0.0, 10.0];
        let (x, best) = solve((g, g0), Some((ce, ce0)), Some((ci, ci0))).unwrap();
        assert_ulps_eq!(best, 6.4);
        assert_ulps_eq!(x[0], 0.0);
        assert_ulps_eq!(x[1], 2.0);
        assert_ulps_eq!(x[2], 0.0);
    }

    // Problem 0 from hmatrix-quadpropp
    #[test]
    fn hmatrix_quadprogpp_problem0() -> Result<()> {
        #[rustfmt::skip]
        let g = array![
            [4.0, 0.0],
            [0.0, 2.0],
        ];
        let g0 = array![-4.0, -8.0];
        let ce: Option<(ArrayView2<_>, ArrayView1<_>)> = None;
        #[rustfmt::skip]
        let ci = array![
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -2.0],
        ];
        let ci0 = array![0.0, 0.0, 2.0];
        let (answer, _) = solve((g, g0), ce, Some((ci, ci0)))?;
        assert_abs_diff_eq!(answer[0], 2.0 / 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(answer[1], 8.0 / 9.0, epsilon = 1e-12);
        Ok(())
    }

    // Problem 1 from hmatrix-quadprogpp
    #[test]
    fn hmatrix_quadprogpp_problem1() -> Result<()> {
        let offset: Array2<f64> = Array::eye(3) * 1e-12;
        #[rustfmt::skip]
        let g = array![
            [      1.0, 2.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0,       0.0],
            [1.0 / 3.0,       0.0, 1.0 / 3.0],
        ] + offset;
        let g0 = array![-2.0, -4.0, 2.0];
        let ce = array![[-3.0], [2.0], [1.0]];
        let ce0 = array![0.0];
        #[rustfmt::skip]
        let ci = array![
            [1.0,        0.0,        0.0],
            [0.0,  1.0 / 3.0, -4.0 / 3.0],
            [0.0, -1.0 / 3.0,  1.0 / 3.0]
        ];
        let ci0 = array![0.0, 0.0, 2.0];
        let (answer, _) = solve((g, g0), Some((ce, ce0)), Some((ci, ci0)))?;
        assert_ulps_eq!(answer[0], 2.0 / 9.0, epsilon = 1e-5);
        assert_abs_diff_eq!(answer[1], 10.0 / 9.0, epsilon = 1e-5);
        assert_abs_diff_eq!(answer[2], -14.0 / 9.0, epsilon = 1e-5);
        Ok(())
    }
}
