use std::ptr;

use ndarray::{ArrayBase, Ix1, Ix2, RawData};
use quadprogpp_sys as sys;

pub fn solve<Sg, Sg0, Sce, Sce0, Sci, Sci0>(
    (g, g0): (ArrayBase<Sg, Ix2>, ArrayBase<Sg0, Ix1>),
    ce: Option<(ArrayBase<Sce, Ix2>, ArrayBase<Sce0, Ix1>)>,
    ci: Option<(ArrayBase<Sci, Ix2>, ArrayBase<Sci0, Ix1>)>,
) -> (Vec<f64>, f64)
where
    Sg: RawData<Elem = f64>,
    Sg0: RawData<Elem = f64>,
    Sce: RawData<Elem = f64>,
    Sce0: RawData<Elem = f64>,
    Sci: RawData<Elem = f64>,
    Sci0: RawData<Elem = f64>,
{
    let (n, m) = match *g.shape() {
        [n, m] => (n, m),
        _ => unreachable!(),
    };
    let mut g = unsafe { sys::new_matrix_from_ptr(g.as_ptr(), n as u32, m as u32) };
    let n = match *g0.shape() {
        [n] => n,
        _ => unreachable!(),
    };
    let mut g0 = unsafe { sys::new_vector_from_ptr(g0.as_ptr(), n as u32) };
    let (ce, ce0) = match ce {
        Some((ce, ce0)) => {
            let (n, m) = match *ce.shape() {
                [n, m] => (n, m),
                _ => unreachable!(),
            };
            let ce = unsafe { sys::new_matrix_from_ptr(ce.as_ptr(), n as u32, m as u32) };
            let n = match *ce0.shape() {
                [n] => n,
                _ => unreachable!(),
            };
            let ce0 = unsafe { sys::new_vector_from_ptr(ce0.as_ptr(), n as u32) };
            (ce, ce0)
        }
        None => {
            let ce = unsafe { sys::new_matrix_from_ptr(ptr::null(), 0, 0) };
            let ce0 = unsafe { sys::new_vector_from_ptr(ptr::null(), 0) };
            (ce, ce0)
        }
    };
    let (ci, ci0) = match ci {
        Some((ci, ci0)) => {
            let (n, m) = match *ci.shape() {
                [n, m] => (n, m),
                _ => unreachable!(),
            };
            let ci = unsafe { sys::new_matrix_from_ptr(ci.as_ptr(), n as u32, m as u32) };
            let n = match *ci0.shape() {
                [n] => n,
                _ => unreachable!(),
            };
            let ci0 = unsafe { sys::new_vector_from_ptr(ci0.as_ptr(), n as u32) };
            (ci, ci0)
        }
        None => {
            let ci = unsafe { sys::new_matrix_from_ptr(ptr::null(), 0, 0) };
            let ci0 = unsafe { sys::new_vector_from_ptr(ptr::null(), 0) };
            (ci, ci0)
        }
    };
    let mut x = unsafe { sys::new_vector(n as u32) };
    let best = sys::solve_quadprog(g.pin_mut(), g0.pin_mut(), &ce, &ce0, &ci, &ci0, x.pin_mut());
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(unsafe { sys::vector_index(&x, i as u32) });
    }
    (v, best)
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use ndarray::{ArrayView1, ArrayView2};

    use super::*;

    #[test]
    fn test() {
        let n = 2;
        let m = 1;
        let p = 3;
        let g = ArrayView2::from_shape((n, n), &[4.0, -2.0, -2.0, 4.0]).unwrap();
        let g0 = ArrayView1::from_shape(n, &[6.0, 0.0]).unwrap();
        let ce = ArrayView2::from_shape((n, m), &[1.0, 1.0]).unwrap();
        let ce0 = ArrayView1::from_shape(m, &[-3.0]).unwrap();
        let ci = ArrayView2::from_shape((n, p), &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let ci0 = ArrayView1::from_shape(p, &[0.0, 0.0, -2.0]).unwrap();
        let (x, best) = solve((g, g0), Some((ce, ce0)), Some((ci, ci0)));
        assert_ulps_eq!(best, 12.0);
        assert_ulps_eq!(x[0], 1.0);
        assert_ulps_eq!(x[1], 2.0);
    }
}
