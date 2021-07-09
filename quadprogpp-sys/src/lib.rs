pub use cxx::Exception;

pub use ffi::*;

#[cxx::bridge(namespace = "quadprogpp")]
mod ffi {
    unsafe extern "C++" {
        include!("quadprogpp-sys/include/wrapper.hpp");

        /// A vector type whose element type is f64.
        type VectorF64;

        /// Creates a new zero-filled [`VectorF64`] of length `n`.
        fn new_vector(n: u32) -> UniquePtr<VectorF64>;

        /// Creates a new [`VectorF64`] from a pointer to the array and its length. Note that it
        /// copies the data.
        ///
        /// # Safety
        ///
        /// This is unsafe due to the use of a raw pointer.
        unsafe fn new_vector_from_ptr(a: *const f64, n: u32) -> UniquePtr<VectorF64>;

        /// Performs indexing operation on the vector.
        ///
        /// # Safety
        ///
        /// This is unsafe because the index range isn't checked.
        unsafe fn vector_index(v: &VectorF64, i: u32) -> f64;

        /// A 2D matrix type whose element type is f64.
        type MatrixF64;

        /// Creates a new `n x m` [`MatrixF64`] from a pointer to a row-major array and its shape.
        /// Note that it copies the data.
        ///
        /// # Safety
        ///
        /// This is unsafe due to the use of a raw pointer.
        unsafe fn new_matrix_from_ptr(a: *const f64, n: u32, m: u32) -> UniquePtr<MatrixF64>;
    }

    unsafe extern "C++" {
        /// Sovles a quadratic programming problem.
        fn solve_quadprog(
            G: Pin<&mut MatrixF64>,
            g0: Pin<&mut VectorF64>,
            CE: &MatrixF64,
            ce0: &VectorF64,
            CI: &MatrixF64,
            ci0: &VectorF64,
            x: Pin<&mut VectorF64>,
        ) -> Result<f64>;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    #[allow(clippy::many_single_char_names, non_snake_case)]
    fn test() {
        let n = 2;
        let m = 1;
        let p = 3;
        let mut G =
            unsafe { new_matrix_from_ptr([4.0, -2.0, -2.0, 4.0].as_ptr() as *const f64, n, n) };
        let mut g0 = unsafe { new_vector_from_ptr([6.0, 0.0].as_ptr() as *const f64, n) };
        let CE = unsafe { new_matrix_from_ptr([1.0, 1.0].as_ptr() as *const f64, n, m) };
        let ce0 = unsafe { new_vector_from_ptr([-3.0].as_ptr() as *const f64, m) };
        let CI = unsafe {
            new_matrix_from_ptr([1.0, 0.0, 1.0, 0.0, 1.0, 1.0].as_ptr() as *const f64, n, p)
        };
        let ci0 = unsafe { new_vector_from_ptr([0.0, 0.0, -2.0].as_ptr() as *const f64, p) };
        let mut x = new_vector(n);
        let r =
            solve_quadprog(G.pin_mut(), g0.pin_mut(), &CE, &ce0, &CI, &ci0, x.pin_mut()).unwrap();
        assert_ulps_eq!(r, 12.0);
        assert_ulps_eq!(unsafe { vector_index(&x, 0) }, 1.0);
        assert_ulps_eq!(unsafe { vector_index(&x, 1) }, 2.0);
    }
}
