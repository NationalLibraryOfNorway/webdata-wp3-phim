use pyo3::{prelude::*};
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::ndarray::ArrayView1;
use numpy::ndarray::ArrayView2;
use numpy::PyArray1;
use numpy::IntoPyArray;

// ----------------------------------------------------------------------------------------------------------
// Copied from https://github.com/emschwartz/hamming-bitwise-fast/blob/main/src/lib.rs.
// Copyright 2024 Evan Schwartz (MIT License).
// ----------------------------------------------------------------------------------------------------------
// A fast, zero-dependency implementation of bitwise Hamming Distance using
// a method amenable to auto-vectorization.
//
// Calculate the bitwise Hamming distance between two byte slices.
//
// While this implementation does not explicitly use SIMD, it uses
// a technique that is amenable to auto-vectorization. Its performance
// is similar to or faster than more complex implementations that use
// explicit SIMD instructions for specific architectures.
//
// # Panics
//
// Panics if the two slices are not the same length.
#[inline]
pub fn hamming_bitwise_fast(x: &[u8], y: &[u8]) -> u32 {
    assert_eq!(x.len(), y.len());

    // Process 8 bytes at a time using u64
    let mut distance = x
        .chunks_exact(8)
        .zip(y.chunks_exact(8))
        .map(|(x_chunk, y_chunk)| {
            // This is safe because we know the chunks are exactly 8 bytes.
            // Also, we don't care whether the platform uses little-endian or big-endian
            // byte order. Since we're only XORing values, we just care that the
            // endianness is the same for both.
            let x_val = u64::from_ne_bytes(x_chunk.try_into().unwrap());
            let y_val = u64::from_ne_bytes(y_chunk.try_into().unwrap());
            (x_val ^ y_val).count_ones()
        })
        .sum::<u32>();

    if x.len() % 8 != 0 {
        distance += x
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(y.chunks_exact(8).remainder())
            .map(|(x_byte, y_byte)| (x_byte ^ y_byte).count_ones())
            .sum::<u32>();
    }

    distance
}

#[doc(hidden)]
#[inline]
pub fn naive_hamming_distance(x: &[u8], y: &[u8]) -> u64 {
    assert_eq!(x.len(), y.len());
    let mut distance: u32 = 0;
    for i in 0..x.len() {
        distance += (x[i] ^ y[i]).count_ones();
    }
    distance as u64
}

#[doc(hidden)]
#[inline]
pub fn naive_hamming_distance_iter(x: &[u8], y: &[u8]) -> u64 {
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones()) as u64
}

// ----------------------------------------------------------------------------------------------------------
// End of copy
// ----------------------------------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (x, y, /))]
pub fn compute_bitwise_hamming_distance(x: &[u8], y: &[u8]) -> PyResult<u32>{
    Ok(hamming_bitwise_fast(x, y))
}


#[pyfunction]
#[pyo3(signature = (x, ys, /))]
pub fn compute_bitwise_hamming_distances<'py>(py: Python<'py>, x: PyReadonlyArray1<u8>, ys: PyReadonlyArray2<u8>) -> Bound<'py, PyArray1<u32>> {
    let x_arr: ArrayView1<u8> = x.as_array();
    let ys_arr: ArrayView2<u8> = ys.as_array();

    let x_vec: Vec<u8> = x_arr.to_vec();
    let hash_size = x_arr.shape()[0];
    let num_cols = ys_arr.shape()[1];

    assert_eq!(hash_size, num_cols);

    let mut out = Vec::new();
    for y in ys_arr.as_slice().unwrap().chunks(hash_size) {
        out.push(hamming_bitwise_fast(x_vec.as_slice(), y));
    };

    out.into_pyarray(py)
}


#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_bitwise_hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_bitwise_hamming_distances, m)?)?;
    Ok(())
}
