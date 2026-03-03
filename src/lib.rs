use pyo3::{prelude::*};
use imagehash;
use image;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadonlyArray3;
use numpy::ndarray::ArrayView1;
use numpy::ndarray::ArrayView2;
use numpy::ndarray::ArrayView3;
use numpy::PyArray1;
use numpy::IntoPyArray;


fn numpy_to_image_buffer(image_array: PyReadonlyArray3<u8>) ->  PyResult<image::RgbImage> {
    let image_view: ArrayView3<u8> = image_array.as_array();
    let height = image_view.shape()[0] as u32;
    let width = image_view.shape()[1] as u32;
    let raw_data = image_view.flatten().to_vec();

    let img: image::RgbImage =
        image::RgbImage::from_vec(width, height, raw_data,)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Failed to create ImageBuffer"))?;

    Ok(img)
}


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



#[pyfunction]
#[pyo3(signature = (image_array, /))]
fn compute_phash<'py>(py: Python<'py>, image_array: PyReadonlyArray3<u8>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    // 1. Get image buffer
    // 2. Convert image buffer to DynamicImage and store in img
    // From DynamicImage docs: DynamicImage::ImageRgb8(rgb)

    let img_buffer: image::RgbImage = numpy_to_image_buffer(image_array)?;
    let img = image::DynamicImage::ImageRgb8(img_buffer);

    let hasher = imagehash::PerceptualHash::new()
        .with_image_size(32, 32)
        .with_hash_size(8, 8)
        .with_resizer(|img, w, h| {
            img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Lanczos3)
        });
    let hash = hasher.hash(&img);
    let bytes = hash.to_bytes();
    Ok(bytes.into_pyarray(py))
}



#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_phash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_bitwise_hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_bitwise_hamming_distances, m)?)?;
    Ok(())
}
