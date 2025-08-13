use pyo3::{prelude::*, BoundObject};
use imagehash;
use image;
use numpy::ndarray::ArrayView3;
use numpy::PyReadonlyArray3;

fn slice_to_u64(bytes: Vec<u8>) -> u64 {
    let slice: &[u8] = &bytes;
    let array: [u8; 8] = slice.try_into().expect("Vec must have exactly 8 elements");
    u64::from_le_bytes(array)
}

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

#[pyfunction]
#[pyo3(signature = (image_array, /))]
fn compute_phash(image_array: PyReadonlyArray3<u8>) -> PyResult<u64> {
    // 1. Get image buffer
    // 2. Convert image buffer to DynamicImage and store in img
    // From DynamicImage docs: DynamicImage::ImageRgb8(rgb)

    let img_buffer: image::RgbImage = numpy_to_image_buffer(image_array)?;
    let img = image::DynamicImage::ImageRgb8(img_buffer);

    let hasher = imagehash::PerceptualHash::new()
        .with_image_size(9, 9)
        .with_hash_size(8, 8)
        .with_resizer(|img, w, h| {
            img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Lanczos3)
        });
    let hash = hasher.hash(&img);
    let bytes = hash.to_bytes();
    Ok(slice_to_u64(bytes))
}

#[pyfunction]
fn save_img(image_array: PyReadonlyArray3<u8>, filename: &str) -> PyResult<()> {
    let img: image::RgbImage = numpy_to_image_buffer(image_array)?;
    img.save(filename).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    Ok(())
}


#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_phash, m)?)?;
    m.add_function(wrap_pyfunction!(save_img, m)?)?;
    Ok(())
}
