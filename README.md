# Phim

A Python library to detect perceptual image duplicates.

## About perceptual hashes

Perceptual hashing algorithms are efficient at detecting near duplicate images which arise, e.g. due to different resolutions or compression rates.
These algorithms work by computing a hash for each image.
This hash works as a sort of fingerprint, and similar images should have similar hashes.
In particular, the hashes are binary vectors, and we count the number of differing entries (or bits) to measure image similarity.

Consider the following simplified example.

There are two images: `img1` and `img2` with these hashes:

```raw
      img1: 10010111
      img2: 10110100
```

We can represent those bit-vectors using two bytes: `151` for `img1` and `180` for `img2`.
In reality, the bit-vectors are longer, so we get additional bytes, which are stored as an array of 8-bit unsigned integers.
However, they differ at three places

```raw
      img1: 10010111
      img2: 10110100
difference: --x---xx
```

so their *Hamming distance* is three.

### Perceptual hashing algorithms

Phim supports two hashing algorithmns: pHash and PDQ-hash.
They are both based on the discrete cosine transform (DCT) and have similar properties.
The main difference is their resolution and which parts of the DCT they use.
The PDQ-hash is a 256-bit perceptual hash algorithm made by Meta aiming for exchanging threat information between companies.
The pHash algorithm is very similar, but it's a bit older and has only 64 bits.

Another difference between the PDQ-hash and the pHash is that the PDQ-hash computes the hash for all eight possible 90 degree rotation/mirror combinations by smartly transforming the DCT.
The same trick could, in theory, be applied during the pHash calculations as well, but it is in general not supported by most implementations (including this one).

Phim has direct bindings for the pHash (via the [`imagehash`](https://docs.rs/imagehash/latest/imagehash/) Rust crate), and it provides a unified interface to the PDQ-hash as well.
However, for the PDQ-hash you also need to install the `pdqhash` Python library, which binds the reference C++ implementation of the PDQ-hash made by the Facebook Threat Exchange.

## Example

### Simple pHash example

```python
import phim
import PIL.Image as Image

img1 = Image.open(...)
img2 = Image.open(...)

phash1 = phim.compute_phash(img1)
phash2 = phim.compute_phash(img2)

differing_bits = phim.compute_hamming_distance(phash1, phash2)
print(f"There are {differing_bits} differing bits between the two pHashes")
```

### Simple PDQ-hash example

```python
import phim
import PIL.Image as Image

img1 = Image.open(...)
img2 = Image.open(...)

pdq_hash_combos1, quality1 = phim.compute_pdq_hash(img1)
pdq_hash_combos2, quality2 = phim.compute_pdq_hash(img2)

differing_bits = phim.compute_hamming_distances(pdq_hash_combos1[0], pdq_hash_combos2)
print(f"There are {min(differing_bits)} differing bits between the two PDQ-hashes")
```

### More intricate example

```python
import phim
import PIL.Image as Image

image_paths = ...
images = [Image.open(p) for p in image_paths]
query_img = Image.open(...)

phash = phim.compute_phash(query_img)
phashes = [phim.compute_phash(img) for img in images]

differing_bits = phim.compute_hamming_distances(phash, phashes)
most_similar_index = differing_bits.argmin()

print(f"The most similar image is {image_paths[most_similar_index]}")
```
