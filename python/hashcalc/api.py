import PIL.Image as Image
from fastapi import FastAPI, UploadFile

from hashcalc.hashcalc import compute_phash, hamming_distance

app = FastAPI()


@app.get("/")
async def hello(name: str) -> str:
    return f"Hello {name}"


@app.post("/get_hash/")
async def upload_file(file: UploadFile) -> int:
    with Image.open(file.file) as img:
        return compute_phash(img)


@app.post("/compare_images/")
async def compare_images(file1: UploadFile, file2: UploadFile) -> int:
    with Image.open(file1.file) as img1, Image.open(file2.file) as img2:
        return hamming_distance(compute_phash(img1), compute_phash(img2))


@app.get("/hamming_distance/{hash1}/{hash2}")
async def calc_hamming_distance(hash1: int, hash2: int) -> int:
    return hamming_distance(hash1, hash2)
