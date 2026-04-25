import io
from collections.abc import Generator
from pathlib import Path

import PIL.Image as Image
import pytest


def open_image(image_path: Path) -> Image.Image:
    return Image.open(io.BytesIO(image_path.read_bytes()))


@pytest.fixture
def data_directory() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def sun_image(data_directory) -> Image.Image:
    return open_image(data_directory / "sun/sun.png")


@pytest.fixture
def sun_images_jpg_compressed(data_directory) -> Generator[Image.Image, None, None]:
    """Yields JPEG-compressed versions of the sun image.

    The images are arranged by quality in order from highest to lowest:
        - sun.q100.jpg - compressed with quality = 100%
        - sun.q085.jpg - compressed with quality = 85%
        - sun.q050.jpg - compressed with quality = 50%
        - sun.q010.jpg - compressed with quality = 10%
        - sun.q001.jpg - compressed with quality = 1%
    """
    return (open_image(img_path) for img_path in sorted(data_directory.glob("sun/sun*.jpg"), reverse=True))


@pytest.fixture
def moon_image(data_directory) -> Image.Image:
    return open_image(data_directory / "moon/moon.png")


@pytest.fixture
def moon_images_jpg_compressed(data_directory) -> Generator[Image.Image, None, None]:
    """
    Yields JPEG-compressed versions of the moon image.
    The images are arranged by quality in order from highest to lowest:
        - moon.q100.jpg - compressed with quality = 100%
        - moon.q085.jpg - compressed with quality = 85%
        - moon.q050.jpg - compressed with quality = 50%
        - moon.q010.jpg - compressed with quality = 10%
        - moon.q001.jpg - compressed with quality = 1%
    """
    return (open_image(img_path) for img_path in sorted(data_directory.glob("moon/moon*.jpg"), reverse=True))


@pytest.fixture
def circles_image(data_directory) -> Image.Image:
    return open_image(data_directory / "other/circles.jpg")


@pytest.fixture
def ocean_image(data_directory) -> Image.Image:
    return open_image(data_directory / "other/ocean.jpg")


@pytest.fixture
def black_spiral_image(data_directory) -> Image.Image:
    return open_image(data_directory / "other/spiral_b.png")


@pytest.fixture
def white_spiral_image(data_directory) -> Image.Image:
    return open_image(data_directory / "other/spiral_w.png")
