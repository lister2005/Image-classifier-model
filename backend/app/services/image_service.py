from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.core.config import MAX_IMAGE_PIXELS, SUPPORTED_CONTENT_TYPES


class InvalidImageError(ValueError):
    pass


def decode_image(contents: bytes, content_type: str) -> Image.Image:
    expected_format = SUPPORTED_CONTENT_TYPES.get(content_type)
    if expected_format is None:
        raise InvalidImageError("Unsupported image type. Upload a JPEG or PNG image.")

    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format != expected_format:
                raise InvalidImageError("The uploaded image does not match its declared type.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image dimensions are too large to process.")
            image.verify()

        with Image.open(BytesIO(contents)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("Image dimensions are too large to process.")
            return image.convert("RGB")
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a valid JPEG or PNG image.") from exc
