from __future__ import annotations

import io
import unittest

from PIL import Image

from src.vision.image_encoder import ImageValidationError, validate_uploaded_image


def image_bytes(image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (12, 8), (12, 120, 220, 160)).save(output, format=image_format)
    return output.getvalue()


class ImageUploadValidationTests(unittest.TestCase):
    def test_png_is_validated_and_converted_to_rgb(self) -> None:
        image = validate_uploaded_image(image_bytes("PNG"), filename="query.png")
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))

    def test_invalid_bytes_are_rejected_with_specific_error(self) -> None:
        with self.assertRaisesRegex(ImageValidationError, "Pillow không đọc được ảnh"):
            validate_uploaded_image(b"not-an-image", filename="query.jpg")

    def test_unsupported_extension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ImageValidationError, "JPG, PNG hoặc WebP"):
            validate_uploaded_image(image_bytes("PNG"), filename="query.gif")

    def test_file_over_five_megabytes_is_rejected_before_decode(self) -> None:
        with self.assertRaisesRegex(ImageValidationError, "vượt giới hạn"):
            validate_uploaded_image(b"x" * 101, filename="query.jpg", max_bytes=100)


if __name__ == "__main__":
    unittest.main()

