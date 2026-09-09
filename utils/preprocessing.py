"""
preprocessing.py
Pipeline preprocessing gambar untuk inference ConvNeXt TomatoCare.

PENTING:
- Input ke model adalah raw pixel values [0, 255] sebagai float32
- Normalisasi ImageNet (mean/std) dilakukan OLEH LAYER INTERNAL MODEL
  (convnext_tiny_prestem_normalization), bukan di sini
- Tidak ada rescaling ke [0, 1] sebelum model
- Hanya: decode -> resize ke (224, 224) -> expand_dims untuk batch
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Konfigurasi preprocessing (identik dengan notebook)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
MAX_FILE_SIZE_MB = 10
MIN_DIMENSION = 50   # Minimum dimension diterima (pixels)
MAX_DIMENSION = 8000  # Maximum dimension (sangat besar mungkin bermasalah)


def validate_image_file(
    uploaded_file,
) -> Tuple[bool, str]:
    """
    Validasi file upload sebelum preprocessing.

    Returns:
        (is_valid, error_message)
    """
    # Check file size
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"Ukuran file terlalu besar ({file_size_mb:.1f} MB). Maksimum {MAX_FILE_SIZE_MB} MB."

    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    filename = uploaded_file.name.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in allowed_extensions:
        return False, f"Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau WEBP."

    # Try opening image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Verify not corrupted
    except Exception:
        return False, "File gambar tidak dapat dibaca. Pastikan file tidak rusak."

    # Check dimensions
    try:
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            return (
                False,
                f"Gambar terlalu kecil ({w}x{h} px). Ukuran minimum adalah {MIN_DIMENSION}x{MIN_DIMENSION} px.",
            )
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            return (
                False,
                f"Gambar terlalu besar ({w}x{h} px). Harap gunakan gambar yang lebih kecil.",
            )
    except Exception:
        return False, "Tidak dapat membaca dimensi gambar."

    return True, ""


def get_image_quality_warnings(pil_image: Image.Image) -> list:
    """
    Cek kualitas gambar secara lightweight dan kembalikan daftar peringatan.
    Peringatan tidak menghentikan inference — hanya informatif.
    """
    warnings = []

    try:
        # Convert ke numpy untuk analisis
        img_array = np.array(pil_image.convert("RGB"), dtype=np.float32)

        # Check brightness
        mean_brightness = img_array.mean()
        if mean_brightness < 30:
            warnings.append("Gambar tampak terlalu gelap. Ini dapat mempengaruhi akurasi prediksi.")
        elif mean_brightness > 240:
            warnings.append("Gambar tampak terlalu terang/overexposed. Ini dapat mempengaruhi akurasi prediksi.")

        # Check low contrast (std deviation)
        std_dev = img_array.std()
        if std_dev < 15:
            warnings.append("Kontras gambar sangat rendah. Coba gunakan gambar dengan pencahayaan lebih baik.")

        # Check image size vs minimum recommended
        w, h = pil_image.size
        if w < 100 or h < 100:
            warnings.append(f"Resolusi gambar rendah ({w}x{h} px). Hasil lebih baik dengan gambar yang lebih besar.")

    except Exception as e:
        logger.warning(f"Tidak dapat melakukan quality check: {e}")

    return warnings


def preprocess_image(uploaded_file) -> Optional[np.ndarray]:
    """
    Preprocessing gambar untuk inference ConvNeXt.

    Pipeline (identik dengan notebook training):
    1. Buka dengan PIL dan konversi ke RGB
    2. Resize ke (IMG_HEIGHT, IMG_WIDTH) = (224, 224)
    3. Konversi ke float32 (range [0, 255])
    4. Expand dims untuk membentuk batch shape (1, 224, 224, 3)

    TIDAK ada rescaling ke [0,1] karena normalisasi dilakukan oleh model.

    Returns:
        numpy array shape (1, 224, 224, 3) float32 [0-255], atau None jika error
    """
    try:
        file_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # Resize ke target size
        pil_image_resized = pil_image.resize(
            (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR
        )

        # Konversi ke numpy float32 [0, 255]
        img_array = np.array(pil_image_resized, dtype=np.float32)

        # Tambahkan batch dimension: (H, W, C) -> (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)

        logger.debug(
            f"Preprocessing selesai: shape={img_batch.shape}, "
            f"range=[{img_batch.min():.1f}, {img_batch.max():.1f}]"
        )

        return img_batch

    except Exception as e:
        logger.error(f"Error saat preprocessing gambar: {e}")
        return None


def load_pil_image(uploaded_file) -> Optional[Image.Image]:
    """
    Buka file upload sebagai PIL Image untuk ditampilkan di UI.

    Returns:
        PIL Image (RGB), atau None jika gagal
    """
    try:
        file_bytes = uploaded_file.getvalue()
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Error membuka gambar untuk preview: {e}")
        return None


def get_image_info(uploaded_file) -> dict:
    """
    Dapatkan informasi metadata gambar untuk ditampilkan di UI.

    Returns:
        dict dengan keys: width, height, file_size_kb, filename
    """
    try:
        file_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(file_bytes))
        w, h = pil_image.size
        return {
            "width": w,
            "height": h,
            "file_size_kb": len(file_bytes) / 1024,
            "filename": uploaded_file.name,
            "mode": pil_image.mode,
        }
    except Exception:
        return {
            "width": 0,
            "height": 0,
            "file_size_kb": 0,
            "filename": getattr(uploaded_file, "name", "unknown"),
            "mode": "unknown",
        }
