"""
prediction.py
Menjalankan inference model dan memformat hasil prediksi.
"""

import logging
from typing import Optional, List, Tuple

import numpy as np

from utils.disease_info import CLASS_NAMES, get_display_name

logger = logging.getLogger(__name__)

# Threshold confidence
LOW_CONFIDENCE_THRESHOLD = 0.50    # Di bawah ini: low confidence warning
MODERATE_CONFIDENCE_THRESHOLD = 0.75  # Di bawah ini: moderate


def predict(model, preprocessed_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Jalankan inference model.

    Args:
        model: Model Keras yang sudah dimuat
        preprocessed_image: numpy array shape (1, 224, 224, 3) float32

    Returns:
        Array probabilitas shape (num_classes,), atau None jika error
    """
    try:
        predictions = model(preprocessed_image, training=False)
        probabilities = predictions.numpy()[0]  # Shape: (10,)
        return probabilities
    except Exception as e:
        logger.error(f"Error saat inference: {e}")
        return None


def format_prediction(probabilities: np.ndarray) -> dict:
    """
    Format hasil prediksi menjadi struktur yang siap digunakan UI.

    Returns:
        dict dengan keys:
          - predicted_class: nama kelas (string)
          - display_name: nama tampilan user-friendly
          - confidence: float 0-1
          - confidence_pct: string formatted percentage
          - top3: list of (class_name, display_name, confidence) tuples, sorted desc
          - all_probs: list of (class_name, display_name, confidence) untuk semua kelas
          - confidence_level: "high" | "moderate" | "low"
          - confidence_label: string label dalam Bahasa Indonesia
          - confidence_description: deskripsi interpretasi
    """
    top_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[top_idx]
    display_name = get_display_name(predicted_class)
    confidence = float(probabilities[top_idx])
    confidence_pct = f"{confidence * 100:.2f}%"

    # Urutkan semua kelas berdasarkan confidence (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    all_probs = [
        (CLASS_NAMES[i], get_display_name(CLASS_NAMES[i]), float(probabilities[i]))
        for i in sorted_indices
    ]
    top3 = all_probs[:3]

    # Confidence interpretation
    if confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        confidence_level = "high"
        confidence_label = "Confidence Tinggi"
        confidence_description = (
            "Model memiliki keyakinan yang kuat terhadap prediksi ini."
        )
    elif confidence >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "moderate"
        confidence_label = "Confidence Sedang"
        confidence_description = (
            "Model menunjukkan ketidakpastian di antara beberapa kondisi yang serupa."
        )
    else:
        confidence_level = "low"
        confidence_label = "Confidence Rendah"
        confidence_description = (
            "Model kurang yakin dengan prediksi ini. "
            "Gambar mungkin kurang jelas, pencahayaan kurang baik, "
            "atau daun tidak terlihat dengan jelas."
        )

    return {
        "predicted_class": predicted_class,
        "display_name": display_name,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "top3": top3,
        "all_probs": all_probs,
        "confidence_level": confidence_level,
        "confidence_label": confidence_label,
        "confidence_description": confidence_description,
    }


def get_confidence_color(confidence_level: str) -> str:
    """Dapatkan warna berdasarkan level confidence."""
    colors = {
        "high": "#2d6a4f",
        "moderate": "#e67e22",
        "low": "#c0392b",
    }
    return colors.get(confidence_level, "#7f8c8d")
