"""
model_loader.py
Memuat model ConvNeXt Transfer Learning dari file weights .h5.
Karena model.weights.h5 hanya berisi bobot (bukan full saved model),
arsitektur harus direkonstruksi persis sama dengan yang digunakan saat training.
"""

import os
import logging
import streamlit as st

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

logger = logging.getLogger(__name__)

# Path ke file weights model (relatif terhadap direktori tomatocare/)
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "model.weights.h5")

# Konfigurasi model (identik dengan notebook)
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 10
SEED = 42


def _build_data_augmentation() -> keras.Sequential:
    """
    Membangun layer data augmentation yang identik dengan training.
    Augmentasi ini ada di dalam model — saat inference (training=False),
    augmentasi ini secara otomatis dinonaktifkan oleh Keras.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            layers.RandomRotation(factor=0.15, fill_mode="reflect", seed=SEED),
            layers.RandomZoom(
                height_factor=(-0.1, 0.1),
                width_factor=(-0.1, 0.1),
                fill_mode="reflect",
                seed=SEED,
            ),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="reflect",
                seed=SEED,
            ),
        ],
        name="data_augmentation",
    )


def _build_convnext_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Merekonstruksi arsitektur ConvNeXt Transfer Learning yang identik dengan notebook.

    Pipeline model:
      Input (224, 224, 3) raw float [0-255]
        -> data_augmentation (inactive during inference)
        -> ConvNeXtTiny backbone (include_top=False, imagenet weights)
           [Includes internal ImageNet normalization: mean=[123.675, 116.28, 103.53]]
        -> GlobalAveragePooling2D
        -> Dense(256) + L2(1e-4)
        -> BatchNormalization
        -> Activation('gelu')
        -> Dropout(0.4)
        -> Dense(num_classes, activation='softmax', name='output_layer_convnext')
    """
    data_augmentation = _build_data_augmentation()

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)

    # Backbone ConvNeXtTiny — include_top=False, dengan normalisasi ImageNet internal
    # weights=None karena kita akan load dari .h5 file
    base_model = keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_tensor=x,
    )
    base_model.trainable = True  # Sama seperti saat fine-tuning selesai

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="output_layer_convnext"
    )(x)

    model = models.Model(
        inputs=inputs, outputs=outputs, name="ConvNeXt_Transfer_Learning"
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model() -> keras.Model:
    """
    Memuat model ConvNeXt dengan bobot dari file .h5.
    Fungsi ini di-cache oleh Streamlit sehingga model hanya dimuat sekali
    per proses/session — tidak ada reload berulang.

    Returns:
        Model Keras yang sudah siap untuk inference.

    Raises:
        FileNotFoundError: Jika file weights tidak ditemukan.
        Exception: Jika terjadi error saat membangun atau memuat model.
    """
    weights_path = os.path.abspath(MODEL_WEIGHTS_PATH)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {weights_path}\n"
            "Pastikan file 'model.weights.h5' ada di direktori tomatocare/."
        )

    logger.info(f"Membangun arsitektur model ConvNeXt Transfer Learning...")
    model = _build_convnext_model(num_classes=NUM_CLASSES)

    logger.info(f"Memuat bobot dari: {weights_path}")
    model.load_weights(weights_path)
    logger.info("Model berhasil dimuat.")

    return model
