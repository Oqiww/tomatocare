# 🍅 TomatoCare — AI-Powered Tomato Leaf Disease Detection

Aplikasi Streamlit untuk deteksi penyakit daun tomat menggunakan model ConvNeXt Transfer Learning yang telah dilatih.

---

## Fitur Utama

- **Deteksi AI** — Upload gambar daun tomat dan dapatkan prediksi instan dari model ConvNeXt
- **Confidence Score** — Lihat seberapa yakin model dengan prediksi, lengkap dengan interpretasi
- **Top-3 Predictions** — Transparansi penuh dengan 3 kondisi paling mungkin
- **Informasi Penyakit** — Deskripsi, tanda visual, dan langkah penanganan untuk setiap kondisi
- **Panduan Penyakit** — Eksplorasi seluruh 10 kelas kondisi yang dapat dideteksi
- **Image Quality Check** — Peringatan otomatis jika kualitas gambar dapat mempengaruhi prediksi
- **Low Confidence Warning** — Peringatan jika model kurang yakin dengan prediksinya
- **Riwayat Sesi** — Lihat prediksi sebelumnya selama sesi berlangsung
- **Download Laporan** — Unduh laporan prediksi dalam format teks
- **About Model** — Informasi teknis lengkap tentang model

---

## Model

| Properti | Nilai |
|---|---|
| Arsitektur | ConvNeXtTiny Transfer Learning |
| Framework | TensorFlow / Keras 3.13.2 |
| Resolusi Input | 224 × 224 × 3 |
| Jumlah Kelas | 10 |
| Aktivasi Output | Softmax |
| Akurasi Validasi | 91.90% |
| Macro F1-Score | 91.91% |

### Kelas yang Dapat Dideteksi (Urutan Index)

| Index | Nama Kelas | Tampilan |
|---|---|---|
| 0 | Tomato___Bacterial_spot | Bacterial Spot |
| 1 | Tomato___Early_blight | Early Blight |
| 2 | Tomato___Late_blight | Late Blight |
| 3 | Tomato___Leaf_Mold | Leaf Mold |
| 4 | Tomato___Septoria_leaf_spot | Septoria Leaf Spot |
| 5 | Tomato___Spider_mites Two-spotted_spider_mite | Spider Mites (Two-spotted) |
| 6 | Tomato___Target_Spot | Target Spot |
| 7 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Yellow Leaf Curl Virus |
| 8 | Tomato___Tomato_mosaic_virus | Tomato Mosaic Virus |
| 9 | Tomato___healthy | Healthy |

---

## Struktur Proyek

```
tomatocare/
├── app.py                    # Aplikasi Streamlit utama
├── model.weights.h5          # File weights model (diperlukan)
├── requirements.txt          # Dependencies Python
├── README.md
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py       # Rekonstruksi arsitektur + load weights
│   ├── preprocessing.py      # Pipeline preprocessing gambar
│   ├── prediction.py         # Inference dan format hasil
│   └── disease_info.py       # Database informasi penyakit
│
└── assets/                   # Aset statis (opsional)
```

---

## Instalasi & Menjalankan

### Prasyarat
- Python 3.10+
- pip

### 1. Persiapkan file model

Salin file `model.weights.h5` ke dalam folder `tomatocare/`:

```
tomatocare/
└── model.weights.h5   ← file ini
```

### 2. Install dependencies

```bash
cd tomatocare
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`.

---

## Model Loading

File `model.weights.h5` adalah file **weights-only** (bukan full saved model).
Oleh karena itu, arsitektur model direkonstruksi secara programatik di `utils/model_loader.py`
sebelum bobot dimuat dengan `model.load_weights()`.

Arsitektur yang direkonstruksi identik dengan yang digunakan saat training:

```python
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)                    # RandomFlip, Rotation, Zoom, Translation
base_model = keras.applications.ConvNeXtTiny(    # Backbone (weights=None, akan di-load dari .h5)
    include_top=False, weights=None, input_tensor=x
)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('gelu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation='softmax', name='output_layer_convnext')(x)
```

### Preprocessing

Input model adalah **raw pixel values [0, 255] sebagai float32**.
Normalisasi ImageNet dilakukan oleh layer internal model (`convnext_tiny_prestem_normalization`).

```python
# Contoh preprocessing (tidak ada rescaling ke [0,1])
img = PIL.Image.open(path).convert("RGB")
img = img.resize((224, 224), Image.BILINEAR)
img_array = np.array(img, dtype=np.float32)        # Range: [0, 255]
img_batch = np.expand_dims(img_array, axis=0)      # Shape: (1, 224, 224, 3)
predictions = model(img_batch, training=False)     # Softmax output
```

---

## Deployment

### Lokal

```bash
streamlit run app.py
```

---

## Keterbatasan

- Model hanya dapat mengidentifikasi 10 kelas penyakit daun tomat yang ada dalam dataset training
- Akurasi 91.9% berarti sekitar 8 dari 100 prediksi dapat tidak akurat
- Kualitas gambar sangat mempengaruhi hasil prediksi
- Model tidak menggantikan diagnosa dari ahli pertanian
- Dataset menggunakan gambar laboratorium — kondisi lapangan berbeda dapat mempengaruhi akurasi

---

## Disclaimer

> Aplikasi ini dibuat untuk tujuan **edukatif dan informasi umum**. Prediksi model AI tidak boleh dijadikan sebagai satu-satunya dasar diagnosis pertanian. Selalu konsultasikan dengan ahli pertanian atau penyuluh pertanian yang berpengalaman untuk penanganan yang tepat.
