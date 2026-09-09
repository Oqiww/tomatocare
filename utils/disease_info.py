"""
disease_info.py
Basis informasi penyakit daun tomat untuk seluruh 10 kelas model.
Urutan kelas mengikuti sorted alphabetical order dari folder dataset.
"""

# Mapping indeks ke nama kelas (sesuai urutan sorted folder dataset)
CLASS_NAMES = [
    "Tomato___Bacterial_spot",                          # 0
    "Tomato___Early_blight",                            # 1
    "Tomato___Late_blight",                             # 2
    "Tomato___Leaf_Mold",                               # 3
    "Tomato___Septoria_leaf_spot",                      # 4
    "Tomato___Spider_mites Two-spotted_spider_mite",    # 5
    "Tomato___Target_Spot",                             # 6
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",           # 7
    "Tomato___Tomato_mosaic_virus",                     # 8
    "Tomato___healthy",                                 # 9
]

# Label tampilan yang ramah pengguna
DISPLAY_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites (Two-spotted)",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy",
}

# Informasi lengkap setiap penyakit
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "display_name": "Bacterial Spot",
        "icon": "Bakteri",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Bacterial Spot disebabkan oleh bakteri Xanthomonas spp. yang menyerang "
            "daun, batang, dan buah tomat. Kondisi lembap dan suhu hangat mendukung "
            "penyebaran penyakit ini."
        ),
        "visual_signs": [
            "Bercak kecil berair berwarna cokelat tua atau hitam pada daun",
            "Tepi bercak sering berwarna kuning (halo kuning)",
            "Bercak dapat bergabung dan menyebabkan daun rontok lebih awal",
            "Permukaan daun tampak seperti 'terbakar' di area yang parah",
        ],
        "general_actions": [
            "Periksa dan singkirkan daun yang sudah sangat terinfeksi",
            "Hindari menyiram dari atas agar daun tidak selalu basah",
            "Pastikan tanaman mendapat sirkulasi udara yang baik",
            "Lakukan rotasi tanaman pada musim tanam berikutnya",
            "Konsultasikan dengan ahli pertanian setempat untuk penanganan lanjutan",
        ],
        "note": "Hindari bekerja di kebun saat tanaman basah untuk mencegah penyebaran.",
    },
    "Tomato___Early_blight": {
        "display_name": "Early Blight",
        "icon": "Jamur",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Early Blight disebabkan oleh jamur Alternaria solani. Penyakit ini "
            "umumnya mulai dari daun bagian bawah yang lebih tua, lalu menyebar "
            "ke atas seiring waktu."
        ),
        "visual_signs": [
            "Bercak gelap berbentuk bulat dengan pola cincin konsentris (seperti sasaran panah)",
            "Halo kuning mengelilingi area bercak",
            "Daun yang terinfeksi parah akan menguning dan rontok",
            "Biasanya dimulai dari daun bagian bawah tanaman",
        ],
        "general_actions": [
            "Singkirkan daun yang terinfeksi parah dari tanaman",
            "Buang sisa daun jatuh dari sekitar tanaman",
            "Pastikan tanaman tidak terlalu rapat agar udara bersirkulasi",
            "Hindari menyiram berlebihan dan pastikan drainase tanah baik",
            "Konsultasikan dengan penyuluh pertanian untuk rekomendasi penanganan",
        ],
        "note": "Early blight paling aktif saat cuaca hangat dan lembap.",
    },
    "Tomato___Late_blight": {
        "display_name": "Late Blight",
        "icon": "Jamur Air",
        "status": "disease",
        "severity": "high",
        "description": (
            "Late Blight disebabkan oleh Phytophthora infestans, patogen yang "
            "bertanggung jawab atas kelaparan besar di Irlandia pada abad ke-19. "
            "Penyakit ini dapat menyebar sangat cepat dalam kondisi dingin dan lembap."
        ),
        "visual_signs": [
            "Bercak besar berwarna cokelat kehijauan atau abu-abu, tampak 'berlemak'",
            "Bagian bawah daun sering menunjukkan lapisan putih berbulu (spora jamur)",
            "Daun cepat mengering dan hitam",
            "Batang dan buah juga dapat terinfeksi",
        ],
        "general_actions": [
            "Isolasi tanaman yang terinfeksi untuk mencegah penyebaran cepat",
            "Singkirkan bagian tanaman yang terinfeksi dengan hati-hati",
            "Hindari membuang sisa tanaman di area kebun",
            "Kurangi kelembapan dengan memperbaiki drainase dan sirkulasi udara",
            "Segera konsultasikan dengan ahli pertanian karena late blight dapat menyebar sangat cepat",
        ],
        "note": "Peringatan: Late blight bisa menyebar cepat. Penanganan dini sangat disarankan.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Leaf Mold",
        "icon": "Jamur",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Leaf Mold disebabkan oleh jamur Passalora fulva. "
            "Penyakit ini paling umum terjadi di lingkungan dengan kelembapan tinggi, "
            "seperti rumah kaca atau daerah dengan cuaca lembap."
        ),
        "visual_signs": [
            "Bercak berwarna kuning pucat pada permukaan atas daun",
            "Lapisan jamur berwarna abu-abu kehijauan atau cokelat pada bagian bawah daun",
            "Daun dapat menguning dan rontok jika infeksi parah",
            "Pola bercak biasanya tidak beraturan",
        ],
        "general_actions": [
            "Kurangi kelembapan di sekitar tanaman",
            "Tingkatkan sirkulasi udara antar tanaman",
            "Hindari menyiram dari atas, siram langsung ke tanah",
            "Singkirkan daun yang terinfeksi parah",
            "Konsultasikan dengan penyuluh pertanian untuk penanganan lebih lanjut",
        ],
        "note": "Penyakit ini sangat bergantung pada kelembapan. Menjaga udara tetap kering adalah kunci pencegahan.",
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Septoria Leaf Spot",
        "icon": "Jamur",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Septoria Leaf Spot disebabkan oleh jamur Septoria lycopersici. "
            "Penyakit ini umumnya muncul setelah bunga pertama muncul dan "
            "menyebar dari bawah ke atas tanaman."
        ),
        "visual_signs": [
            "Bercak kecil bulat berwarna putih atau krem dengan tepi cokelat gelap",
            "Titik hitam kecil (pycnidia) terlihat di tengah bercak",
            "Daun yang terinfeksi menguning di sekitar bercak",
            "Dimulai dari daun bawah dan menyebar ke atas",
        ],
        "general_actions": [
            "Singkirkan daun yang terinfeksi segera setelah terdeteksi",
            "Hindari membasahi daun saat menyiram",
            "Pastikan jarak tanam cukup untuk sirkulasi udara yang baik",
            "Bersihkan sisa tanaman di akhir musim tanam",
            "Konsultasikan dengan ahli pertanian setempat",
        ],
        "note": "Septoria menyebar melalui air percikan. Metode penyiraman sangat penting.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Spider Mites (Two-spotted)",
        "icon": "Hama",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Tungau laba-laba bintik dua (Tetranychus urticae) adalah hama kecil "
            "yang mengisap cairan dari sel daun tomat. Hama ini sangat kecil dan "
            "sulit terlihat dengan mata telanjang, tetapi jaring-jaring halusnya "
            "sering terlihat di bawah daun."
        ),
        "visual_signs": [
            "Daun tampak berbintik-bintik atau berwarna keperakan/kuning pucat",
            "Jaring-jaring halus pada bagian bawah daun",
            "Daun dapat menguning, mengering, dan rontok",
            "Hama kecil berwarna merah/hijau/kuning terlihat di bawah daun",
        ],
        "general_actions": [
            "Periksa bagian bawah daun secara rutin",
            "Semprotkan air ke bawah daun untuk mengurangi populasi",
            "Singkirkan daun yang terinfeksi parah",
            "Pastikan tanaman tidak stres akibat kekeringan",
            "Konsultasikan dengan ahli pertanian untuk penanganan hama yang tepat",
        ],
        "note": "Tungau berkembang biak cepat dalam kondisi panas dan kering.",
    },
    "Tomato___Target_Spot": {
        "display_name": "Target Spot",
        "icon": "Jamur",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Target Spot disebabkan oleh jamur Corynespora cassiicola. "
            "Penyakit ini dapat menyerang daun, batang, dan buah tomat, "
            "dan sering terlihat mirip dengan early blight."
        ),
        "visual_signs": [
            "Bercak gelap berbentuk tidak beraturan dengan pola cincin konsentris",
            "Ukuran bercak lebih besar dan lebih tidak beraturan dibanding early blight",
            "Halo kuning atau cokelat pucat di sekitar bercak",
            "Daun yang terinfeksi parah cepat menguning dan rontok",
        ],
        "general_actions": [
            "Singkirkan daun yang sangat terinfeksi",
            "Pastikan sirkulasi udara yang baik di sekitar tanaman",
            "Hindari kelembapan berlebih pada daun",
            "Lakukan inspeksi rutin terutama pada daun bagian bawah",
            "Konsultasikan dengan penyuluh pertanian untuk rekomendasi penanganan",
        ],
        "note": "Target spot dapat terlihat mirip early blight. Identifikasi yang akurat penting.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Yellow Leaf Curl Virus",
        "icon": "Virus",
        "status": "disease",
        "severity": "high",
        "description": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) adalah virus yang ditularkan "
            "oleh kutu kebul (Bemisia tabaci). Virus ini tidak dapat disembuhkan "
            "setelah tanaman terinfeksi, sehingga pencegahan sangat penting."
        ),
        "visual_signs": [
            "Daun muda menggulung ke atas dan ke dalam",
            "Daun berwarna kuning dengan tepi yang menggulung",
            "Tanaman tampak kerdil dan pertumbuhan terhambat",
            "Bunga dapat rontok dan produksi buah berkurang drastis",
        ],
        "general_actions": [
            "Identifikasi dan kendalikan populasi kutu kebul di sekitar tanaman",
            "Singkirkan tanaman yang sangat terinfeksi untuk mencegah penyebaran",
            "Gunakan penutup tanaman atau jaring insect-proof jika memungkinkan",
            "Laporkan ke penyuluh pertanian karena TYLCV adalah penyakit serius",
            "Pilih varietas tomat yang tahan TYLCV untuk musim tanam berikutnya",
        ],
        "note": "Peringatan: Virus ini tidak dapat disembuhkan. Pengendalian vektor (kutu kebul) adalah prioritas.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "icon": "Virus",
        "status": "disease",
        "severity": "moderate",
        "description": (
            "Tomato Mosaic Virus (ToMV) adalah virus yang menyebar melalui kontak "
            "mekanis, benih terinfeksi, atau serangga. Virus ini dapat bertahan "
            "lama di tanah dan pada peralatan berkebun."
        ),
        "visual_signs": [
            "Pola mozaik kuning-hijau yang tidak merata pada daun",
            "Daun tampak berkerut, menggulung, atau terdistorsi",
            "Warna daun tidak merata antara hijau tua dan kuning-hijau pucat",
            "Pertumbuhan tanaman menjadi terhambat",
        ],
        "general_actions": [
            "Cuci tangan sebelum dan sesudah menangani tanaman",
            "Bersihkan alat berkebun secara rutin",
            "Singkirkan tanaman yang terinfeksi parah",
            "Kendalikan serangga yang dapat menjadi vektor penyebaran",
            "Konsultasikan dengan ahli pertanian untuk penanganan lebih lanjut",
        ],
        "note": "Virus ini menyebar melalui sentuhan. Kebersihan alat dan tangan sangat penting.",
    },
    "Tomato___healthy": {
        "display_name": "Healthy",
        "icon": "Sehat",
        "status": "healthy",
        "severity": "none",
        "description": (
            "Daun tomat ini tampak sehat berdasarkan analisis model AI. "
            "Daun sehat umumnya berwarna hijau merata tanpa bercak, "
            "perubahan warna tidak normal, atau tanda-tanda kerusakan."
        ),
        "visual_signs": [
            "Warna hijau merata dan cerah",
            "Tidak ada bercak, lesion, atau perubahan warna abnormal",
            "Tekstur daun tampak normal dan tidak terdistorsi",
            "Tidak ada tanda-tanda hama atau penyakit",
        ],
        "general_actions": [
            "Lanjutkan perawatan rutin seperti biasa",
            "Pantau tanaman secara berkala untuk deteksi dini masalah",
            "Pastikan nutrisi, air, dan cahaya matahari tercukupi",
            "Jaga kebersihan area tanam untuk mencegah penyakit",
        ],
        "note": "Tetap lakukan pemantauan rutin untuk menjaga kesehatan tanaman.",
    },
}


def get_display_name(class_name: str) -> str:
    """Dapatkan nama tampilan yang ramah pengguna dari nama kelas model."""
    return DISPLAY_NAMES.get(class_name, class_name.replace("Tomato___", "").replace("_", " "))


def get_disease_info(class_name: str) -> dict:
    """Dapatkan informasi lengkap penyakit berdasarkan nama kelas model."""
    return DISEASE_INFO.get(class_name, {
        "display_name": get_display_name(class_name),
        "icon": "Unknown",
        "status": "unknown",
        "severity": "unknown",
        "description": "Informasi untuk kondisi ini belum tersedia.",
        "visual_signs": [],
        "general_actions": ["Konsultasikan dengan ahli pertanian setempat."],
        "note": "",
    })


def get_severity_color(severity: str) -> str:
    """Dapatkan warna CSS berdasarkan tingkat keparahan."""
    colors = {
        "none": "#2d6a4f",
        "moderate": "#e67e22",
        "high": "#c0392b",
        "unknown": "#7f8c8d",
    }
    return colors.get(severity, "#7f8c8d")
