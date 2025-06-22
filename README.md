# ANPR (Automatic Number Plate Recognition)

## 📌 Deskripsi Proyek
Project ini berhasil mendapatkan peringkat ketiga pada BSS Hackathon #OpenTheGate2025. Proyek ini merupakan implementasi sistem **Automatic Number Plate Recognition (ANPR)** yang berfungsi untuk mendeteksi dan mengenali jenis plat dan kendaraan secara otomatis menggunakan teknologi AI berbasis model ONNX.

---

## 📂 Struktur Proyek
```
ANPR/
├── main.py
├── requirements.txt
├── model/
│   ├── type-plate-classes.json
│   ├── type-plate-detect.onnx
│   ├── type-plate-id2label.json
│   ├── vehicle-plate-classes.json
│   ├── vehicle-plate-detect.onnx
│   └── vehicle-plate-id2label.json
├── static/
│   ├── script.js
│   ├── style.css
│   ├── results/
│   └── uploads/
└── templates/
    └── index.html
```

---

## 🚀 Instalasi
Ikuti langkah-langkah berikut untuk mengatur proyek secara lokal:

1. **Clone Repository**
```bash
git clone <url_repository_ANPR>
cd ANPR
```

2. **Instal Dependensi**
```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Proyek
Untuk menjalankan proyek, gunakan perintah berikut di terminal Anda:

```bash
python main.py
```

Setelah itu buka URL berikut di browser Anda:

```
http://localhost:8000
```

---

## 🤖 Informasi Model
Proyek ini menggunakan model yang disimpan dalam format ONNX (`.onnx`):

| Model                         | Deskripsi                             |
|-------------------------------|---------------------------------------|
| **type-plate-detect.onnx**    | Deteksi jenis plat kendaraan          |
| **vehicle-plate-detect.onnx** | Deteksi kendaraan dan plat nomor      |

File mapping dalam format JSON:

- `type-plate-classes.json`
- `type-plate-id2label.json`
- `vehicle-plate-classes.json`
- `vehicle-plate-id2label.json`

---

## 🎨 UI dan Template
Tampilan pengguna (UI) proyek ini dibuat dengan:

- **HTML/CSS**: Direktori `static/style.css` dan `templates/index.html`
- **JavaScript**: Direktori `static/script.js`

---

## 📝 Note
Proyek ini dibangun menggunakan framework **FastAPI** yang berbasis Python, dikenal karena performanya yang tinggi dan cepat, serta dikombinasikan dengan model **ONNX** yang menawarkan inferensi sangat cepat sehingga cocok untuk integrasi langsung di aplikasi web.
**Perhatian**: Repository ini **tidak menyertakan database**, sehingga kamu mungkin akan mendapatkan error saat menjalankan seluruh proyek secara langsung setelah melakukan clone repository. Pastikan kamu sudah mengonfigurasi database terlebih dahulu atau menyesuaikan kode yang ada sesuai kebutuhanmu.

---

## ✍️ Penyesuaian dan Kontribusi
Jika ada saran atau ingin berkontribusi, silakan ajukan **Pull Request** atau buka **issue** pada repository ini.
