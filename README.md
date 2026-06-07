# 🚢 Implementasi Decision Tree From Scratch - Prediksi Titanic (Kelompok 15)

Repositori ini berisi proyek tugas besar mata kuliah **Kecerdasan Buatan** program studi *Software Engineering*. Proyek ini bertujuan untuk mengimplementasikan algoritma **Decision Tree (Pohon Keputusan)** secara murni (*from scratch*) tanpa menggunakan library Machine Learning eksternal seperti *Scikit-Learn* atau *Pandas*.

Model ini dilatih menggunakan data historis manifes kapal Titanic untuk memprediksi probabilitas kelangsungan hidup penumpang berdasarkan karakteristik individu.

---

## 👥 Anggota Kelompok
1. Rahmadanis Danang Kumala - 103122400066
2. Alfan Didan Septiandri Argandi - 20102225

---

## 🛠️ Fitur & Struktur Algoritma
Sistem ini dibangun menggunakan Python standar dengan memanfaatkan modul bawaan `math` untuk kalkulasi logaritma. Fitur-fitur utama yang berhasil diimplementasikan meliputi:

1. **Robust CSV Parser Manual (`parse_csv_line`):** Mampu memotong baris data `.csv` dengan aman tanpa merusak struktur kolom akibat adanya tanda koma di dalam tanda kutip nama penumpang (Contoh: `"Braund, Mr. Owen Harris"`).
2. **Kalkulasi Entropy (`calculate_entropy`):** Mengukur tingkat keacakan/ketidakmurnian data target `Survived`.
3. **Information Gain (`calculate_information_gain`):** Menentukan bobot fitur terbaik untuk pembelahan simpul (*node splitting*).
4. **Recursive Tree Builder (`build_tree`):** Membangun arsitektur pohon keputusan secara rekursif hingga mencapai *leaf node* (daun keputusan akhir).
5. **Data Binning & Handling Missing Value:** Mengubah data kontinu (`Age`) menjadi data diskrit kategorikal (Anak-anak, Dewasa, Lansia) serta mengisi data kosong dengan nilai median (28 tahun).

---

## 📊 Representasi Pohon Keputusan Terbentuk
Berdasarkan hasil kalkulasi nilai *Information Gain* tertinggi pada data training, alur hierarki logika tebakan yang dihasilkan oleh model adalah sebagai berikut:

```text
[Akar Utama] Apakah Jenis Kelamin Penumpang?
│
├── 👨 Sex = 0 (Laki-laki)
│   └── Cek Kelas Tiket (Pclass):
│       ├── Kelas 1 ──> Umur: Anak-anak (SELAMAT), Dewasa/Lansia (MENINGGAL)
│       ├── Kelas 2 ──> Umur: Anak-anak (SELAMAT), Dewasa/Lansia (MENINGGAL)
│       └── Kelas 3 ──> Semua Umur (MENINGGAL)
│
└── 👩 Sex = 1 (Perempuan)
    └── Cek Kelas Tiket (Pclass):
        ├── Kelas 1 ──> Semua Umur (SELAMAT)
        ├── Kelas 2 ──> Semua Umur (SELAMAT)
        └── Kelas 3 ──> Umur: Dewasa (MENINGGAL), Anak-anak/Lansia (SELAMAT)
```

## 📈 Hasil & Performa Model
* Total Dataset Latihan (train.csv): 891 baris data
* Total Dataset Ujian (test.csv): 418 baris data
* Tingkat Akurasi Evaluasi Internal: 80.25%

Akurasi sebesar 80.25% menunjukkan bahwa model logika murni tanpa library ini mampu menangkap pola sosiologis dan historis nyata dari tragedi Titanic secara objektif.

## 🚀 Cara Menjalankan 
1. Struktur Direktori Folder
Pastikan file data .csv berada dalam satu folder root yang sama dengan file kode utama:
```text
📂 Tubes Kecerdasan Buatan Kelompok 15/
├── 📄 train.csv
├── 📄 test.csv
└── 📄 main.py
```

2. Eksekusi Program
Buka terminal atau command prompt pada direktori tersebut, kemudian jalankan perintah:
```text
py main.py 
```

## Dibuat untuk Tugas Besar Kecerdasan Buatan Semester 4 
## By : Kelompok 15 