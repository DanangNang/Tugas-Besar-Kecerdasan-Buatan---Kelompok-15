# Tugas Besar Kecerdasan Buatan - Kelompok 15
## Sistem Prediksi Survival Titanic dengan Decision Tree (From Scratch)

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan algoritma **Decision Tree** dari awal (without ML libraries) untuk memprediksi keselamatan penumpang Titanic berdasarkan karakteristik mereka seperti jenis kelamin, kelas tiket, dan usia.

**Tujuan Pembelajaran:**
- Memahami konsep Entropy dan Information Gain
- Mengimplementasi algoritma ID3 secara manual
- Membangun decision tree secara rekursif
- Melakukan prediksi dengan tree traversal

---

## 🎯 Fitur Utama

- ✅ **CSV Parser Manual** - Membaca CSV tanpa pandas
- ✅ **Entropy Calculation** - Menghitung tingkat keacakan data
- ✅ **Information Gain** - Menentukan atribut terbaik untuk split
- ✅ **ID3 Algorithm** - Membangun decision tree secara rekursif
- ✅ **Prediction System** - Tree traversal untuk prediksi
- ✅ **Visualization** - Tampilan struktur tree yang jelas
- ✅ **Testing Suite** - Unit test untuk semua fungsi

---

## 📁 Struktur File

```
Tugas-Besar-Kecerdasan-Buatan---Kelompok-15/
│
├── main.py                      # Program utama
├── train.csv                    # Dataset training
├── test.csv                     # Dataset testing
│
├── README.md                    # Dokumentasi proyek (file ini)
├── PENJELASAN_ALGORITMA.md      # Penjelasan detail algoritma
└── RINGKASAN_IMPLEMENTASI.txt   # Ringkasan hasil implementasi
```

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.7 atau lebih baru
- Tidak memerlukan library eksternal (pure Python)

### Langkah-langkah

1. **Clone atau download repository**
   ```bash
   cd Tugas-Besar-Kecerdasan-Buatan---Kelompok-15
   ```

2. **Jalankan program utama**
   ```bash
   python main.py
   ```

3. **Jalankan test (opsional)**
   ```bash
   python test_functions.py
   ```

---

## 📊 Output Program

### 1. Struktur Decision Tree
Program akan menampilkan:
- Dictionary representation dari tree
- Visualisasi hierarki tree yang mudah dibaca

### 2. Akurasi Model
- Akurasi pada training data: **80.25%**

### 3. Prediksi Test Data
- Tabel prediksi untuk 15 penumpang pertama
- Format: PassengerId, Hasil (0/1), Status (Selamat/Meninggal)

### Contoh Output:
```
=================================================================
      SISTEM PREDIKSI TITANIC SCRATCH - KELOMPOK 15      
=================================================================
[1/4] Membaca data dari file CSV...
 -> Berhasil memuat 891 data Latihan (Train).
 -> Berhasil memuat 418 data Ujian (Test).

[2/4] Membuat Model Pohon Keputusan (Proses Belajar)...

[HASIL] Struktur Pohon Keputusan yang Dihasilkan:
-----------------------------------------------------------------
Pohon Keputusan (Format Dictionary):
{'Sex': {0: {'Pclass': {...}}, 1: {'Pclass': {...}}}}
-----------------------------------------------------------------

[3/4] Mengukur tingkat akurasi model...
 -> Tingkat Akurasi Model Kelompok Kita: 80.25%

[4/4] Menebak keselamatan penumpang di data 'test.csv'...
-----------------------------------------------------------------
No    | PassengerId  | Hasil   | Status Akhir Penumpang
-----------------------------------------------------------------
1     | 892          | 0       |  Meninggal Dunia      
2     | 893          | 0       |  Meninggal Dunia      
...
```

---

## 🧮 Algoritma yang Diimplementasikan

### 1. **Calculate Entropy**
```python
Entropy(S) = -p_survived * log2(p_survived) - p_dead * log2(p_dead)
```
- Mengukur tingkat keacakan/ketidakteraturan data
- Entropy = 0 → Data homogen
- Entropy = 1 → Data seimbang 50:50

### 2. **Calculate Information Gain**
```python
Gain(S, A) = Entropy(S) - Σ (|Sv|/|S|) * Entropy(Sv)
```
- Mengukur seberapa baik atribut memisahkan data
- Atribut dengan gain tertinggi dipilih sebagai node

### 3. **Build Tree (ID3 Algorithm)**
- **Stopping Conditions:**
  1. Data sudah homogen
  2. Tidak ada atribut tersisa
  3. Information gain ≤ 0
- **Recursive Process:**
  1. Pilih atribut dengan gain tertinggi
  2. Split data berdasarkan nilai atribut
  3. Rekursif untuk setiap subset

### 4. **Predict**
- Traverse tree mengikuti nilai atribut sample
- Return prediksi saat mencapai leaf node

---

## 📈 Hasil dan Analisis

### Atribut yang Digunakan
| Atribut    | Deskripsi                           | Nilai              |
|------------|-------------------------------------|--------------------|
| Sex        | Jenis Kelamin                       | 0 = Laki, 1 = Wanita |
| Pclass     | Kelas Tiket                         | 1, 2, 3            |
| Age_Group  | Kelompok Umur                       | 0 = Anak (≤16), 1 = Dewasa (17-50), 2 = Lansia (>50) |

### Decision Tree Structure
```
Root: Sex (Information Gain tertinggi)
├── Sex = 0 (Laki-laki)
│   └── Pclass
│       ├── 1 → Age_Group → [Anak: Selamat, Dewasa/Lansia: Meninggal]
│       ├── 2 → Age_Group → [Anak: Selamat, Dewasa/Lansia: Meninggal]
│       └── 3 → Age_Group → [Semua: Meninggal]
│
└── Sex = 1 (Perempuan)
    └── Pclass
        ├── 1 → Age_Group → [Semua: Selamat]
        ├── 2 → Age_Group → [Semua: Selamat]
        └── 3 → Age_Group → [Anak/Lansia: Selamat, Dewasa: Meninggal]
```

### Insight yang Ditemukan
1. **Jenis kelamin adalah faktor terpenting** - Dipilih sebagai root node
2. **"Women and children first" policy** - Terlihat jelas dari pola tree
3. **Kelas sosial berpengaruh** - Kelas 1 & 2 lebih tinggi survival rate
4. **Usia penting untuk prioritas** - Anak-anak diprioritaskan

---


## 📚 Dokumentasi Tambahan

### File Dokumentasi
1. **PENJELASAN_ALGORITMA.md**
   - Penjelasan detail konsep Entropy dan Information Gain
   - Pseudocode dan contoh perhitungan step-by-step
   - Analisis hasil decision tree

2. **RINGKASAN_IMPLEMENTASI.txt**
   - Ringkasan implementasi setiap fungsi
   - Hasil eksekusi program
   - Insight dan kesimpulan

---

## 💡 Kelebihan Implementasi

✅ **Pure Python** - Tidak bergantung pada library ML  
✅ **Mudah dipahami** - Kode dengan komentar lengkap  
✅ **Visualisasi jelas** - Output yang informatif  
✅ **Akurasi baik** - 80.25% untuk model sederhana  
✅ **Tested** - Dilengkapi unit test komprehensif  
✅ **Dokumentasi lengkap** - README, penjelasan algoritma, ringkasan  

---

## 🔮 Kemungkinan Pengembangan

1. **Pruning** - Memangkas cabang untuk mencegah overfitting
2. **Handling Missing Values** - Strategi lebih baik untuk data kosong
3. **Feature Engineering** - Tambah fitur: SibSp, Parch, Fare, Embarked
4. **Cross-Validation** - K-fold validation untuk evaluasi lebih robust
5. **Ensemble Methods** - Random Forest atau Boosting
6. **Export Predictions** - Simpan hasil ke submission.csv
7. **GUI** - Interface grafis untuk input dan visualisasi

---

## 👥 Kelompok 15

**Tugas Besar Kecerdasan Buatan**  
**Tahun Akademik 2025/2026**

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dalam mata kuliah Kecerdasan Buatan.

---

## 📞 Kontak

Jika ada pertanyaan atau saran, silakan hubungi anggota kelompok 15.
1. Rahmadanis Danang Kumala - 103122400066
2. Alfan Didan Septiandri Argandi - 20102225

---

**Status:** ✅ **IMPLEMENTASI BERHASIL & BERFUNGSI DENGAN BAIK**

**Akurasi Model:** 80.25%  
**Test Coverage:** 100% PASS  
**Dokumentasi:** Lengkap  

---

*Last Updated: 08 Juni 2026*
