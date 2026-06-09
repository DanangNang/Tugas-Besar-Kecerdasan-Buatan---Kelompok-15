# Plan: Menambahkan Metrik Evaluasi (Accuracy, Recall, F1 Score)

## Tujuan
Menambahkan fungsi untuk mengukur performa model Decision Tree dengan metrik evaluasi standar: **Accuracy**, **Recall**, **Precision**, dan **F1 Score**.

## Konteks Proyek
- **File utama**: `main.py` (331 baris)
- **Model**: Decision Tree ID3 dari scratch
- **Dataset**: Titanic survival prediction
- **Target**: Binary classification (0 = Meninggal, 1 = Selamat)
- **Akurasi saat ini**: 80.25% (hanya accuracy yang diukur)

## Metrik yang Akan Ditambahkan

### 1. Confusion Matrix
Matrix 2x2 yang menunjukkan:
- **True Positive (TP)**: Prediksi selamat, aktual selamat
- **True Negative (TN)**: Prediksi meninggal, aktual meninggal
- **False Positive (FP)**: Prediksi selamat, aktual meninggal
- **False Negative (FN)**: Prediksi meninggal, aktual selamat

### 2. Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Persentase prediksi yang benar dari total prediksi.

### 3. Precision
```
Precision = TP / (TP + FP)
```
Dari semua yang diprediksi selamat, berapa persen yang benar-benar selamat.

### 4. Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
Dari semua yang benar-benar selamat, berapa persen yang berhasil diprediksi selamat.

### 5. F1 Score
```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean dari Precision dan Recall (balance antara keduanya).

## Langkah Implementasi

### Langkah 1: Tambah Fungsi `calculate_confusion_matrix()`
**Lokasi**: Setelah fungsi `predict()` (sekitar baris 235)

**Fungsi**:
```python
def calculate_confusion_matrix(actual, predicted):
    """
    Menghitung confusion matrix dari hasil prediksi.
    
    Returns:
        tuple: (TP, TN, FP, FN)
    """
    TP = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    TN = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    FP = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    FN = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
    return TP, TN, FP, FN
```

**Penjelasan**:
- Input: list nilai aktual dan list nilai prediksi
- Output: tuple (TP, TN, FP, FN)
- Menggunakan zip untuk membandingkan setiap pasangan (actual, predicted)

### Langkah 2: Tambah Fungsi `calculate_metrics()`
**Lokasi**: Setelah `calculate_confusion_matrix()`

**Fungsi**:
```python
def calculate_metrics(TP, TN, FP, FN):
    """
    Menghitung metrik evaluasi dari confusion matrix.
    
    Returns:
        dict: {'accuracy', 'precision', 'recall', 'f1_score'}
    """
    total = TP + TN + FP + FN
    
    # Accuracy
    accuracy = (TP + TN) / total if total > 0 else 0
    
    # Precision (hindari division by zero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
```

**Penjelasan**:
- Input: nilai TP, TN, FP, FN dari confusion matrix
- Output: dictionary berisi semua metrik
- Menangani division by zero dengan conditional checks

### Langkah 3: Tambah Fungsi `print_evaluation_report()`
**Lokasi**: Setelah `calculate_metrics()`

**Fungsi**:
```python
def print_evaluation_report(TP, TN, FP, FN, metrics):
    """
    Mencetak laporan evaluasi yang lengkap dan mudah dibaca.
    """
    print("\n" + "=" * 65)
    print("           LAPORAN EVALUASI MODEL (TRAINING DATA)")
    print("=" * 65)
    
    # Confusion Matrix
    print("\n[A] CONFUSION MATRIX")
    print("-" * 65)
    print(f"                    Prediksi: Meninggal | Prediksi: Selamat")
    print(f"Aktual: Meninggal   TN = {TN:<16} | FP = {FP:<16}")
    print(f"Aktual: Selamat     FN = {FN:<16} | TP = {TP:<16}")
    print("-" * 65)
    
    # Metrik Detail
    print("\n[B] METRIK EVALUASI")
    print("-" * 65)
    print(f"1. Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"   └─> {TP + TN} prediksi benar dari {TP + TN + FP + FN} total data")
    print()
    print(f"2. Precision : {metrics['precision']*100:.2f}%")
    print(f"   └─> Dari {TP + FP} prediksi 'Selamat', {TP} benar-benar selamat")
    print()
    print(f"3. Recall    : {metrics['recall']*100:.2f}%")
    print(f"   └─> Dari {TP + FN} yang selamat, {TP} berhasil diprediksi")
    print()
    print(f"4. F1 Score  : {metrics['f1_score']*100:.2f}%")
    print(f"   └─> Harmonic mean dari Precision dan Recall")
    print("-" * 65)
    
    # Interpretasi
    print("\n[C] INTERPRETASI")
    print("-" * 65)
    if metrics['recall'] > 0.8:
        print("✓ Recall tinggi: Model baik mendeteksi orang yang selamat")
    else:
        print("⚠ Recall rendah: Model banyak melewatkan orang yang selamat")
    
    if metrics['precision'] > 0.8:
        print("✓ Precision tinggi: Prediksi 'Selamat' sangat akurat")
    else:
        print("⚠ Precision rendah: Banyak false alarm untuk prediksi 'Selamat'")
    
    if metrics['f1_score'] > 0.75:
        print("✓ F1 Score baik: Model seimbang antara Precision dan Recall")
    else:
        print("⚠ F1 Score rendah: Perlu perbaikan model")
    print("=" * 65)
```

**Penjelasan**:
- Menampilkan confusion matrix dalam format tabel
- Menampilkan semua metrik dengan penjelasan kontekstual
- Memberikan interpretasi otomatis berdasarkan threshold

### Langkah 4: Modifikasi Bagian `if __name__ == "__main__"`
**Lokasi**: Bagian [3/4] Mengukur Akurasi Internal (baris 301-310)

**Perubahan**:
```python
# 3. Evaluasi Model dengan Metrik Lengkap
print("\n[3/4] Mengevaluasi performa model...")

# Kumpulkan semua prediksi dan label aktual
actual_labels = []
predicted_labels = []

for row in train_data:
    actual_labels.append(row[0])
    predicted_labels.append(predict(pohon_keputusan, row))

# Hitung confusion matrix
TP, TN, FP, FN = calculate_confusion_matrix(actual_labels, predicted_labels)

# Hitung metrik evaluasi
metrics = calculate_metrics(TP, TN, FP, FN)

# Tampilkan laporan evaluasi lengkap
print_evaluation_report(TP, TN, FP, FN, metrics)
```

**Penjelasan**:
- Mengganti perhitungan akurasi sederhana dengan sistem evaluasi lengkap
- Mengumpulkan semua prediksi dan label aktual dalam list terpisah
- Memanggil fungsi-fungsi baru untuk analisis mendalam

### Langkah 5: Update Bagian Prediksi Test Data
**Lokasi**: Bagian [4/4] (opsional, untuk menambahkan confidence)

**Perubahan Minor** (opsional):
- Tetap tampilkan 15 prediksi pertama seperti sekarang
- Bisa tambahkan statistik agregat di akhir tabel

## Output yang Diharapkan

### Contoh Output Baru:
```
=================================================================
           LAPORAN EVALUASI MODEL (TRAINING DATA)
=================================================================

[A] CONFUSION MATRIX
-----------------------------------------------------------------
                    Prediksi: Meninggal | Prediksi: Selamat
Aktual: Meninggal   TN = 468            | FP = 81
Aktual: Selamat     FN = 95             | TP = 247
-----------------------------------------------------------------

[B] METRIK EVALUASI
-----------------------------------------------------------------
1. Accuracy  : 80.25%
   └─> 715 prediksi benar dari 891 total data

2. Precision : 75.30%
   └─> Dari 328 prediksi 'Selamat', 247 benar-benar selamat

3. Recall    : 72.22%
   └─> Dari 342 yang selamat, 247 berhasil diprediksi

4. F1 Score  : 73.73%
   └─> Harmonic mean dari Precision dan Recall
-----------------------------------------------------------------

[C] INTERPRETASI
-----------------------------------------------------------------
⚠ Recall rendah: Model banyak melewatkan orang yang selamat
⚠ Precision rendah: Banyak false alarm untuk prediksi 'Selamat'
⚠ F1 Score rendah: Perlu perbaikan model
=================================================================
```

## Testing dan Validasi

### Validasi Manual:
1. **Cek Total**: TP + TN + FP + FN harus sama dengan jumlah data training (891)
2. **Cek Accuracy**: (TP + TN) / Total harus sama dengan akurasi yang sudah ada (80.25%)
3. **Cek Range**: Semua metrik harus antara 0-1 (atau 0-100%)

### Edge Cases:
- Division by zero (sudah ditangani dengan conditional checks)
- Data kosong (tidak mungkin karena data training pasti ada)

## File yang Akan Dimodifikasi

### 1. `main.py`
- **Baris ditambah**: ~100 baris (3 fungsi baru + modifikasi main)
- **Total baris akhir**: ~431 baris
- **Backward compatible**: Output lama tetap ada, hanya ditambahkan detail

### 2. `README.md` (opsional update)
- Update bagian "Hasil dan Analisis" dengan metrik baru
- Tambahkan penjelasan tentang Precision, Recall, F1 Score

## Keuntungan Implementasi Ini

### Keuntungan Teknis:
1. ✅ **Evaluasi Komprehensif** - Tidak hanya accuracy, tapi semua metrik penting
2. ✅ **Insight Lebih Dalam** - Tahu apakah model bias ke kelas tertentu
3. ✅ **Standar Industri** - Menggunakan metrik yang umum di ML
4. ✅ **Pure Python** - Tetap tidak pakai library eksternal

### Keuntungan Akademis:
1. ✅ **Pemahaman Konsep** - Implementasi manual membantu pemahaman
2. ✅ **Dokumentasi Lengkap** - Setiap metrik dijelaskan dengan jelas
3. ✅ **Interpretasi Otomatis** - Membantu analisis hasil

### Keuntungan Praktis:
1. ✅ **Deteksi Masalah** - Bisa tahu kalau model bias atau overfitting
2. ✅ **Perbandingan Model** - Memudahkan evaluasi model alternatif
3. ✅ **Output Profesional** - Laporan evaluasi yang terstruktur

## Risiko dan Mitigasi

### Risiko 1: Performa
- **Risiko**: Loop tambahan untuk collect predictions bisa lambat
- **Mitigasi**: Data training hanya 891 rows, impact minimal
- **Status**: Tidak signifikan

### Risiko 2: Backward Compatibility
- **Risiko**: Output berubah, mungkin tidak sesuai ekspektasi
- **Mitigasi**: Struktur output tetap sama, hanya ditambahkan detail
- **Status**: Aman

### Risiko 3: Kompleksitas Kode
- **Risiko**: Kode jadi lebih panjang dan kompleks
- **Mitigasi**: Fungsi terpisah dengan dokumentasi jelas
- **Status**: Terkontrol

## Estimasi Waktu Implementasi

1. **Implementasi fungsi baru**: 15-20 menit
2. **Modifikasi main block**: 10 menit
3. **Testing dan validasi**: 10-15 menit
4. **Update dokumentasi**: 10 menit

**Total estimasi**: 45-55 menit

## Checklist Implementasi

- [ ] Tambah fungsi `calculate_confusion_matrix()`
- [ ] Tambah fungsi `calculate_metrics()`
- [ ] Tambah fungsi `print_evaluation_report()`
- [ ] Modifikasi bagian [3/4] di main block
- [ ] Test dengan menjalankan `python main.py`
- [ ] Validasi: cek TP+TN+FP+FN = 891
- [ ] Validasi: cek accuracy = 80.25%
- [ ] Validasi: cek semua metrik dalam range 0-100%
- [ ] (Opsional) Update README.md dengan metrik baru

## Referensi Formula

### Confusion Matrix:
```
                 Predicted
                 0      1
Actual  0       TN     FP
        1       FN     TP
```

### Metrik:
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

## Kesimpulan

Implementasi metrik evaluasi ini akan memberikan insight yang jauh lebih mendalam tentang performa model Decision Tree. Dengan confusion matrix, precision, recall, dan F1 score, kita bisa:

1. Mengetahui apakah model lebih baik memprediksi survival atau death
2. Mendeteksi bias pada kelas tertentu
3. Membandingkan trade-off antara false positive dan false negative
4. Menggunakan standar evaluasi yang sama dengan industri ML

Semua ini dilakukan dengan pure Python, tanpa library eksternal, sesuai dengan filosofi proyek "from scratch" yang sudah ada.
