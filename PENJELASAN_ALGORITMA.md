# Penjelasan Detail Algoritma Decision Tree - Kelompok 15

## 📚 Konsep Dasar

### 1. Entropy (Entropi)
Entropi mengukur tingkat **ketidakteraturan** atau **keacakan** dalam data.

**Formula:**
```
Entropy(S) = -p₁ * log₂(p₁) - p₂ * log₂(p₂)
```

Dimana:
- S = Dataset
- p₁ = Proporsi data dengan label 1 (Selamat)
- p₂ = Proporsi data dengan label 0 (Meninggal)

**Interpretasi:**
- Entropy = 0 → Data homogen (semua sama)
- Entropy = 1 → Data seimbang 50:50
- Entropy tinggi = Data acak, sulit diprediksi
- Entropy rendah = Data teratur, mudah diprediksi

**Contoh Perhitungan:**
```
Dataset: 100 penumpang
- 60 selamat (p₁ = 0.6)
- 40 meninggal (p₂ = 0.4)

Entropy = -(0.6 * log₂(0.6)) - (0.4 * log₂(0.4))
        = -(0.6 * -0.737) - (0.4 * -1.322)
        = 0.442 + 0.529
        = 0.971
```

---

### 2. Information Gain
Information Gain mengukur **seberapa baik** suatu atribut memisahkan data.

**Formula:**
```
Gain(S, A) = Entropy(S) - Σ (|Sᵥ|/|S|) * Entropy(Sᵥ)
```

Dimana:
- S = Dataset awal
- A = Atribut yang diuji
- Sᵥ = Subset data untuk nilai v dari atribut A
- |Sᵥ|/|S| = Bobot subset

**Interpretasi:**
- Gain tinggi = Atribut bagus untuk memisahkan data
- Gain rendah = Atribut kurang informatif
- Atribut dengan Gain tertinggi dipilih sebagai node

**Contoh Perhitungan:**
```
Dataset Awal: 100 penumpang (60 selamat, 40 meninggal)
Entropy(S) = 0.971

Uji Atribut "Sex":
- Laki-laki (50 orang): 15 selamat, 35 meninggal
  Entropy = -(0.3*log₂(0.3)) - (0.7*log₂(0.7)) = 0.881
  
- Perempuan (50 orang): 45 selamat, 5 meninggal
  Entropy = -(0.9*log₂(0.9)) - (0.1*log₂(0.1)) = 0.469

Remainder = (50/100)*0.881 + (50/100)*0.469 = 0.675

Gain(S, Sex) = 0.971 - 0.675 = 0.296
```

---

## 🌳 Algoritma ID3 (Iterative Dichotomiser 3)

### Langkah-langkah Pembangunan Tree:

1. **Basis Kasus (Berhenti)**
   - Semua data memiliki label yang sama → Return label
   - Tidak ada atribut tersisa → Return label mayoritas
   - Information Gain semua atribut ≤ 0 → Return mayoritas

2. **Rekursif (Lanjut)**
   - Hitung Information Gain untuk semua atribut
   - Pilih atribut dengan Gain tertinggi
   - Buat node dengan atribut terpilih
   - Pisahkan data berdasarkan nilai atribut
   - Rekursif untuk setiap subset

### Pseudocode:
```
function build_tree(data, attributes):
    if all labels are same:
        return that label
    
    if no attributes left:
        return majority label
    
    best_attr = attribute with highest information_gain
    
    if information_gain(best_attr) <= 0:
        return majority label
    
    tree = {best_attr: {}}
    
    for each value in best_attr:
        subset = data where best_attr == value
        tree[best_attr][value] = build_tree(subset, attributes - best_attr)
    
    return tree
```

---

## 🔍 Contoh Eksekusi Step-by-Step

### Dataset Contoh (Sederhana):
| Survived | Pclass | Sex | Age_Group |
|----------|--------|-----|-----------|
| 0        | 3      | 0   | 1         |
| 1        | 1      | 1   | 0         |
| 1        | 2      | 1   | 1         |
| 0        | 3      | 0   | 2         |

### Iterasi 1: Pilih Root Node
**Hitung Entropy Awal:**
```
Total: 4 data (2 selamat, 2 meninggal)
Entropy(S) = -(0.5*log₂(0.5)) - (0.5*log₂(0.5)) = 1.0
```

**Hitung Gain untuk setiap atribut:**

**A. Atribut "Sex":**
```
Sex=0 (Laki): 2 data (0 selamat, 2 meninggal) → Entropy = 0
Sex=1 (Wanita): 2 data (2 selamat, 0 meninggal) → Entropy = 0

Remainder = (2/4)*0 + (2/4)*0 = 0
Gain(Sex) = 1.0 - 0 = 1.0 ✓ TERTINGGI
```

**B. Atribut "Pclass":**
```
Pclass=1: 1 data → Entropy = 0
Pclass=2: 1 data → Entropy = 0
Pclass=3: 2 data (0 selamat, 2 meninggal) → Entropy = 0

Remainder = (1/4)*0 + (1/4)*0 + (2/4)*0 = 0
Gain(Pclass) = 1.0 - 0 = 1.0
```

**Pilihan:** Sex dipilih karena Gain tertinggi (1.0) dan lebih sederhana

### Iterasi 2: Cabang Sex=0
Data subset: 2 data, semua meninggal
→ **LEAF NODE: Return 0**

### Iterasi 3: Cabang Sex=1
Data subset: 2 data, semua selamat
→ **LEAF NODE: Return 1**

### Hasil Tree:
```
{
  'Sex': {
    0: 0,  # Laki-laki → Meninggal
    1: 1   # Perempuan → Selamat
  }
}
```

---

## 📊 Hasil pada Dataset Titanic Lengkap

### Struktur Tree yang Dihasilkan:
```
Sex (Root - Gain tertinggi)
├── Sex = 0 (Laki-laki)
│   └── Pclass
│       ├── 1 → Age_Group → [1, 0, 0]
│       ├── 2 → Age_Group → [1, 0, 0]
│       └── 3 → Age_Group → [0, 0, 0]
│
└── Sex = 1 (Perempuan)
    └── Pclass
        ├── 1 → Age_Group → [1, 1, 1]
        ├── 2 → Age_Group → [1, 1, 1]
        └── 3 → Age_Group → [1, 0, 1]
```

### Insight:
1. **Sex adalah faktor terpenting** (dipilih sebagai root)
2. **"Women and children first"** terlihat jelas dari tree
3. **Kelas sosial (Pclass)** mempengaruhi peluang selamat
4. **Usia** penting terutama untuk prioritas anak-anak

---

## 🎯 Cara Prediksi dengan Tree

### Algoritma Prediksi:
```python
def predict(tree, sample):
    # Jika sudah di leaf (bukan dict lagi), return nilai
    if not isinstance(tree, dict):
        return tree
    
    # Ambil nama atribut di node ini
    attribute = list(tree.keys())[0]
    
    # Ambil nilai atribut dari sample
    value = sample[attribute_index]
    
    # Ambil subtree
    subtree = tree[attribute][value]
    
    # Rekursif ke subtree
    return predict(subtree, sample)
```

### Contoh Prediksi:
**Sample:** Perempuan (Sex=1), Kelas 3 (Pclass=3), Dewasa (Age=1)

**Penelusuran Tree:**
```
1. Mulai di root: Sex
   → sample[Sex] = 1 (Perempuan)
   → Masuk cabang Sex=1

2. Node: Pclass
   → sample[Pclass] = 3
   → Masuk cabang Pclass=3

3. Node: Age_Group
   → sample[Age_Group] = 1 (Dewasa)
   → Masuk cabang Age_Group=1

4. Leaf node: 0
   → Prediksi: MENINGGAL
```

---

## 📈 Evaluasi Model

### Akurasi: 80.25%
```
Akurasi = (Prediksi Benar / Total Data) × 100%
        = (715 / 891) × 100%
        = 80.25%
```

### Confusion Matrix (Konseptual):
```
                Prediksi
              Mati  Hidup
Aktual Mati   [TP]  [FN]
      Hidup   [FP]  [TN]
```

### Kelebihan Model:
✓ Mudah dipahami dan divisualisasikan
✓ Tidak perlu normalisasi data
✓ Dapat menangani data kategorikal
✓ Menunjukkan fitur mana yang paling penting

### Kekurangan Model:
✗ Rentan overfitting pada data training
✗ Sensitif terhadap perubahan kecil data
✗ Tidak optimal untuk data kontinu
✗ Bias terhadap atribut dengan banyak nilai

---

## 🔧 Cara Kerja Kode Python

### 1. Load Data (load_csv)
```python
# Membaca CSV manual tanpa pandas
# Menangani koma dalam quotes
# Konversi data ke numeric
# Binning umur ke kategori
```

### 2. Build Tree (build_tree)
```python
# Rekursif dengan stopping conditions
# Hitung information gain semua atribut
# Pilih atribut terbaik
# Split data dan rekursif
```

### 3. Predict (predict)
```python
# Traverse tree mengikuti nilai atribut
# Rekursif sampai leaf node
# Return prediksi
```

---

## ✅ Kesimpulan

Decision Tree berhasil diimplementasi dengan:
- **Akurasi 80.25%** pada training data
- **Interpretability tinggi** - mudah dipahami
- **From scratch** - tanpa library ML
- **Pembelajaran konsep** entropy, information gain, dan rekursi

Model ini menunjukkan bahwa **jenis kelamin, kelas tiket, dan usia** adalah faktor utama dalam prediksi survival penumpang Titanic.

---

**Kelompok 15 - Tugas Besar Kecerdasan Buatan**
