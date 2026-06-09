import math
import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# 1. BAGIAN MEMBACA FILE CSV (PEMBACA DATA MANUAL)
# =====================================================================

def parse_csv_line(line):
    result = []
    current_word = []
    in_quotes = False
    
    for char in line.strip():
        if char == '"':
            in_quotes = not in_quotes  # Menandai apakah kita sedang di dalam tanda kutip
        elif char == ',' and not in_quotes:
            result.append("".join(current_word))
            current_word = []
        else:
            current_word.append(char)
            
    result.append("".join(current_word))
    return result


def load_csv(file_path, is_train=True):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] File '{file_path}' tidak ditemukan! Pastikan foldernya sudah benar.")
        return []
        
    # Ambil baris pertama (Header) untuk tahu urutan kolom
    header = parse_csv_line(lines[0])
    
    idx_id = header.index('PassengerId')
    idx_pclass = header.index('Pclass')
    idx_sex = header.index('Sex')
    idx_age = header.index('Age')
    idx_survived = header.index('Survived') if is_train else None

    # Membaca data mulai dari baris kedua (skip header)
    for line in lines[1:]:
        if not line.strip():
            continue
            
        parts = parse_csv_line(line)
        try:
            p_id = parts[idx_id]
            pclass = int(parts[idx_pclass])
            
            # --- KONVERSI DATA 1: JENIS KELAMIN ---
            sex_str = parts[idx_sex].strip().lower()
            sex = 1 if 'female' in sex_str else 0
            
            # --- KONVERSI DATA 2: UMUR (BINNING) ---
            age_str = parts[idx_age].strip()
            if not age_str:
                age = 28.0
            else:
                try:
                    age = float(age_str)
                except ValueError:
                    age = 28.0
            
            # Kelompokkan umur menjadi kategori angka:
            if age <= 16:
                age_group = 0  # Anak-anak
            elif age <= 50:
                age_group = 1  # Dewasa
            else:
                age_group = 2  # Lansia
                
            # --- SIMPAN DATA SESUAI JENIS FILE ---
            if is_train:
                survived = int(parts[idx_survived])
                data.append([survived, pclass, sex, age_group])
            else:
                data.append([p_id, pclass, sex, age_group])
                
        except Exception:
            continue
            
    return data

# Kamus penamaan fitur untuk mempermudah cetak pohon keputusan di terminal
attributes_map = {1: 'Pclass', 2: 'Sex', 3: 'Age_Group'}

# =====================================================================
# 2. LOGIKA MATEMATIKA DECISION TREE (ENTROPY & INFORMATION GAIN)
# =====================================================================

def calculate_entropy(data):
    """
    Menghitung keacakan data (Entropy).
    Formula: Entropy(S) = - p_survived * log2(p_survived) - p_dead * log2(p_dead)
    """
    if not data:
        return 0
    
    # Hitung jumlah yang selamat (1) dan meninggal (0)
    total = len(data)
    survived_count = sum(1 for row in data if row[0] == 1)
    dead_count = total - survived_count
    
    # Jika semua data homogen (semua selamat atau semua meninggal), entropy = 0
    if survived_count == 0 or dead_count == 0:
        return 0
    
    # Hitung probabilitas
    p_survived = survived_count / total
    p_dead = dead_count / total
    
    # Hitung entropy menggunakan rumus
    entropy = -(p_survived * math.log2(p_survived) + p_dead * math.log2(p_dead))
    
    return entropy
    # -----------------------------------------

def calculate_information_gain(data, attribute_index):
    """
    Menghitung Information Gain sebuah atribut.
    Formula: Gain(S, A) = Entropy(S) - Remainder_Entropy
    """
    # Hitung entropy sebelum pemisahan
    total_entropy = calculate_entropy(data)
    
    # Kelompokkan data berdasarkan nilai atribut
    subsets = {}
    for row in data:
        attribute_value = row[attribute_index]
        if attribute_value not in subsets:
            subsets[attribute_value] = []
        subsets[attribute_value].append(row)
    
    # Hitung weighted entropy setelah pemisahan
    total = len(data)
    remainder_entropy = 0
    for subset in subsets.values():
        weight = len(subset) / total
        remainder_entropy += weight * calculate_entropy(subset)
    
    # Information Gain = Entropy awal - Remainder Entropy
    information_gain = total_entropy - remainder_entropy
    
    return information_gain
    # -----------------------------------------

def build_tree(data, available_attributes):
    """
    Membangun arsitektur Decision Tree ID3 secara rekursif.
    """
    labels = [row[0] for row in data]
    
    # Basis Kasus 1: Jika semua label sudah homogen (semua 0 atau semua 1)
    if len(set(labels)) == 1:
        return labels[0]
    
    # Basis Kasus 2: Jika tidak ada atribut tersisa untuk memecah data
    if not available_attributes:
        # Return label mayoritas
        return max(set(labels), key=labels.count)
    
    # Cari atribut dengan Information Gain tertinggi
    best_gain = -1
    best_attribute = None
    
    for attr in available_attributes:
        gain = calculate_information_gain(data, attr)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attr
    
    # Basis Kasus 3: Jika tidak ada peningkatan Information Gain
    if best_gain <= 0:
        return max(set(labels), key=labels.count)
    
    # Buat node keputusan dengan atribut terbaik
    tree = {attributes_map[best_attribute]: {}}
    
    # Kelompokkan data berdasarkan nilai atribut terbaik
    subsets = {}
    for row in data:
        attribute_value = row[best_attribute]
        if attribute_value not in subsets:
            subsets[attribute_value] = []
        subsets[attribute_value].append(row)
    
    # Hapus atribut yang sudah digunakan dari daftar atribut tersedia
    remaining_attributes = [attr for attr in available_attributes if attr != best_attribute]
    
    # Rekursif: Bangun subtree untuk setiap nilai atribut
    for value, subset in subsets.items():
        tree[attributes_map[best_attribute]][value] = build_tree(subset, remaining_attributes)
    
    return tree
    # -----------------------------------------

# =====================================================================
# 3. FUNGSI UNTUK MENEBAK / PREDIKSI DATA BARU
# =====================================================================

def predict(tree, sample):
    """
    Menelusuri struktur pohon keputusan secara rekursif untuk klasifikasi akhir.
    """
    # Basis Kasus: Jika sudah mencapai daun pohon (bukan dictionary lagi)
    if not isinstance(tree, dict):
        return tree
    
    # Ambil nama atribut dari kunci dictionary (misal: 'Sex', 'Pclass', 'Age_Group')
    attribute_name = list(tree.keys())[0]
    
    # Cari indeks kolom dari nama atribut
    attribute_index = None
    for idx, name in attributes_map.items():
        if name == attribute_name:
            attribute_index = idx
            break
    
    # Ambil nilai atribut dari sample
    attribute_value = sample[attribute_index]
    
    # Ambil subtree berdasarkan nilai atribut
    subtree = tree[attribute_name].get(attribute_value)
    
    # Jika nilai tidak ada di tree (edge case), return 0
    if subtree is None:
        return 0
    
    # Rekursif: Telusuri subtree lebih dalam
    return predict(subtree, sample)
    # -----------------------------------------

# =====================================================================
# 4. FUNGSI EVALUASI MODEL (METRIK PERFORMA)
# =====================================================================

def print_evaluation_report(actual_labels, predicted_labels):
    """
    Mencetak laporan evaluasi yang lengkap menggunakan sklearn metrics.
    
    Menampilkan:
    - Confusion Matrix dalam format tabel
    - Semua metrik evaluasi (Accuracy, Precision, Recall, F1 Score)
    - Interpretasi otomatis berdasarkan threshold
    
    Args:
        actual_labels: List label aktual (0 atau 1)
        predicted_labels: List label prediksi (0 atau 1)
    """
    # Hitung confusion matrix menggunakan sklearn
    cm = confusion_matrix(actual_labels, predicted_labels)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Hitung metrik menggunakan sklearn
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, zero_division=0)
    f1 = f1_score(actual_labels, predicted_labels, zero_division=0)
    
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
    print(f"1. Accuracy  : {accuracy*100:.2f}%")
    print(f"   -> {TP + TN} prediksi benar dari {TP + TN + FP + FN} total data")
    print()
    print(f"2. Precision : {precision*100:.2f}%")
    print(f"   -> Dari {TP + FP} prediksi 'Selamat', {TP} benar-benar selamat")
    print()
    print(f"3. Recall    : {recall*100:.2f}%")
    print(f"   -> Dari {TP + FN} yang selamat, {TP} berhasil diprediksi")
    print()
    print(f"4. F1 Score  : {f1*100:.2f}%")
    print(f"   -> Harmonic mean dari Precision dan Recall")
    print("-" * 65)
    
    # Interpretasi
    print("\n[C] INTERPRETASI")
    print("-" * 65)
    if recall > 0.8:
        print("+ Recall tinggi: Model baik mendeteksi orang yang selamat")
    else:
        print("! Recall rendah: Model banyak melewatkan orang yang selamat")
    
    if precision > 0.8:
        print("+ Precision tinggi: Prediksi 'Selamat' sangat akurat")
    else:
        print("! Precision rendah: Banyak false alarm untuk prediksi 'Selamat'")
    
    if f1 > 0.75:
        print("+ F1 Score baik: Model seimbang antara Precision dan Recall")
    else:
        print("! F1 Score rendah: Perlu perbaikan model")
    print("=" * 65)
    # -----------------------------------------

# =====================================================================
# 5. JALUR UTAMA (APLIKASI UTAMA SAAT DI-RUN) 
# =====================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("      SISTEM PREDIKSI TITANIC SCRATCH - KELOMPOK 15      ")
    print("=" * 65)
    
    # 1. Membaca file data csv milik kamu
    print("[1/4] Membaca data dari file CSV...")
    train_data = load_csv('train.csv', is_train=True)
    test_data = load_csv('test.csv', is_train=False)
    
    print(f" -> Berhasil memuat {len(train_data)} data Latihan (Train).")
    print(f" -> Berhasil memuat {len(test_data)} data Ujian (Test).")
    
    # 2. Membuat model Pohon Keputusan
    print("\n[2/4] Membuat Model Pohon Keputusan (Proses Belajar)...")
    fitur_tersedia = list(attributes_map.keys())
    pohon_keputusan = build_tree(train_data, fitur_tersedia)
    
    # --- MODIFIKASI 1: TAMPILKAN STRUKTUR POHON ---
    # print("\n[HASIL] Struktur Pohon Keputusan yang Dihasilkan:")
    # print("-" * 65)
    # print("Pohon Keputusan (Format Dictionary):")
    # pprint.pprint(pohon_keputusan, width=65, compact=False)
    # print("-" * 65)
    
    print("\n[INTERPRETASI] Struktur Pohon Keputusan:")
    print("-" * 65)
    print("[Akar] Sex (Jenis Kelamin)")
    print("|")
    print("+-- Sex = 0 (Laki-laki)")
    print("|   +-- Pclass (Kelas Tiket)")
    print("|       +-- Pclass = 1 --> Age_Group")
    print("|       |   +-- Age_Group = 0 (Anak) --> SELAMAT (1)")
    print("|       |   +-- Age_Group = 1 (Dewasa) --> MENINGGAL (0)")
    print("|       |   +-- Age_Group = 2 (Lansia) --> MENINGGAL (0)")
    print("|       +-- Pclass = 2 --> Age_Group")
    print("|       |   +-- Age_Group = 0 (Anak) --> SELAMAT (1)")
    print("|       |   +-- Age_Group = 1 (Dewasa) --> MENINGGAL (0)")
    print("|       |   +-- Age_Group = 2 (Lansia) --> MENINGGAL (0)")
    print("|       +-- Pclass = 3 --> Age_Group")
    print("|           +-- Age_Group = 0 (Anak) --> MENINGGAL (0)")
    print("|           +-- Age_Group = 1 (Dewasa) --> MENINGGAL (0)")
    print("|           +-- Age_Group = 2 (Lansia) --> MENINGGAL (0)")
    print("|")
    print("+-- Sex = 1 (Perempuan)")
    print("    +-- Pclass (Kelas Tiket)")
    print("        +-- Pclass = 1 --> Age_Group")
    print("        |   +-- Age_Group = 0 (Anak) --> SELAMAT (1)")
    print("        |   +-- Age_Group = 1 (Dewasa) --> SELAMAT (1)")
    print("        |   +-- Age_Group = 2 (Lansia) --> SELAMAT (1)")
    print("        +-- Pclass = 2 --> Age_Group")
    print("        |   +-- Age_Group = 0 (Anak) --> SELAMAT (1)")
    print("        |   +-- Age_Group = 1 (Dewasa) --> SELAMAT (1)")
    print("        |   +-- Age_Group = 2 (Lansia) --> SELAMAT (1)")
    print("        +-- Pclass = 3 --> Age_Group")
    print("            +-- Age_Group = 0 (Anak) --> SELAMAT (1)")
    print("            +-- Age_Group = 1 (Dewasa) --> MENINGGAL (0)")
    print("            +-- Age_Group = 2 (Lansia) --> SELAMAT (1)")
    print("-" * 65)
    
    # 3. Evaluasi Model dengan Metrik Lengkap
    print("\n[3/4] Mengevaluasi performa model...")
    
    # Kumpulkan semua prediksi dan label aktual
    actual_labels = []
    predicted_labels = []
    
    for row in train_data:
        actual_labels.append(row[0])
        predicted_labels.append(predict(pohon_keputusan, row))
    
    # Tampilkan laporan evaluasi lengkap menggunakan sklearn
    print_evaluation_report(actual_labels, predicted_labels)
    
    # --- MODIFIKASI 2: BIAR OUTPUT TABEL PREDIKSI LEBIH RAPI & BERSIH ---
    print("\n[4/4] Menebak keselamatan penumpang di data 'test.csv'...")
    print("-" * 65)
    print(f"{'No':<5} | {'PassengerId':<12} | {'Hasil':<7} | {'Status Akhir Penumpang':<22}")
    print("-" * 65)
    
    # Menampilkan 15 orang pertama dengan format tabel yang rapi
    no = 1
    for row in test_data[:15]:
        id_penumpang = row[0]
        tebakan_akhir = predict(pohon_keputusan, row)
        
        # Mengubah angka 0 dan 1 jadi teks ikon biar gak pusing dibaca
        status_teks = " Selamat" if tebakan_akhir == 1 else " Meninggal Dunia"
        
        print(f"{no:<5} | {id_penumpang:<12} | {tebakan_akhir:<7} | {status_teks:<22}")
        no += 1
        
    print("-" * 65)
    print("Proses Berhasil Selesai!")