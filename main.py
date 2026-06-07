import math
import pprint

# =====================================================================
# 1. BAGIAN MEMBACA FILE CSV (PEMBACA DATA MANUAL)
# =====================================================================

def parse_csv_line(line):
    """
    Fungsi ini gunanya untuk memotong baris CSV berdasarkan tanda koma.
    Hebatnya, fungsi ini bisa tahu kalau ada koma di dalam tanda kutip nama 
    (seperti "Braund, Mr. Owen"), koma tersebut TIDAK akan memotong data.
    """
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
    """
    Fungsi untuk membaca file 'train.csv' atau 'test.csv'.
    Di sini data mentah dibersihkan dan diubah jadi angka biasa agar komputer paham.
    """
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
# TUGAS KELOMPOK: SILAKAN LENGKAPI LOGIKA DI BAWAH INI (# TODO)
# =====================================================================

def calculate_entropy(data):
    """
    # TODO: Tugas Anggota Kelompok
    Menghitung keacakan data (Entropy).
    Rumus Matematika: Entropy(S) = - p_survived * log2(p_survived) - p_dead * log2(p_dead)
    Catatan: Target label 'Survived' (0 atau 1) berada di indeks ke-0 pada tiap baris data training.
    """
    if not data:
        return 0
        
    # --- MULAI COLOG / KODE KALIAN DI SINI ---
    pass # Hapus 'pass' ini jika kode kalian sudah ditulis
    # -----------------------------------------


def calculate_information_gain(data, attribute_index):
    """
    # TODO: Tugas Anggota Kelompok
    Menghitung Information Gain untuk mengetahui bobot terbaik suatu kolom/atribut.
    Rumus Matematika: Gain(S, A) = Entropy(Sebelum Dipisah) - Remainder_Entropy(Setelah Dipisah)
    Di mana Remainder_Entropy adalah total weighted entropy dari setiap subset percabangan data.
    """
    # --- MULAI COLOG / KODE KALIAN DI SINI ---
    return 0.0 # Ganti return default ini dengan hasil hitungan gain kalian
    # -----------------------------------------


def build_tree(data, available_attributes):
    """
    # TODO: Tugas Anggota Kelompok
    Membangun struktur Decision Tree (Nested Dictionary) secara rekursif.
    Petunjuk Basis Kasus (Stopping Conditions):
    1. Jika semua target label sudah homogen (selamat semua atau meninggal semua).
    2. Jika available_attributes sudah kosong (tidak ada fitur sisa untuk memecah data).
    3. Jika tidak ada peningkatan Information Gain yang berarti (Gain <= 0).
    """
    labels = [row[0] for row in data]
    
    # --- MULAI COLOG / KODE KALIAN DI SINI ---
    
    # Template return sementara agar program utama (Main) tidak error saat di-run kosong
    # Jika logika kalian selesai, hapus dictionary tiruan di bawah ini.
    mock_tree = {
        'Sex': {
            0: {'Pclass': {1: {'Age_Group': {0: 1, 1: 0, 2: 0}}, 2: {'Age_Group': {0: 1, 1: 0, 2: 0}}, 3: {'Age_Group': {0: 0, 1: 0, 2: 0}}}},
            1: {'Pclass': {1: {'Age_Group': {0: 1, 1: 1, 2: 1}}, 2: {'Age_Group': {0: 1, 1: 1, 2: 1}}, 3: {'Age_Group': {0: 1, 1: 0, 2: 1}}}}
        }
    }
    return mock_tree
    # -----------------------------------------


# =====================================================================
# 3. FUNGSI UNTUK MENEBAK / PREDIKSI DATA BARU
# TUGAS KELOMPOK: SILAKAN LENGKAPI LOGIKA DI BAWAH INI (# TODO)
# =====================================================================

def predict(tree, sample):
    """
    # TODO: Tugas Anggota Kelompok
    Menelusuri struktur pohon keputusan (nested dictionary) secara rekursif untuk menebak hasil akhir (0 atau 1).
    Petunjuk: 
    - Jika objek 'tree' bukan lagi berbentuk dictionary (isinstance(tree, dict) bernilai False), artinya perayapan sudah menyentuh ujung daun keputusan. Langsung kembalikan nilainya.
    - Gunakan attributes_map untuk mencocokkan kunci string string nama atribut dengan indeks kolom data sampel.
    """
    # --- MULAI COLOG / KODE KALIAN DI SINI ---
    return 0 # Ganti return default ini dengan logika penelusuran rekursif kalian
    # -----------------------------------------


# =====================================================================
# 4. JALUR UTAMA (APLIKASI UTAMA SAAT DI-RUN) 
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
    
    # --- MODIFIKASI 1: BIAR OUTPUT POHON TIDAK PUSING (BENTUK HIERARKI) ---
    print("\n[HASIL] Struktur Pohon Keputusan Kelompok 15:")
    print("-" * 65)
    print("[Akar Utama] Apakah Jenis Kelamin Penumpang?")
    print("│")
    print("├── Sex = 0 (Laki-laki)")
    print("│   └── Cek Kelas Tiket (Pclass):")
    print("│       ├── Kelas 1 ──> Umur: Anak-anak ( SELAMAT), Dewasa/Lansia ( MENINGGAL)")
    print("│       ├── Kelas 2 ──> Umur: Anak-anak ( SELAMAT), Dewasa/Lansia ( MENINGGAL)")
    print("│       └── Kelas 3 ──> Semua Umur ( MENINGGAL)")
    print("│")
    print("└── Sex = 1 (Perempuan)")
    print("    └── Cek Kelas Tiket (Pclass):")
    print("        ├── Kelas 1 ──> Semua Umur ( SELAMAT)")
    print("        ├── Kelas 2 ──> Semua Umur ( SELAMAT)")
    print("        └── Kelas 3 ──> Umur: Dewasa ( MENINGGAL), Anak-anak/Lansia ( SELAMAT)")
    print("-" * 65)
    
    # 3. Mengukur Akurasi Internal
    print("\n[3/4] Mengukur tingkat akurasi model...")
    total_benar = 0
    for row in train_data:
        tebakan = predict(pohon_keputusan, row)
        if tebakan == row[0]:
            total_benar += 1
            
    skor_akurasi = (total_benar / len(train_data)) * 100
    print(f" -> Tingkat Akurasi Model Kelompok Kita: {skor_akurasi:.2f}%")
    
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