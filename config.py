import os
import re
import json
import math
import glob

import cv2
import numpy as np
from skimage.morphology import skeletonize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PORT     = '/dev/tty.usbserial-A5069RR4'
BAUDRATE = 57600

OUTPUT_DIR            = os.path.join(BASE_DIR, 'raw_images')
FOLDER_MINUTIAE       = os.path.join(BASE_DIR, 'minutiae')
FOLDER_HASIL          = os.path.join(BASE_DIR, 'hasil_matching')
FOLDER_GABOR_FITUR    = os.path.join(BASE_DIR, 'gabor_features')
FOLDER_HASIL_GABOR    = os.path.join(BASE_DIR, 'hasil_gabor')
FOLDER_MATCHING_GABOR = os.path.join(BASE_DIR, 'hasil_matching_gabor')
FOLDER_EVALUASI       = os.path.join(BASE_DIR, 'evaluasi')

for _f in (OUTPUT_DIR, FOLDER_MINUTIAE, FOLDER_HASIL, FOLDER_GABOR_FITUR,
           FOLDER_HASIL_GABOR, FOLDER_MATCHING_GABOR, FOLDER_EVALUASI):
    os.makedirs(_f, exist_ok=True)


# ======================================================================
# 1. PARAMETER SENSOR DAN AKUISISI  (tidak diubah)
# ======================================================================
WIDTH  = 256
HEIGHT = 288

KODE_JARI_DATASET   = 'b'
JUMLAH_DATASET      = 8
JUMLAH_IDENTIFIKASI = 5
RIDGE_TERANG = False         # citra AS608: ridge GELAP, valley terang.


# ======================================================================
# 2. PARAMETER [P1] -- METODE MINUTIAE (Jain dkk. 1997)
# ======================================================================

P1_BLOK_W          = 16      
P1_SMOOTH_D        = 5      

P1_AMBANG_CERTAINTY = 0.30   #  T_s
P1_AMBANG_KONSISTENSI = 4.0  #  T_c
P1_ITERASI_KONSISTENSI = (2, 4)   
P1_MARGIN_TEPI      = 12     

P1_MASK_L          = 11      
P1_MASK_H          = 7       
P1_AMBANG_RIDGE    = 1.5     # T_ridge
P1_JUMLAH_ARAH_BIN = 16      

P1_LUAS_MIN_BERCAK = 25      

P1_SPUR_SUDUT_MIN  = 70.0    # T_lower (derajat); T_upper = 110
P1_SPUR_MAKS       = 20      # T_branch (piksel)
P1_JEMBATAN_MAKS   = 15      # T_break (piksel)

P1_JARAK_ANTAR_RIDGE = 9.0   # 
P1_RIDGE_SAMPEL    = 10      # e
P1_RIDGE_PANJANG   = 30      # 

P1_TOLERANSI_SUDUT = 30.0    # 
P1_KOTAK_R         = 8.0     # 
P1_KOTAK_E         = 7.5     # 
P1_BOBOT_R         = 1.0     # 
P1_BOBOT_E         = 2.0     # 
P1_BOBOT_THETA     = 0.1     # 
P1_PENALTI         = 200.0 * (P1_BOBOT_R + P1_BOBOT_E + P1_BOBOT_THETA)
P1_LAJU_BELAJAR    = 0.5     
P1_AMBANG_RIDGE_SELARAS = 0.8    
P1_KANDIDAT_DP     = 12
P1_KANDIDAT_SELARAS = 12     

P1_MINUTIAE_MIN    = 25      

THRESHOLD_COCOK = 0.8249


# ======================================================================
# 3. PARAMETER [P2] -- METODE TEKSTUR / FingerCode (Jain dkk. 2000)
# ======================================================================

P2_ITERASI_W       = (15, 10, 5)  
P2_SMOOTH          = 5      
P2_JARI_INTEGRASI  = 12      

P2_LEBAR_PITA      = 20      # b, lebar tiap pita (piksel)
P2_SEKTOR_PITA     = 16      # k, jumlah sektor tiap pita
P2_JARI_DALAM      = 20      # jari-jari lingkaran dalam yang dibuang
P2_JUMLAH_PITA     = 5       # B, jumlah pita

P2_M0              = 100.0    
P2_V0              = 100.0   

P2_GABOR_K         = 10.0    # K, jarak antar-ridge rata-rata (piksel)
P2_GABOR_DELTA     = 4.0     # delta_x = delta_y
P2_GABOR_MASK      = 33      # ukuran mask
P2_GABOR_MIN       = 0.05    # ambang nilai mask yang dipakai
P2_ORIENTASI       = 8       

P2_ROTASI_LANGKAH  = (-2, -1, 0, 1, 2)   
P2_ROTASI_HALUS    = 11.25               

SKOR_THRESHOLD_GABOR = 39.7441

P2_JUMLAH_SEKTOR = P2_JUMLAH_PITA * P2_SEKTOR_PITA
GABOR_DIMENSI    = P2_ORIENTASI * P2_JUMLAH_SEKTOR
P2_JARI_LUAR     = P2_JARI_DALAM + P2_JUMLAH_PITA * P2_LEBAR_PITA


# ======================================================================
# 4. NAMA LAMA YANG MASIH DIPAKAI MODUL LAIN
# ======================================================================
BORDER_MARGIN        = P1_MARGIN_TEPI
PAKAI_ALIGNMENT      = True

GABOR_ORIENTASI = P2_ORIENTASI
GABOR_FREKUENSI = 1.0 / P2_GABOR_K
GABOR_GRID      = P2_SEKTOR_PITA     # kolom peta ciri = sektor per pita

PANJANG_KODE_BANDING = 3
PANJANG_GARIS = 10


# ======================================================================
# ======================================================================
NAMA_JARI = {
    'a': 'jempol',
    'b': 'telunjuk',
    'c': 'jari tengah',
    'd': 'jari manis',
    'e': 'kelingking',
}

POLA_DATASET      = re.compile(r'^([a-e])_([a-z]{2,6})(\d{2})$')
POLA_DATASET_LAMA = re.compile(r'^([a-e])_(\d{2})_([a-z]{2,6})$')
POLA_IDENTIFIKASI = re.compile(r'^fingerprint_([a-z]{2,6})(\d{2})$')


def nama_dasar(path_atau_nama):
    return os.path.splitext(os.path.basename(path_atau_nama))[0]


def info_berkas(path_atau_nama):
    nama = nama_dasar(path_atau_nama)

    m = POLA_DATASET.match(nama)
    if m:
        kode_jari, kode_subjek, nomor = m.groups()
        return {'jenis': 'dataset', 'nama': nama, 'kode_jari': kode_jari,
                'jari': NAMA_JARI.get(kode_jari, '?'), 'nomor': nomor,
                'kode_subjek': kode_subjek, 'label': f'{kode_jari}_{kode_subjek}'}

    m = POLA_DATASET_LAMA.match(nama)
    if m:
        kode_jari, nomor, kode_subjek = m.groups()
        return {'jenis': 'dataset', 'nama': nama, 'kode_jari': kode_jari,
                'jari': NAMA_JARI.get(kode_jari, '?'), 'nomor': nomor,
                'kode_subjek': kode_subjek, 'label': f'{kode_jari}_{kode_subjek}'}

    m = POLA_IDENTIFIKASI.match(nama)
    if m:
        kode_subjek, nomor = m.groups()
        return {'jenis': 'identifikasi', 'nama': nama, 'kode_jari': None,
                'jari': '(citra uji)', 'nomor': nomor,
                'kode_subjek': kode_subjek, 'label': None}

    return None


def is_dataset(path_atau_nama):
    info = info_berkas(path_atau_nama)
    return info is not None and info['jenis'] == 'dataset'


def is_identifikasi(path_atau_nama):
    info = info_berkas(path_atau_nama)
    return info is not None and info['jenis'] == 'identifikasi'


def kode_banding(kode_subjek):
    return (kode_subjek or '')[:PANJANG_KODE_BANDING].lower()


def subjek_dari(path_atau_nama):
    info = info_berkas(path_atau_nama)
    return info['kode_subjek'] if info else None


def pasangan_genuine(nama_uji, nama_referensi):
    a, b = subjek_dari(nama_uji), subjek_dari(nama_referensi)
    if a is None or b is None:
        return None
    return kode_banding(a) == kode_banding(b)


def kategori_hasil(genuine, cocok):
    if genuine is None:
        return 'tidak diketahui', '-'
    if genuine and cocok:
        return 'benar cocok', 'TM'
    if genuine and not cocok:
        return 'salah tidak cocok', 'FNM'
    if (not genuine) and cocok:
        return 'salah cocok', 'FM'
    return 'benar tidak cocok', 'TN'


def daftar_dataset(folder=None, ekstensi='*.bmp', kode_subjek=None):
    folder = folder or OUTPUT_DIR
    hasil = [f for f in glob.glob(os.path.join(folder, ekstensi)) if is_dataset(f)]
    if kode_subjek:
        kb = kode_banding(kode_subjek)
        hasil = [f for f in hasil if kode_banding(subjek_dari(f)) == kb]
    return sorted(hasil)


def daftar_identifikasi(folder=None, ekstensi='*.bmp', kode_subjek=None):
    folder = folder or OUTPUT_DIR
    hasil = [f for f in glob.glob(os.path.join(folder, ekstensi)) if is_identifikasi(f)]
    if kode_subjek:
        kb = kode_banding(kode_subjek)
        hasil = [f for f in hasil if kode_banding(subjek_dari(f)) == kb]
    return sorted(hasil)


def nomor_berikutnya_dataset(kode_subjek, kode_jari=None, folder=None):
    kode_jari = kode_jari or KODE_JARI_DATASET
    tertinggi = 0
    for f in daftar_dataset(folder):
        info = info_berkas(f)
        if info['kode_jari'] == kode_jari and info['kode_subjek'] == kode_subjek:
            tertinggi = max(tertinggi, int(info['nomor']))
    return tertinggi + 1


def nomor_berikutnya_identifikasi(kode_subjek, folder=None):
    tertinggi = 0
    for f in daftar_identifikasi(folder):
        info = info_berkas(f)
        if info['kode_subjek'] == kode_subjek:
            tertinggi = max(tertinggi, int(info['nomor']))
    return tertinggi + 1


def nama_dataset(kode_subjek, nomor, kode_jari=None):
    kode_jari = kode_jari or KODE_JARI_DATASET
    return f'{kode_jari}_{kode_subjek}{nomor:02d}'


def nama_identifikasi(kode_subjek, nomor):
    return f'fingerprint_{kode_subjek}{nomor:02d}'


# ======================================================================
# ======================================================================
def tanya_pilihan(pertanyaan, pilihan_sah):
    tampil = "/".join(pilihan_sah)
    while True:
        jawab = input(f"{pertanyaan} [{tampil}]: ").strip().lower()
        if jawab in pilihan_sah:
            return jawab
        print(f"  Masukan tidak dikenali. Pilih salah satu: {tampil}")


def tanya_ya_tidak(pertanyaan):
    while True:
        jawab = input(f"{pertanyaan} [y/n]: ").strip().lower()
        if jawab in ('y', 'ya', 'yes'):
            return True
        if jawab in ('n', 'no', 'tidak'):
            return False
        print("  Ketik y atau n.")


def tanya_kode_subjek(pesan="Kode subjek (inisial nama, contoh cbs)"):
    while True:
        kode = input(f"{pesan}: ").strip().lower()
        if re.fullmatch(r'[a-z]{2,6}', kode):
            print(f"  -> Kode subjek: {kode}")
            return kode
        print("  Kode hanya huruf, panjang 2-6 karakter. Contoh: cbs")


def tanya_angka(pertanyaan, minimum=1, maksimum=99):
    while True:
        teks = input(f"{pertanyaan} [{minimum}-{maksimum}]: ").strip()
        if teks.isdigit() and minimum <= int(teks) <= maksimum:
            return int(teks)
        print(f"  Harus angka {minimum} sampai {maksimum}.")


# ======================================================================
# 7. [P1] BAB II-A -- MEDAN ORIENTASI DAN SEGMENTASI
# ======================================================================
def medan_orientasi(img_gray, W=None, D=None, haluskan=True):
    """Estimasi medan orientasi cara kuadrat terkecil."""
    W = W or P1_BLOK_W
    D = D or P1_SMOOTH_D

    g = img_gray.astype(np.float32)
    Gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    h, w = g.shape
    nb_y, nb_x = h // W, w // W

    Gxx, Gyy, Gxy = Gx * Gx, Gy * Gy, Gx * Gy

    Vx = np.zeros((nb_y, nb_x), np.float32)   # jumlah 2 Gx Gy
    Vy = np.zeros((nb_y, nb_x), np.float32)   # jumlah Gx^2 - Gy^2
    Vt = np.zeros((nb_y, nb_x), np.float32)   # jumlah Gx^2 + Gy^2

    for i in range(nb_y):
        for j in range(nb_x):
            s = (slice(i * W, (i + 1) * W), slice(j * W, (j + 1) * W))
            sxx = float(Gxx[s].sum())
            syy = float(Gyy[s].sum())
            sxy = float(Gxy[s].sum())
            Vx[i, j] = 2.0 * sxy
            Vy[i, j] = sxx - syy
            Vt[i, j] = sxx + syy

    theta = 0.5 * np.arctan2(Vx, Vy) + math.pi / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        certainty = np.sqrt(Vx ** 2 + Vy ** 2) / np.maximum(Vt, 1e-9)
    certainty = np.nan_to_num(np.clip(certainty, 0.0, 1.0))

    if haluskan:
        phi_x = cv2.blur(np.cos(2.0 * theta), (D, D))
        phi_y = cv2.blur(np.sin(2.0 * theta), (D, D))
        theta = 0.5 * np.arctan2(phi_y, phi_x)

    return theta, certainty, Vt


def _konsistensi(orientasi_blok, D=None):
    """Consistency level C(i,j), [P1] Gbr. 8 langkah 4."""
    D = D or P1_SMOOTH_D
    deg = np.mod(np.degrees(orientasi_blok), 180.0).astype(np.float32)
    h, w = deg.shape
    r = D // 2
    pad = np.pad(deg, r, mode='edge')
    akum = np.zeros_like(deg)
    N = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            t = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            d = np.abs((t - deg + 90.0) % 180.0 - 90.0)   # 0..90 derajat
            akum += d * d
            N += 1
    return np.sqrt(akum) / N


def medan_orientasi_p1(img_gray, W=None, D=None):
    """Medan orientasi HIERARKIS [P1] Gbr. 8, lengkap langkah 1-5."""
    W = W or P1_BLOK_W
    D = D or P1_SMOOTH_D
    o, cert, energi = medan_orientasi(img_gray, W=W, D=D, haluskan=False)

    for pengali in P1_ITERASI_KONSISTENSI:
        buruk = _konsistensi(o, D) > P1_AMBANG_KONSISTENSI
        if not buruk.any():
            break
        o_kasar, _, _ = medan_orientasi(img_gray, W=W * pengali, D=D,
                                        haluskan=False)
        o_naik = np.repeat(np.repeat(o_kasar, pengali, axis=0), pengali, axis=1)
        hy = min(o.shape[0], o_naik.shape[0])
        hx = min(o.shape[1], o_naik.shape[1])
        bagian = buruk[:hy, :hx]
        o[:hy, :hx] = np.where(bagian, o_naik[:hy, :hx], o[:hy, :hx])

    return o, cert, energi


def segmentasi_certainty(certainty, energi, bentuk_citra, W=None,
                         ambang=None):
    """Segmentasi daerah sidik jari, [P1]."""
    W = W or P1_BLOK_W
    ambang = P1_AMBANG_CERTAINTY if ambang is None else ambang

    energi_norm = energi / max(float(energi.max()), 1e-9)
    mask_blok = ((certainty >= ambang) & (energi_norm >= 0.02)).astype(np.uint8)

    mask_blok = cv2.morphologyEx(mask_blok, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask_blok = cv2.morphologyEx(mask_blok, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_blok, 8)
    if n > 1:
        terbesar = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask_blok = (lab == terbesar).astype(np.uint8)

    h, w = bentuk_citra
    mask = cv2.resize(mask_blok * 255, (mask_blok.shape[1] * W,
                                        mask_blok.shape[0] * W),
                      interpolation=cv2.INTER_NEAREST)
    penuh = np.zeros((h, w), np.uint8)
    penuh[:mask.shape[0], :mask.shape[1]] = mask

    m = P1_MARGIN_TEPI
    penuh[:m, :] = 0
    penuh[-m:, :] = 0
    penuh[:, :m] = 0
    penuh[:, -m:] = 0
    return penuh


def _orientasi_per_piksel(orientasi_blok, bentuk_citra, W=None):
    """Membesarkan medan orientasi aras blok menjadi aras piksel."""
    W = W or P1_BLOK_W
    h, w = bentuk_citra
    cos2 = cv2.resize(np.cos(2 * orientasi_blok),
                      (orientasi_blok.shape[1] * W, orientasi_blok.shape[0] * W),
                      interpolation=cv2.INTER_LINEAR)
    sin2 = cv2.resize(np.sin(2 * orientasi_blok),
                      (orientasi_blok.shape[1] * W, orientasi_blok.shape[0] * W),
                      interpolation=cv2.INTER_LINEAR)
    o = 0.5 * np.arctan2(sin2, cos2)
    penuh = np.zeros((h, w), np.float32)
    penuh[:o.shape[0], :o.shape[1]] = o
    if o.shape[0] < h:
        penuh[o.shape[0]:, :o.shape[1]] = o[-1:, :]
    if o.shape[1] < w:
        penuh[:, o.shape[1]:] = penuh[:, o.shape[1] - 1:o.shape[1]]
    return penuh


# ======================================================================
# 8. [P1] BAB II-B -- DETEKSI RIDGE DENGAN DUA MASK h_t DAN h_b
# ======================================================================
def _mask_ridge(sudut, L=None, H=None):
    """
    Membuat sepasang mask h_t dan h_b untuk satu arah ridge, [P1] Pers. 4-6.
    """
    L = L or P1_MASK_L
    H = H or P1_MASK_H

    sisi = 2 * (max(L, H) // 2) + 1 + 2 * (H // 2)
    r = sisi // 2
    ht = np.zeros((sisi, sisi), np.float32)
    hb = np.zeros((sisi, sisi), np.float32)

    ux, uy = math.cos(sudut), math.sin(sudut)          # sepanjang ridge
    nx, ny = -math.sin(sudut), math.cos(sudut)         # tegak lurus ridge
    geser = H / 2.0
    setengah_L = L // 2

    def tebar(mask, t, bobot):
        """Menaruh bobot pada garis sepanjang ridge yang digeser sejauh t."""
        for k in range(-setengah_L, setengah_L + 1):
            x = k * ux + t * nx
            y = k * uy + t * ny
            xi, yi = int(round(x)) + r, int(round(y)) + r
            if 0 <= xi < sisi and 0 <= yi < sisi:
                mask[yi, xi] += bobot

    n_titik = 2 * setengah_L + 1
    tebar(ht, 0.0, 1.0 / n_titik)
    tebar(ht, +geser, -1.0 / n_titik)
    tebar(hb, 0.0, 1.0 / n_titik)
    tebar(hb, -geser, -1.0 / n_titik)
    return ht, hb


def deteksi_ridge(img_gray, orientasi_blok, mask_roi, ambang=None):
    """Peta ridge biner, [P1] Bab II-B."""
    ambang = P1_AMBANG_RIDGE if ambang is None else ambang

    g = img_gray.astype(np.float32)
    if not RIDGE_TERANG:
        g = 255.0 - g          # [P1] menganggap ridge = maksimum lokal

    h, w = g.shape
    o_piksel = _orientasi_per_piksel(orientasi_blok, (h, w))

    nb = P1_JUMLAH_ARAH_BIN
    idx = np.round(((o_piksel + math.pi / 2) % math.pi) / (math.pi / nb)).astype(int) % nb

    ridge = np.zeros((h, w), np.uint8)
    for b in range(nb):
        sudut = b * (math.pi / nb) - math.pi / 2
        ht, hb = _mask_ridge(sudut)
        rt = cv2.filter2D(g, cv2.CV_32F, ht)
        rb = cv2.filter2D(g, cv2.CV_32F, hb)
        pilih = (idx == b)
        ridge[pilih & (rt > ambang) & (rb > ambang)] = 255

    ridge[mask_roi == 0] = 0
    return ridge


def bersihkan_lubang_bercak(ridge, luas_min=None):
    """Membuang lubang dan bercak, [P1] Bab II-B."""
    luas_min = P1_LUAS_MIN_BERCAK if luas_min is None else luas_min

    biner = (ridge > 0).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(biner, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < luas_min:
            biner[lab == i] = 0

    lubang = 1 - biner
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lubang.astype(np.uint8), 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < luas_min:
            biner[lab == i] = 1

    return (biner * 255).astype(np.uint8)


# ======================================================================
# 9. [P1] BAB II-C -- PENIPISAN, PENGHALUSAN RANGKA, DETEKSI MINUTIAE
# ======================================================================
_TETANGGA_8 = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]


def _tetangga_hidup(skel_int, x, y):
    h, w = skel_int.shape
    out = []
    for dx, dy in _TETANGGA_8:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and skel_int[ny, nx] == 1:
            out.append((nx, ny))
    return out


def _peta_derajat(skel_int):
    """Banyak tetangga hidup tiap piksel rangka (nilai 0..8)."""
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(skel_int.astype(np.uint8), cv2.CV_8U, k) * skel_int


def haluskan_rangka(skel, spur_maks=None, jembatan_maks=None):
    """Penghalusan rangka sebelum deteksi minutiae, [P1] Bab II-C hlm. 1375."""
    spur_maks = P1_SPUR_MAKS if spur_maks is None else spur_maks
    jembatan_maks = P1_JEMBATAN_MAKS if jembatan_maks is None else jembatan_maks

    s = (skel > 0).astype(np.uint8)

    def arah_dari(titik, larangan, langkah=8):
        """Vektor arah rata-rata beberapa piksel pertama dari sebuah titik."""
        cx, cy = titik
        sebelumnya = larangan
        akhir = titik
        for _ in range(langkah):
            tet = [p for p in _tetangga_hidup(s, cx, cy) if p != sebelumnya]
            if len(tet) != 1:
                break
            sebelumnya = (cx, cy)
            cx, cy = tet[0]
            akhir = (cx, cy)
        return (akhir[0] - titik[0], akhir[1] - titik[1])

    for _ in range(3):
        deg = _peta_derajat(s)
        ujung = list(zip(*np.where((deg == 1) & (s == 1))))
        dibuang = 0
        for (y, x) in ujung:
            if s[y, x] != 1:
                continue
            jalur = [(x, y)]
            cx, cy = x, y
            sebelumnya = None
            sampai_cabang = False
            for _ in range(spur_maks):
                tet = [p for p in _tetangga_hidup(s, cx, cy) if p != sebelumnya]
                if len(tet) != 1:
                    break
                sebelumnya = (cx, cy)
                cx, cy = tet[0]
                if len(_tetangga_hidup(s, cx, cy)) > 2:
                    sampai_cabang = True
                    break
                jalur.append((cx, cy))
            if not sampai_cabang or len(jalur) > spur_maks:
                continue

            lain = [p for p in _tetangga_hidup(s, cx, cy) if p != sebelumnya]
            if len(lain) < 2:
                continue
            d1 = arah_dari(lain[0], (cx, cy))
            d2 = arah_dari(lain[1], (cx, cy))
            ax = (lain[0][0] - cx + d1[0]) - (lain[1][0] - cx + d2[0])
            ay = (lain[0][1] - cy + d1[1]) - (lain[1][1] - cy + d2[1])
            na = math.hypot(ax, ay)
            bx, by = x - cx, y - cy
            nb = math.hypot(bx, by)
            if na < 1e-6 or nb < 1e-6:
                continue
            kos = abs((ax * bx + ay * by) / (na * nb))
            sudut = math.degrees(math.acos(min(1.0, kos)))     # 0..90
            if sudut <= P1_SPUR_SUDUT_MIN:
                continue          # setara 70 < sudut < 110 pada garis trunk
            for (px, py) in jalur:
                s[py, px] = 0
            dibuang += 1
        if dibuang == 0:
            break

    deg = _peta_derajat(s)
    ujung = [(int(x), int(y)) for (y, x) in zip(*np.where((deg == 1) & (s == 1)))]
    arah = {}
    for (x, y) in ujung:
        d = arah_dari((x, y), None)
        arah[(x, y)] = (-d[0], -d[1])          # menghadap keluar dari ridge

    def ada_ridge_lain(a, b):
        """Benar bila ada piksel rangka lain di ruas garis a-b."""
        n = int(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
        for t in range(1, n):
            px = int(round(a[0] + (b[0] - a[0]) * t / n))
            py = int(round(a[1] + (b[1] - a[1]) * t / n))
            if s[py, px] == 1:
                return True
        return False

    dipakai = set()
    for i, a in enumerate(ujung):
        if a in dipakai:
            continue
        for b in ujung[i + 1:]:
            if b in dipakai:
                continue
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            if d < 2 or d > jembatan_maks:
                continue
            if ada_ridge_lain(a, b):
                continue                        # "no other ridges pass through"
            va, vb = arah[a], arah[b]
            na = math.hypot(*va) or 1.0
            nb = math.hypot(*vb) or 1.0
            if (va[0] * vb[0] + va[1] * vb[1]) / (na * nb) > -0.3:
                continue
            cv2.line(s, a, b, 1, 1)
            dipakai.add(a)
            dipakai.add(b)
            break

    s = skeletonize(s.astype(bool)).astype(np.uint8)
    return s * 255


def thinning(binary):
    """Penipisan rangka lalu penghalusan, [P1] Bab II-B dan II-C."""
    skel = skeletonize((binary // 255).astype(bool)).astype(np.uint8) * 255
    skel = haluskan_rangka(skel)
    print(f"    Thinning selesai (piksel putih: {cv2.countNonZero(skel)})")
    return skel


def ekstraksi_minutiae(skeleton):
    """Deteksi minutiae, [P1] Bab II-C."""
    skel_int = (skeleton // 255).astype(np.uint8)
    deg = _peta_derajat(skel_int)

    ridge_endings, bifurcations = [], []
    h, w = skel_int.shape
    ys, xs = np.where(skel_int == 1)
    for y, x in zip(ys, xs):
        if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
            continue
        d = int(deg[y, x])
        if d == 1:
            ridge_endings.append((int(x), int(y)))
        elif d > 2:
            bifurcations.append((int(x), int(y)))

    print(f"    Sebelum penapisan: {len(ridge_endings)} ending, "
          f"{len(bifurcations)} bifurcation")
    return ridge_endings, bifurcations, skel_int.astype(int)


def hitung_inter_ridge_width(skel_int):
    """Jarak antar-ridge rata-rata, [P1] Bab II-C."""
    h, w = skel_int.shape
    jumlah_ridge = jumlah_baris = 0
    for i in range(10, h - 10, 10):
        count = int(np.sum(skel_int[i, :]))
        if count > 0:
            jumlah_ridge += w / count
            jumlah_baris += 1
    ukur = jumlah_ridge / jumlah_baris if jumlah_baris > 0 else 0.0
    D = float(P1_JARAK_ANTAR_RIDGE)
    print(f"    Inter-ridge width: {D:.1f} piksel (tetapan alat; "
          f"ukuran citra ini {ukur:.1f})")
    return D


def filter_minutiae(endings, bifurcations, skel_int):
    """Penapisan minutiae."""
    h, w = skel_int.shape
    D = hitung_inter_ridge_width(skel_int)
    bm = BORDER_MARGIN

    mask = _KONTEKS.get('mask')
    if mask is not None and mask.shape == skel_int.shape:
        k = int(max(3, round(D))) | 1
        sah = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    else:
        sah = np.full((h, w), 255, np.uint8)

    def di_dalam(x, y):
        return (bm < x < w - bm and bm < y < h - bm and sah[y, x] > 0)

    ending_final = [(x, y) for x, y in endings if di_dalam(x, y)]
    bifur_final = [(x, y) for x, y in bifurcations if di_dalam(x, y)]

    print(f"    Setelah penapisan: {len(ending_final)} ending, "
          f"{len(bifur_final)} bifurcation "
          f"(total {len(ending_final) + len(bifur_final)})")
    return ending_final, bifur_final


# ----------------------------------------------------------------------
# 9b. Ridge pendamping dan arah minutia ([P1] Bab II-C)
# ----------------------------------------------------------------------
def telusuri_ridge(skel_int, x, y, panjang=None):
    """Menelusuri ridge yang menempel pada sebuah minutia."""
    panjang = panjang or P1_RIDGE_PANJANG
    s = np.asarray(skel_int, dtype=np.uint8)
    awal = _tetangga_hidup(s, x, y)
    if not awal:
        return [(x, y)]

    jalur_terbaik = []
    for mulai in awal:
        jalur = [(x, y), mulai]
        dikunjungi = {(x, y), mulai}
        cx, cy = mulai
        for _ in range(panjang - 1):
            lanjut = [p for p in _tetangga_hidup(s, cx, cy) if p not in dikunjungi]
            if not lanjut:
                break
            if len(lanjut) > 1:
                dx0, dy0 = cx - jalur[-2][0], cy - jalur[-2][1]
                lanjut.sort(key=lambda p: -((p[0] - cx) * dx0 + (p[1] - cy) * dy0))
            cx, cy = lanjut[0]
            dikunjungi.add((cx, cy))
            jalur.append((cx, cy))
        if len(jalur) > len(jalur_terbaik):
            jalur_terbaik = jalur
    return jalur_terbaik


def hitung_arah(skel_int, x, y, panjang=None):
    """
    Arah minutia dalam radian, memakai konvensi matematika
    (0 = ke kanan, positif berlawanan arah jarum jam).
    """
    jalur = telusuri_ridge(skel_int, x, y, panjang)
    if len(jalur) < 2:
        return 0.0
    ex, ey = jalur[-1]
    dx, dy = ex - x, -(ey - y)      # sumbu y citra ke bawah, jadi dibalik
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def bentuk_ridge(skel_int, x, y, theta, D, panjang=None, n_sampel=None):
    """
    Ridge pendamping sebagai isyarat satu dimensi, [P1] Bab II-C dan Gbr. 12.
    """
    n_sampel = n_sampel or P1_RIDGE_SAMPEL
    panjang = panjang or int(round(n_sampel * D))
    jalur = telusuri_ridge(skel_int, x, y, panjang)
    if len(jalur) < 2:
        return np.zeros((2, n_sampel), np.float32)

    c, s = math.cos(-theta), math.sin(-theta)
    lokal = []
    for (px, py) in jalur:
        dx, dy = px - x, -(py - y)
        lokal.append(((dx * c - dy * s) / D,      # searah minutia
                      (dx * s + dy * c) / D))     # melintang minutia

    lokal = np.asarray(lokal, np.float32)
    t = np.linspace(0.0, 1.0, len(lokal))
    tq = np.linspace(0.0, 1.0, n_sampel)
    return np.stack([np.interp(tq, t, lokal[:, 0]),
                     np.interp(tq, t, lokal[:, 1])]).astype(np.float32)


# ======================================================================
# 10. PIPELINE SATU CITRA
# ======================================================================
_KONTEKS = {}     # menyimpan medan orientasi terakhir untuk keperluan gambar


def siapkan_medan_dan_segmentasi(img_gray):
    """
    Langkah pertama pipeline minutiae: medan orientasi dan segmentasi ROI.
    [P1] Bab II-A, Pers. 2-3, dan Gbr. 8.
    """
    o, cert, energi = medan_orientasi_p1(img_gray)
    mask = segmentasi_certainty(cert, energi, img_gray.shape)
    _KONTEKS['orientasi'] = o
    _KONTEKS['certainty'] = cert
    _KONTEKS['energi'] = energi
    _KONTEKS['mask'] = mask
    luas = int(cv2.countNonZero(mask))
    persen = 100.0 * luas / (img_gray.shape[0] * img_gray.shape[1])
    print(f"    Segmentasi certainty level: {luas} piksel "
          f"({persen:.1f}% luas citra), blok {P1_BLOK_W}x{P1_BLOK_W}")
    return img_gray


def preprocessing(img_gray):
    """Deteksi ridge, [P1] Bab II-A dan II-B."""
    if _KONTEKS.get('mask') is None or _KONTEKS['mask'].shape != img_gray.shape:
        siapkan_medan_dan_segmentasi(img_gray)
    o = _KONTEKS['orientasi']
    mask = _KONTEKS['mask']

    ridge = deteksi_ridge(img_gray, o, mask)
    ridge = bersihkan_lubang_bercak(ridge)
    print(f"    Deteksi ridge selesai (mask {P1_MASK_L}x{P1_MASK_H}, "
          f"Tridge={P1_AMBANG_RIDGE}).")
    return ridge


def ekstraksi_dari_file(path):
    """Pipeline minutiae lengkap dari sebuah berkas BMP."""
    img_full = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        return None
    img = siapkan_medan_dan_segmentasi(img_full)
    binary = preprocessing(img)
    skeleton = thinning(binary)
    endings_raw, bifurs_raw, skel_int = ekstraksi_minutiae(skeleton)
    endings, bifurs = filter_minutiae(endings_raw, bifurs_raw, skel_int)
    return endings, bifurs, img, binary, skeleton, skel_int


# ======================================================================
# 11. VISUALISASI OVERLAY MINUTIAE  (tata letak tidak diubah)
# ======================================================================
def gambar_overlay(img_gray, endings, bifurs, skel_int, theta_e=None, theta_b=None):
    color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for i, (x, y) in enumerate(endings):
        cv2.circle(color, (x, y), 4, (0, 0, 255), 1)
        sudut = theta_e[i] if theta_e is not None else hitung_arah(skel_int, x, y)
        x2 = int(x + PANJANG_GARIS * math.cos(sudut))
        y2 = int(y - PANJANG_GARIS * math.sin(sudut))
        cv2.line(color, (x, y), (x2, y2), (0, 0, 255), 1)

    for i, (x, y) in enumerate(bifurs):
        u = 5
        segitiga = np.array([(x, y - u), (x - u, y + u), (x + u, y + u)])
        cv2.polylines(color, [segitiga], True, (255, 0, 0), 1)
        sudut = theta_b[i] if theta_b is not None else hitung_arah(skel_int, x, y)
        x2 = int(x + PANJANG_GARIS * math.cos(sudut))
        y2 = int(y - PANJANG_GARIS * math.sin(sudut))
        cv2.line(color, (x, y), (x2, y2), (255, 0, 0), 1)

    return color


# ======================================================================
# 12. BACA / TULIS BERKAS CIRI MINUTIAE  (bentuk berkas tidak diubah)
# ======================================================================
def path_json_minutiae(nama_atau_path):
    return os.path.join(FOLDER_MINUTIAE, f'{nama_dasar(nama_atau_path)}_minutiae.json')


def cari_json_untuk_bmp(nama_bmp):
    p = path_json_minutiae(nama_bmp)
    return p if os.path.exists(p) else None


def muat_minutiae_dari_json(json_path, dengan_arah=False):
    """
    Membaca berkas ciri minutiae.
    dengan_arah=False -> ((x, y), ...)
    dengan_arah=True  -> ((x, y, theta), ...)
    """
    if not json_path or not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        print(f"    Gagal membaca: {os.path.basename(json_path)}")
        return None

    def ambil(kunci):
        out = []
        for m in data.get(kunci, []):
            if dengan_arah:
                ry = m.get('ridge')
                rx = m.get('ridge_x')
                if ry is not None:
                    ry = np.asarray(ry, dtype=np.float32)
                    rx = (np.asarray(rx, dtype=np.float32) if rx is not None
                          else np.linspace(0.0, 1.0, len(ry), dtype=np.float32))
                    out.append((int(m['x']), int(m['y']),
                                float(m.get('theta', 0.0)),
                                np.stack([rx, ry])))
                else:
                    out.append((int(m['x']), int(m['y']),
                                float(m.get('theta', 0.0))))
            else:
                out.append((int(m['x']), int(m['y'])))
        return out

    return ambil('endings'), ambil('bifurcations')


# ======================================================================
# 13. [P1] BAB III -- PENCOCOKAN MINUTIAE
# ======================================================================
# ----------------------------------------------------------------------

def _bungkus_sudut(a):
    """Membawa sudut ke rentang (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rapikan(daftar, tipe):
    """
    Menyeragamkan bentuk minutia menjadi (x, y, theta, tipe, ridge)
    dengan ridge berukuran (2, P1_RIDGE_SAMPEL): baris 0 = x lokal,
    baris 1 = y lokal (d_i pada [P1] Gbr. 13).
    """
    kosong = np.zeros((2, P1_RIDGE_SAMPEL), np.float32)

    def bentuk(r):
        if r is None:
            return kosong.copy()
        a = np.asarray(r, np.float32)
        if a.ndim == 1:                     # berkas ciri lama: hanya d_i
            return np.stack([np.linspace(0.0, 1.0, len(a), dtype=np.float32), a])
        return a

    out = []
    for m in daftar:
        if isinstance(m, dict):
            out.append((int(m['x']), int(m['y']), float(m.get('theta', 0.0)),
                        m.get('tipe', tipe), bentuk(m.get('ridge'))))
        elif len(m) >= 4 and isinstance(m[3], (np.ndarray, list, tuple)):
            out.append((int(m[0]), int(m[1]), float(m[2]), tipe, bentuk(m[3])))
        elif len(m) >= 3:
            out.append((int(m[0]), int(m[1]), float(m[2]), tipe, kosong.copy()))
        else:
            out.append((int(m[0]), int(m[1]), 0.0, tipe, kosong.copy()))
    return out


def transformasi_minutiae(daftar, phi, tx, ty):
    """
    Memutar sebesar phi terhadap titik (0,0) citra lalu menggeser (tx, ty).
    """
    c, s = math.cos(phi), math.sin(phi)
    keluar = []
    for m in daftar:
        x, y, th, tipe = m[0], m[1], m[2], m[3]
        sisa = m[4:] if len(m) > 4 else ()
        keluar.append((x * c + y * s + tx,
                       -x * s + y * c + ty,
                       _bungkus_sudut(th + phi), tipe) + tuple(sisa))
    return keluar


def cari_kandidat_transformasi(m1, m2, jumlah=None):
    """Penyelarasan lewat pencocokan ridge, [P1] Bab III-A dan Gbr. 13."""
    jumlah = jumlah or P1_KANDIDAT_SELARAS
    if not m1 or not m2:
        return []

    X1 = np.stack([m[4][0] for m in m1])       # (n1, L) x lokal
    Y1 = np.stack([m[4][1] for m in m1])       # (n1, L) d_i
    X2 = np.stack([m[4][0] for m in m2])
    Y2 = np.stack([m[4][1] for m in m2])
    tipe1 = np.array([m[3] == 'E' for m in m1])
    tipe2 = np.array([m[3] == 'E' for m in m2])

    norm1 = np.sqrt((Y1 ** 2).sum(axis=1))
    norm2 = np.sqrt((Y2 ** 2).sum(axis=1))
    S = (Y1 @ Y2.T) / np.maximum(norm1[:, None] * norm2[None, :], 1e-9)
    S[tipe1[:, None] != tipe2[None, :]] = -np.inf

    n2 = S.shape[1]
    urut = np.argsort(S, axis=None, kind='stable')[::-1]

    G1 = np.arctan2(Y1, X1)
    G2 = np.arctan2(Y2, X2)

    hasil, cadangan = [], []
    for k in urut[:jumlah * 8]:
        i, j = divmod(int(k), n2)
        s_ij = float(S[i, j])
        if not np.isfinite(s_ij):
            break
        _d = G2[j] - G1[i]
        _rata = math.atan2(float(np.sin(_d).sum()), float(np.cos(_d).sum()))
        phi = _bungkus_sudut(m2[j][2] - m1[i][2] + _rata)
        c, s = math.cos(phi), math.sin(phi)
        x1, y1 = m1[i][0], m1[i][1]
        tx = m2[j][0] - (x1 * c + y1 * s)
        ty = m2[j][1] - (-x1 * s + y1 * c)
        calon = (phi, tx, ty, s_ij, j)
        if s_ij > P1_AMBANG_RIDGE_SELARAS:
            hasil.append(calon)
        else:
            cadangan.append(calon)
        if len(hasil) >= jumlah:
            break

    if not hasil:
        hasil = cadangan[:jumlah]
    return hasil


# ----------------------------------------------------------------------
# 13b. Pencocokan string elastis di koordinat polar ([P1] Pers. 7-16)
# ----------------------------------------------------------------------
def _ke_polar(daftar, x0, y0):
    """Mengubah minutiae menjadi string polar terhadap titik acuan (x0, y0)."""
    out = []
    for m in daftar:
        dx, dy = m[0] - x0, m[1] - y0
        r = math.hypot(dx, dy)
        e = math.atan2(-dy, dx)          # sumbu y citra ke bawah
        out.append((r, e, m[2], m[3]))
    out.sort(key=lambda t: t[1])
    return out


# ----------------------------------------------------------------------
# INTI PEMROGRAMAN DINAMIS TERKOMPILASI (opsional)
# ----------------------------------------------------------------------
try:
    from numba import njit as _njit
    _ADA_NUMBA = True
except Exception:
    _ADA_NUMBA = False
    def _njit(*a, **k):
        def _bungkus(f):
            return f
        return _bungkus


@_njit(cache=True, fastmath=False)
def _dp_inti(r1a, e1a, t1a, k1a, r2a, e2a, t2a, k2a,
             lebar_r, lebar_e, rho, a_, b_, g_, OM, eta, harfiah):
    """Matriks jarak sunting [P1] Pers. 7-16. Mengembalikan matriks jejak."""
    n1 = r1a.shape[0]
    n2 = r2a.shape[0]
    Cm = np.empty((n1 + 1, n2 + 1))
    jejak = np.zeros((n1 + 1, n2 + 1), np.int8)
    pus_r = np.zeros((n1 + 1, n2 + 1))
    pus_e = np.zeros((n1 + 1, n2 + 1))
    Cm[0, 0] = 0.0
    for i in range(1, n1 + 1):
        Cm[i, 0] = i * OM
        jejak[i, 0] = 1
    for j in range(1, n2 + 1):
        Cm[0, j] = j * OM
        jejak[0, j] = 2
    for i in range(1, n1 + 1):
        r1 = r1a[i - 1]
        e1d = e1a[i - 1]
        t1d = t1a[i - 1]
        k1 = k1a[i - 1]
        for j in range(1, n2 + 1):
            pr = pus_r[i - 1, j - 1]
            pe = pus_e[i - 1, j - 1]
            dr_mentah = r1 - r2a[j - 1]
            de_mentah = (e1d - e2a[j - 1] + 180.0) % 360.0 - 180.0
            dt = (t1d - t2a[j - 1] + 180.0) % 360.0 - 180.0
            if dt < 0.0:
                dt = -dt
            dr = dr_mentah - pr
            de = de_mentah - pe
            cocok = False
            biaya = OM
            if (-lebar_r < dr < lebar_r and -lebar_e < de < lebar_e
                    and dt < rho and k1 == k2a[j - 1]):
                biaya = (a_ * (dr if dr >= 0.0 else -dr)
                         + b_ * (de if de >= 0.0 else -de) + g_ * dt)
                cocok = True
            a = Cm[i - 1, j] + OM
            b = Cm[i, j - 1] + OM
            c = Cm[i - 1, j - 1] + biaya
            if c <= a and c <= b:
                Cm[i, j] = c
                jejak[i, j] = 3 if cocok else 4
                if cocok:
                    pus_r[i, j] = pr + eta * (dr_mentah if harfiah else dr)
                    pus_e[i, j] = pe + eta * (de_mentah if harfiah else de)
                else:
                    pus_r[i, j] = pr
                    pus_e[i, j] = pe
            elif a <= b:
                Cm[i, j] = a
                jejak[i, j] = 1
                pus_r[i, j] = pus_r[i - 1, j]
                pus_e[i, j] = pus_e[i - 1, j]
            else:
                Cm[i, j] = b
                jejak[i, j] = 2
                pus_r[i, j] = pus_r[i, j - 1]
                pus_e[i, j] = pus_e[i, j - 1]
    return jejak


def _string_matching_elastis(s1, s2):
    """Pencocokan string elastis dengan pemrograman dinamis, [P1] Pers. 7-16."""
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return []

    lebar_r = float(P1_KOTAK_R)          # delta_h - pusat  (piksel)
    lebar_e = float(P1_KOTAK_E)          # epsilon_h - pusat (derajat)
    rho = float(P1_TOLERANSI_SUDUT)      # derajat
    a_, b_, g_ = P1_BOBOT_R, P1_BOBOT_E, P1_BOBOT_THETA
    OM = float(P1_PENALTI)
    eta = float(P1_LAJU_BELAJAR)
    harfiah = bool(globals().get('P1_KOTAK_ADAPTIF_HARFIAH', False))
    BESAR = 1e18

    def beda_sudut_deg(a, b):
        """Selisih dua sudut dalam derajat, hasil di (-180, 180]."""
        d = (a - b + 180.0) % 360.0 - 180.0
        return d

    _deg = math.degrees

    if _ADA_NUMBA:
        _r1 = np.array([p[0] for p in s1], np.float64)
        _e1 = np.degrees(np.array([p[1] for p in s1], np.float64))
        _t1 = np.degrees(np.array([p[2] for p in s1], np.float64))
        _r2 = np.array([p[0] for p in s2], np.float64)
        _e2 = np.degrees(np.array([p[1] for p in s2], np.float64))
        _t2 = np.degrees(np.array([p[2] for p in s2], np.float64))
        _kode = {}
        def _ke_angka(v):
            if v not in _kode:
                _kode[v] = len(_kode)
            return _kode[v]
        _k1 = np.array([_ke_angka(p[3]) for p in s1], np.int64)
        _k2 = np.array([_ke_angka(p[3]) for p in s2], np.int64)
        jejak = _dp_inti(_r1, _e1, _t1, _k1, _r2, _e2, _t2, _k2,
                         lebar_r, lebar_e, rho, float(a_), float(b_),
                         float(g_), OM, eta, harfiah)
        pasangan = []
        i, j = n1, n2
        while i > 0 and j > 0:
            arah = jejak[i, j]
            if arah == 3:
                r1, e1, t1, _ = s1[i - 1]
                r2, e2, t2, _ = s2[j - 1]
                pasangan.append((i - 1, j - 1, abs(r1 - r2),
                                 abs(_bungkus_sudut(t1 - t2))))
                i -= 1
                j -= 1
            elif arah == 4:
                i -= 1
                j -= 1
            elif arah == 1:
                i -= 1
            else:
                j -= 1
        pasangan.reverse()
        return pasangan

    S2 = [(p[0], _deg(p[1]), _deg(p[2]), p[3]) for p in s2]

    C = [[BESAR] * (n2 + 1) for _ in range(n1 + 1)]
    jejak = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    pus_r = [[0.0] * (n2 + 1) for _ in range(n1 + 1)]   # pusat kotak radius
    pus_e = [[0.0] * (n2 + 1) for _ in range(n1 + 1)]   # pusat kotak sudut

    C[0][0] = 0.0
    for i in range(1, n1 + 1):
        C[i][0] = i * OM
        jejak[i][0] = 1
    for j in range(1, n2 + 1):
        C[0][j] = j * OM
        jejak[0][j] = 2

    for i in range(1, n1 + 1):
        r1, e1, t1, k1 = s1[i - 1]
        e1d, t1d = math.degrees(e1), math.degrees(t1)
        Ci, Cim, Ji = C[i], C[i - 1], jejak[i]
        pri, prim = pus_r[i], pus_r[i - 1]
        pei, peim = pus_e[i], pus_e[i - 1]
        for j in range(1, n2 + 1):
            r2, e2d, t2d, k2 = S2[j - 1]
            pr, pe = prim[j - 1], peim[j - 1]

            dr_mentah = r1 - r2
            de_mentah = (e1d - e2d + 180.0) % 360.0 - 180.0
            dt = (t1d - t2d + 180.0) % 360.0 - 180.0
            if dt < 0.0:
                dt = -dt
            dr, de = dr_mentah - pr, de_mentah - pe

            if (-lebar_r < dr < lebar_r and -lebar_e < de < lebar_e
                    and dt < rho and k1 == k2):
                biaya = (a_ * (dr if dr >= 0.0 else -dr)
                         + b_ * (de if de >= 0.0 else -de) + g_ * dt)
                cocok = True
            else:
                biaya = OM
                cocok = False

            a = Cim[j] + OM
            b = Ci[j - 1] + OM
            c = Cim[j - 1] + biaya

            if c <= a and c <= b:
                Ci[j], Ji[j] = c, (3 if cocok else 4)
                if cocok:
                    tambah_r = dr_mentah if harfiah else dr
                    tambah_e = de_mentah if harfiah else de
                    pri[j] = pr + eta * tambah_r
                    pei[j] = pe + eta * tambah_e
                else:
                    pri[j], pei[j] = pr, pe
            elif a <= b:
                Ci[j], Ji[j] = a, 1
                pri[j], pei[j] = prim[j], peim[j]
            else:
                Ci[j], Ji[j] = b, 2
                pri[j], pei[j] = pri[j - 1], pei[j - 1]

    pasangan = []
    i, j = n1, n2
    while i > 0 and j > 0:
        arah = jejak[i][j]
        if arah == 3:
            r1, e1, t1, _ = s1[i - 1]
            r2, e2, t2, _ = s2[j - 1]
            pasangan.append((i - 1, j - 1, abs(r1 - r2),
                             abs(_bungkus_sudut(t1 - t2))))
            i -= 1
            j -= 1
        elif arah == 4:
            i -= 1
            j -= 1
        elif arah == 1:
            i -= 1
        else:
            j -= 1
    pasangan.reverse()
    return pasangan


def _hitung_cepat(m1t, m2):
    """Penghitung kasar berapa minutia yang jatuh di dalam kotak pembatas."""
    if not m1t or not m2:
        return 0
    a = np.array([(m[0], m[1], m[2]) for m in m1t], np.float32)
    b = np.array([(m[0], m[1], m[2]) for m in m2], np.float32)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    dt = np.abs(np.arctan2(np.sin(a[:, None, 2] - b[None, :, 2]),
                           np.cos(a[:, None, 2] - b[None, :, 2])))
    ok = (d < P1_KOTAK_R) & (dt < math.radians(P1_TOLERANSI_SUDUT))
    return int(min(ok.any(axis=1).sum(), ok.any(axis=0).sum()))


def cocokkan_minutiae_lengkap(e1, b1, e2, b2):
    """Pencocokan minutiae lengkap, [P1] Gbr. 11."""
    m1 = _rapikan(e1, 'E') + _rapikan(b1, 'B')
    m2 = _rapikan(e2, 'E') + _rapikan(b2, 'B')

    if not m1 or not m2:
        return {'jumlah_cocok': 0, 'skor': 0.0, 'pasangan': [],
                'total_uji': len(m1), 'total_galeri': len(m2),
                'alignment': {'dipakai': bool(PAKAI_ALIGNMENT),
                              'rotasi_deg': 0.0, 'geser_x': 0.0,
                              'geser_y': 0.0, 'suara': 0}}

    daftar_transformasi = [(0.0, 0.0, 0.0, 0.0, None)]     # tanpa penyelarasan
    if PAKAI_ALIGNMENT:
        daftar_transformasi += cari_kandidat_transformasi(m1, m2)

    peringkat = []
    for tr in daftar_transformasi:
        phi, tx, ty, _, _ = tr
        m1t = m1 if (phi == 0.0 and tx == 0.0 and ty == 0.0) \
            else transformasi_minutiae(m1, phi, tx, ty)
        peringkat.append((_hitung_cepat(m1t, m2), tr, m1t))
    peringkat.sort(key=lambda t: -t[0])

    terbaik_pas, terbaik_tr, terbaik_m1 = [], daftar_transformasi[0], m1
    for _, tr, m1t in peringkat[:P1_KANDIDAT_DP]:
        j_acuan = tr[4]
        if j_acuan is not None:
            x0, y0 = m2[j_acuan][0], m2[j_acuan][1]
        else:
            x0 = sum(m[0] for m in m2) / len(m2)
            y0 = sum(m[1] for m in m2) / len(m2)
        s1 = _ke_polar(m1t, x0, y0)
        s2 = _ke_polar(m2, x0, y0)
        pas = _string_matching_elastis(s1, s2)
        if len(pas) > len(terbaik_pas):
            peta1 = sorted(range(len(m1t)),
                           key=lambda k: math.atan2(-(m1t[k][1] - y0),
                                                    m1t[k][0] - x0))
            peta2 = sorted(range(len(m2)),
                           key=lambda k: math.atan2(-(m2[k][1] - y0),
                                                    m2[k][0] - x0))
            terbaik_pas = [(peta1[i], peta2[j], d, ds) for i, j, d, ds in pas]
            terbaik_tr, terbaik_m1 = tr, m1t

    phi, tx, ty, mutu, _ = terbaik_tr

    pasangan = []
    for no, (i, j, d, ds) in enumerate(terbaik_pas, start=1):
        asli, selaras, p2 = m1[i], terbaik_m1[i], m2[j]
        pasangan.append({
            'no': no,
            'tipe': 'Ridge Ending' if asli[3] == 'E' else 'Bifurcation',
            'tipe_kode': asli[3],
            'x1': asli[0], 'y1': asli[1], 'theta1_deg': math.degrees(asli[2]),
            'x1_selaras': round(float(selaras[0]), 1),
            'y1_selaras': round(float(selaras[1]), 1),
            'x2': p2[0], 'y2': p2[1], 'theta2_deg': math.degrees(p2[2]),
            'jarak': float(d),
            'selisih_sudut_deg': math.degrees(ds),
        })

    M, N = len(m2), len(m1)
    Mpq = len(pasangan)
    skor = (100.0 * Mpq * Mpq / (M * N)) if (M > 0 and N > 0) else 0.0

    return {
        'jumlah_cocok': len(pasangan),
        'skor': skor,
        'pasangan': pasangan,
        'total_uji': len(m1),
        'total_galeri': len(m2),
        'alignment': {
            'dipakai': bool(PAKAI_ALIGNMENT),
            'rotasi_deg': math.degrees(phi),
            'geser_x': float(tx),
            'geser_y': float(ty),
            'suara': round(float(mutu), 3),
        },
    }


def cocokkan_minutiae(e1, b1, e2, b2):
    """Bentuk keluaran tetap: (jumlah_cocok, skor, pasangan, tot1, tot2)."""
    h = cocokkan_minutiae_lengkap(e1, b1, e2, b2)
    return (h['jumlah_cocok'], h['skor'], h['pasangan'],
            h['total_uji'], h['total_galeri'])


# ======================================================================
# 14. [P2] -- FingerCode (Jain dkk. 2000)
# ======================================================================

def titik_acuan(img_gray):
    """Penentuan titik acuan, [P2] Bab II-A, Pers. 6-15."""
    h, w = img_gray.shape
    pusat = (w // 2, h // 2)
    radius = None

    for w_blok in P2_ITERASI_W:
        o, cert, energi = medan_orientasi(img_gray, W=w_blok, D=P2_SMOOTH)
        E = np.sin(o).astype(np.float32)               # Pers. 14

        mask_blok = segmentasi_certainty(cert, energi,
                                         (cert.shape[0] * w_blok,
                                          cert.shape[1] * w_blok),
                                         W=w_blok)[::w_blok, ::w_blok]
        mask_blok = mask_blok[:E.shape[0], :E.shape[1]]
        if mask_blok.shape != E.shape:
            m = np.zeros(E.shape, np.uint8)
            m[:mask_blok.shape[0], :mask_blok.shape[1]] = mask_blok
            mask_blok = m
        E = E * (mask_blok > 0)

        rr = max(2, int(round(P2_JARI_INTEGRASI / w_blok * 4)))
        sisi = 2 * rr + 1
        yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        cakram = (xx ** 2 + yy ** 2) <= rr ** 2
        RI = np.where(cakram & (xx < 0), 1.0, 0.0).astype(np.float32)
        RII = np.where(cakram & (xx >= 0), 1.0, 0.0).astype(np.float32)
        A = cv2.filter2D(E, cv2.CV_32F, RI) - cv2.filter2D(E, cv2.CV_32F, RII)

        A[mask_blok == 0] = -1e9
        tepi = max(1, rr // 2)
        A[:tepi, :] = -1e9
        A[-tepi:, :] = -1e9
        A[:, :tepi] = -1e9
        A[:, -tepi:] = -1e9

        if radius is not None:
            batas = np.full(A.shape, -1e9, np.float32)
            bx, by = pusat[0] // w_blok, pusat[1] // w_blok
            r_blok = max(1, int(round(radius / w_blok)))
            y1, y2 = max(0, by - r_blok), min(A.shape[0], by + r_blok + 1)
            x1, x2 = max(0, bx - r_blok), min(A.shape[1], bx + r_blok + 1)
            if y2 > y1 and x2 > x1:
                batas[y1:y2, x1:x2] = A[y1:y2, x1:x2]
                A = batas

        iy, ix = np.unravel_index(int(np.argmax(A)), A.shape)
        pusat = (int(ix * w_blok + w_blok // 2), int(iy * w_blok + w_blok // 2))
        radius = 2 * w_blok

    return pusat


def daftar_sektor(x0=None, y0=None):
    """Tesselasi melingkar, [P2] Pers. 1-5 dan Gbr. 4."""
    b = P2_LEBAR_PITA
    k = P2_SEKTOR_PITA
    r0 = P2_JARI_DALAM
    sel = []
    for i in range(P2_JUMLAH_SEKTOR):
        pita = i // k
        seg = i % k
        r_dalam = r0 + pita * b
        r_luar = r_dalam + b
        a1 = seg * (2 * math.pi / k)
        a2 = a1 + (2 * math.pi / k)
        rm = 0.5 * (r_dalam + r_luar)
        am = 0.5 * (a1 + a2)
        xc = rm * math.cos(am)
        yc = rm * math.sin(am)
        sel.append({
            'indeks': i, 'baris': pita, 'kolom': seg,
            'r_dalam': r_dalam, 'r_luar': r_luar,
            'sudut_awal_deg': round(math.degrees(a1), 2),
            'sudut_akhir_deg': round(math.degrees(a2), 2),
            'x1': int(round((x0 or 0) + r_dalam * math.cos(a1))),
            'y1': int(round((y0 or 0) + r_dalam * math.sin(a1))),
            'x2': int(round((x0 or 0) + r_luar * math.cos(a2))),
            'y2': int(round((y0 or 0) + r_luar * math.sin(a2))),
            'x_pusat': int(round((x0 or 0) + xc)),
            'y_pusat': int(round((y0 or 0) + yc)),
        })
    return sel


def _peta_sektor():
    """
    Peta indeks sektor pada bingkai ROI berukuran 2R x 2R, dihitung sekali
    lalu disimpan. Nilai -1 berarti di luar daerah minat.
    """
    if 'peta_sektor' in _KONTEKS:
        return _KONTEKS['peta_sektor']
    R = P2_JARI_LUAR
    yy, xx = np.mgrid[-R:R, -R:R].astype(np.float32)
    r = np.hypot(xx, yy)
    a = np.mod(np.arctan2(yy, xx), 2 * math.pi)
    pita = np.floor((r - P2_JARI_DALAM) / P2_LEBAR_PITA).astype(int)
    seg = np.floor(a / (2 * math.pi / P2_SEKTOR_PITA)).astype(int)
    seg = np.clip(seg, 0, P2_SEKTOR_PITA - 1)
    peta = pita * P2_SEKTOR_PITA + seg
    peta[(pita < 0) | (pita >= P2_JUMLAH_PITA)] = -1
    _KONTEKS['peta_sektor'] = peta
    return peta


def ambil_roi(img_gray, pusat):
    """Memotong daerah minat berukuran 2R x 2R di sekitar titik acuan."""
    R = P2_JARI_LUAR
    h, w = img_gray.shape
    x0, y0 = pusat
    roi = np.zeros((2 * R, 2 * R), np.float32)
    sah = np.zeros((2 * R, 2 * R), bool)

    xs, xe = x0 - R, x0 + R
    ys, ye = y0 - R, y0 + R
    sx1, sx2 = max(0, xs), min(w, xe)
    sy1, sy2 = max(0, ys), min(h, ye)
    if sx2 > sx1 and sy2 > sy1:
        roi[sy1 - ys:sy2 - ys, sx1 - xs:sx2 - xs] = img_gray[sy1:sy2, sx1:sx2]
        sah[sy1 - ys:sy2 - ys, sx1 - xs:sx2 - xs] = True
    return roi, sah


def normalisasi_sektor(roi, sah, peta):
    """Normalisasi tiap sektor ke rerata M0 dan variansi V0, [P2] Pers. 16."""
    keluar = np.full(roi.shape, P2_M0, np.float32)
    for i in range(P2_JUMLAH_SEKTOR):
        m = (peta == i) & sah
        n = int(m.sum())
        if n < 8:
            continue
        nilai = roi[m]
        Mi = float(nilai.mean())
        Vi = float(nilai.var())
        if Vi < 1e-6:
            keluar[m] = P2_M0
            continue
        d = np.sqrt(P2_V0 * (nilai - Mi) ** 2 / Vi)
        keluar[m] = np.where(nilai > Mi, P2_M0 + d, P2_M0 - d)
    return keluar


def buat_gabor_kernels():
    """Bank filter Gabor genap-simetris, [P2] Pers. 17-19."""
    if 'kernel_gabor' in _KONTEKS:
        return _KONTEKS['kernel_gabor']

    n = P2_GABOR_MASK // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    f = 1.0 / P2_GABOR_K
    dx = dy = P2_GABOR_DELTA

    kernels = []
    for i in range(P2_ORIENTASI):
        theta = i * (math.pi / P2_ORIENTASI)
        xa = xx * math.sin(theta) + yy * math.cos(theta)
        ya = xx * math.cos(theta) - yy * math.sin(theta)
        g = np.exp(-0.5 * (xa ** 2 / dx ** 2 + ya ** 2 / dy ** 2)) \
            * np.cos(2 * math.pi * f * xa)
        g[np.abs(g) <= P2_GABOR_MIN] = 0.0
        kernels.append(g.astype(np.float32))

    _KONTEKS['kernel_gabor'] = kernels
    return kernels


def gabor_responses(citra):
    """Tanggapan bank filter untuk tiap arah, [P2] Bab II-B."""
    kernels = buat_gabor_kernels()
    c = citra.astype(np.float32)
    return [cv2.filter2D(c, cv2.CV_32F, k) for k in kernels]


def _aad_fingercode(responses, peta, sah):
    """Ciri AAD tiap sektor tiap arah, [P2] Pers. 20."""
    vektor = np.zeros(GABOR_DIMENSI, np.float32)
    for o, resp in enumerate(responses):
        for i in range(P2_JUMLAH_SEKTOR):
            m = (peta == i) & sah
            if int(m.sum()) < 8:
                continue
            nilai = resp[m]
            vektor[o * P2_JUMLAH_SEKTOR + i] = float(
                np.abs(nilai - nilai.mean()).mean())
    return vektor


def fingercode_dari_citra(img_gray, pusat=None):
    """FingerCode lengkap dari satu citra, [P2] Bab II."""
    if pusat is None:
        pusat = titik_acuan(img_gray)
    roi, sah = ambil_roi(img_gray, pusat)
    peta = _peta_sektor()
    norm = normalisasi_sektor(roi, sah, peta)
    responses = gabor_responses(norm)
    vektor = _aad_fingercode(responses, peta, sah)

    luar = int(((peta >= 0) & (~sah)).sum())
    total = int((peta >= 0).sum())
    data = {'pusat': pusat, 'roi': roi, 'sah': sah, 'peta': peta,
            'norm': norm, 'responses': responses,
            'roi_terpotong': luar > 0,
            'persen_terpotong': 100.0 * luar / max(total, 1)}
    return vektor, data


def fingercode_pasangan(img_gray):
    """DUA FingerCode untuk satu sidik jari, [P2] Bab III."""
    v0, data = fingercode_dari_citra(img_gray)
    px, py = float(data['pusat'][0]), float(data['pusat'][1])
    M = cv2.getRotationMatrix2D((px, py), P2_ROTASI_HALUS, 1.0)
    citra_putar = cv2.warpAffine(img_gray, M,
                                 (img_gray.shape[1], img_gray.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    v1, _ = fingercode_dari_citra(citra_putar, pusat=data['pusat'])
    return v0, v1, data


def putar_fingercode(v, langkah):
    """Putaran siklis FingerCode, [P2] Pers. 21-23:"""
    k = P2_SEKTOR_PITA
    v = np.asarray(v, np.float32)
    out = np.empty_like(v)
    for o in range(P2_ORIENTASI):
        o_asal = (o + langkah) % P2_ORIENTASI          # Pers. 23
        for i in range(P2_JUMLAH_SEKTOR):
            pita = i // k
            seg = (i + k + langkah) % k                # Pers. 22
            out[o * P2_JUMLAH_SEKTOR + i] = \
                v[o_asal * P2_JUMLAH_SEKTOR + pita * k + seg]
    return out


def jarak_fingercode(v_uji, v_galeri):
    """Jarak pencocokan, [P2] Bab III."""
    a = np.asarray(v_uji, np.float32)[:GABOR_DIMENSI]
    b = np.asarray(v_galeri, np.float32)
    template = [b[:GABOR_DIMENSI]]
    if b.size >= 2 * GABOR_DIMENSI:
        template.append(b[GABOR_DIMENSI:2 * GABOR_DIMENSI])
    return float(min(np.linalg.norm(a - putar_fingercode(t, s))
                     for t in template for s in P2_ROTASI_LANGKAH))


# ----------------------------------------------------------------------
# 14b. Pembantu indeks FingerCode
# ----------------------------------------------------------------------
def indeks_vektor(orientasi, baris, kolom):
    """Letak nilai ciri di dalam FingerCode."""
    i = orientasi * P2_JUMLAH_SEKTOR + baris * P2_SEKTOR_PITA + kolom
    return i, i


def jarak_ternormalisasi(v1, v2):
    """Jarak Euclidean antar-FingerCode, [P2] Bab III."""
    return jarak_fingercode(v1, v2)


def skor_kemiripan(v1, v2):
    """Skor kemiripan 0 sampai 100."""
    d = jarak_fingercode(v1, v2) / math.sqrt(GABOR_DIMENSI)   # N = 640
    return float(max(0.0, min(100.0, 100.0 - d)))


def peta_kontribusi(v1, v2):
    """
    Sumbangan tiap sektor terhadap jarak kuadrat, untuk visualisasi.
    Bentuk keluarannya (jumlah pita x sektor per pita).
    """
    a = np.asarray(v1, np.float64)[:GABOR_DIMENSI]
    b = np.asarray(v2, np.float64)[:GABOR_DIMENSI]
    langkah_terbaik = min(P2_ROTASI_LANGKAH,
                          key=lambda s: np.linalg.norm(a - putar_fingercode(b, s)))
    b = putar_fingercode(b.astype(np.float32), langkah_terbaik).astype(np.float64)
    diff2 = (a - b) ** 2
    kontrib = np.zeros((P2_JUMLAH_PITA, P2_SEKTOR_PITA))
    for o in range(P2_ORIENTASI):
        for p in range(P2_JUMLAH_PITA):
            for s in range(P2_SEKTOR_PITA):
                kontrib[p, s] += diff2[o * P2_JUMLAH_SEKTOR + p * P2_SEKTOR_PITA + s]
    return kontrib


def muat_template_gabor(folder=None):
    """Semua FingerCode yang tersimpan."""
    folder = folder or FOLDER_GABOR_FITUR
    templates = []
    for path_json in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path_json, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        if "vektor" not in data:
            continue
        templates.append({
            "nama": data.get("source", os.path.basename(path_json)),
            "nama_dasar": nama_dasar(data.get("source", path_json)),
            "vektor": np.array(data["vektor"], dtype=np.float32),
            "path": path_json,
        })
    return templates
