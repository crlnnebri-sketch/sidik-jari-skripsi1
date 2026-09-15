import csv
import json

import matplotlib.pyplot as plt

from config import *


GABOR_FOLDER = FOLDER_GABOR_FITUR
FOLDER_HASIL_PNG = FOLDER_HASIL_GABOR


# ======================================================================
# PEMILIHAN CITRA
# ======================================================================
def pilih_citra():

    print("\n" + "=" * 60)
    print("  CITRA MANA YANG MAU DIEKSTRAKSI?")
    print("=" * 60)
    print(f"  1 = DATASET       (citra galeri, {JUMLAH_DATASET} citra per subjek)")
    print(f"  2 = IDENTIFIKASI  (citra uji,    {JUMLAH_IDENTIFIKASI} citra per subjek)")
    pilih = tanya_pilihan("Pilihan Anda", ['1', '2'])

    if pilih == '1':
        label, cari, pesan = 'DATASET', daftar_dataset, 'menu 1'
    else:
        label, cari, pesan = 'IDENTIFIKASI', daftar_identifikasi, 'menu 2'

    semua = cari()
    if not semua:
        print(f"\n  Belum ada citra {label} di raw_images/.")
        print(f"  Ambil citra terlebih dahulu!")
        return [], ''

    subjek = sorted({subjek_dari(f) for f in semua})
    print(f"\n  Subjek {label.lower()} yang tersedia: {', '.join(subjek)}")
    for k in subjek:
        milik = cari(kode_subjek=k)
        print(f"    {k:<8} {len(milik)} citra  "
              f"({nama_dasar(milik[0])} ... {nama_dasar(milik[-1])})")
    print("\n  Ketik kode subjek, atau ketik 'semua' untuk seluruh subjek.")

    while True:
        jawab = input("  Kode subjek: ").strip().lower()
        if jawab == 'semua':
            return semua, f'{label} seluruh subjek ({len(semua)} citra)'
        terpilih = cari(kode_subjek=jawab)
        if terpilih:
            return terpilih, f'{label} subjek {jawab} ({len(terpilih)} citra)'
        print(f"  Tidak ada berkas {label.lower()} untuk kode '{jawab}'. Coba lagi.")


# ======================================================================
# PEMROSESAN
# ======================================================================
def proses_satu_file(path_bmp):
    """Ekstraksi FingerCode satu berkas, [P2] Jain dkk. 2000."""
    gray = cv2.imread(path_bmp, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  Gagal membaca: {path_bmp}")
        return None, None, None

    vektor, vektor_11, d = fingercode_pasangan(gray)
    responses = d['responses']
    peta, sah = d['peta'], d['sah']

    print(f"    Titik acuan [P2] II-A: {d['pusat']}   "
          f"ROI jari-jari {P2_JARI_LUAR} px, "
          f"{P2_JUMLAH_PITA} pita x {P2_SEKTOR_PITA} sektor = "
          f"{P2_JUMLAH_SEKTOR} sektor")
    if d['roi_terpotong']:
        print(f"    Catatan: {d['persen_terpotong']:.1f}% luas ROI keluar "
              f"bingkai citra, sektornya diberi nilai nol.")

    sektor = daftar_sektor(d['pusat'][0], d['pusat'][1])
    tabel = []
    for o in range(GABOR_ORIENTASI):
        sudut = round(o * 180.0 / GABOR_ORIENTASI, 2)
        for s in sektor:
            i_aad, _ = indeks_vektor(o, s['baris'], s['kolom'])
            tabel.append({
                'orientasi': o, 'sudut_deg': sudut,
                'baris': s['baris'], 'kolom': s['kolom'],
                'x_pusat': s['x_pusat'], 'y_pusat': s['y_pusat'],
                'x1': s['x1'], 'y1': s['y1'], 'x2': s['x2'], 'y2': s['y2'],
                'idx_mean': i_aad, 'idx_std': i_aad,
                'mean': float(vektor[i_aad]), 'std': float(vektor[i_aad]),
            })

    combined = np.mean(np.stack([np.abs(r) for r in responses], axis=0), axis=0)

    heatmap = np.zeros((P2_JUMLAH_PITA, P2_SEKTOR_PITA))
    for i in range(P2_JUMLAH_SEKTOR):
        heatmap[i // P2_SEKTOR_PITA, i % P2_SEKTOR_PITA] = float(
            np.mean([vektor[o * P2_JUMLAH_SEKTOR + i]
                     for o in range(GABOR_ORIENTASI)]))

    data_visual = {'roi': d['roi'].astype(np.uint8), 'ternormalisasi': d['norm'],
                   'responses': [np.abs(r) for r in responses],
                   'combined': combined, 'heatmap': heatmap,
                   'peta': peta, 'sah': sah, 'pusat': d['pusat']}
    vektor_simpan = np.concatenate([vektor, vektor_11]).astype(np.float32)
    return vektor_simpan, data_visual, tabel


def simpan_ciri(path_bmp, vektor, tabel):
    nama = nama_dasar(path_bmp)
    info = info_berkas(path_bmp)

    data = {
        'source': os.path.basename(path_bmp),
        'nama': nama,
        'jenis': info['jenis'] if info else 'tidak dikenal',
        'kode_subjek': info['kode_subjek'] if info else None,
        'metode': 'FingerCode [P2] Jain dkk. 2000, IEEE TIP 9(5):846-859',
        'parameter': {
            'orientasi': GABOR_ORIENTASI,
            'frekuensi_f': GABOR_FREKUENSI,
            'inter_ridge_K': P2_GABOR_K,
            'kernel_size': P2_GABOR_MASK,
            'delta_x_y': P2_GABOR_DELTA,
            'ambang_mask': P2_GABOR_MIN,
            'jumlah_pita_B': P2_JUMLAH_PITA,
            'sektor_per_pita_k': P2_SEKTOR_PITA,
            'lebar_pita_b': P2_LEBAR_PITA,
            'jari_dalam_r0': P2_JARI_DALAM,
            'jari_luar': P2_JARI_LUAR,
            'normalisasi_M0': P2_M0,
            'normalisasi_V0': P2_V0,
        },
        'dimensi': int(GABOR_DIMENSI),
        'jumlah_fingercode': int(len(vektor) // GABOR_DIMENSI),
        'keterangan_vektor': ('dua FingerCode berurutan masing-masing 640 '
                              'nilai: citra asli lalu citra diputar 11,25 '
                              'derajat ([P2] Bab III, sepuluh template)'),
        'rumus_indeks': ('idx = orientasi*(B*k) + pita*k + sektor ; '
                         'satu nilai AAD tiap sektor tiap arah ([P2] Pers. 20)'),
        'grid_sel': daftar_sektor(),
        'sudut_orientasi_deg': [round(o * 180.0 / GABOR_ORIENTASI, 2)
                                for o in range(GABOR_ORIENTASI)],
        'vektor': [float(v) for v in vektor],
    }

    path_json = os.path.join(FOLDER_GABOR_FITUR, nama + '.json')
    with open(path_json, 'w') as f:
        json.dump(data, f, indent=2)

    path_csv = os.path.join(FOLDER_GABOR_FITUR, nama + '_sel.csv')
    with open(path_csv, 'w', newline='') as f:
        kolom = ['orientasi', 'sudut_deg', 'baris', 'kolom', 'x_pusat', 'y_pusat',
                 'x1', 'y1', 'x2', 'y2', 'idx_mean', 'idx_std', 'mean', 'std']
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        for baris in tabel:
            w.writerow({k: (round(baris[k], 6) if isinstance(baris[k], float)
                            else baris[k]) for k in kolom})

    print(f"  JSON ciri : {path_json}   (dimensi {GABOR_DIMENSI} x "
          f"{len(vektor) // GABOR_DIMENSI} FingerCode)")
    print(f"  CSV sel   : {path_csv}   ({len(tabel)} baris)")
    return path_json


def visualisasi_ekstraksi(nama_file, data_visual, path_simpan, tampilkan=False):
    n_baris_orientasi = (GABOR_ORIENTASI + 3) // 4
    total_baris = 1 + n_baris_orientasi
    fig, axes = plt.subplots(total_baris, 4,
                             figsize=(16, 3.3 * total_baris), facecolor='white')
    fig.subplots_adjust(hspace=0.22, wspace=0.15, top=0.90, bottom=0.03,
                        left=0.03, right=0.97)
    fig.suptitle(
        f"Ekstraksi FingerCode [P2] — {nama_file}\n"
        f"({GABOR_ORIENTASI} orientasi, {P2_JUMLAH_SEKTOR} sektor, "
        f"f={GABOR_FREKUENSI:.2f}, vektor {GABOR_DIMENSI}d)",
        fontsize=12, fontweight='bold')

    axes[0, 0].imshow(data_visual["roi"], cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f"ROI di sekitar titik acuan "
                         f"({2*P2_JARI_LUAR}x{2*P2_JARI_LUAR})",
                         fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(data_visual["ternormalisasi"], cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Normalisasi tiap sektor\n(M0=V0=100, [P2] Pers. 16)",
                         fontsize=9, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(data_visual["combined"], cmap='hot')
    axes[0, 2].set_title("Gabor combined response", fontsize=9, fontweight='bold')
    axes[0, 2].axis('off')

    im = axes[0, 3].imshow(data_visual["heatmap"], cmap='YlOrRd', interpolation='nearest')
    axes[0, 3].set_title(f"Peta ciri AAD "
                         f"({P2_JUMLAH_PITA} pita x {P2_SEKTOR_PITA} sektor)",
                         fontsize=9, fontweight='bold')
    axes[0, 3].set_xticks(range(0, P2_SEKTOR_PITA, 2))
    axes[0, 3].set_yticks(range(P2_JUMLAH_PITA))
    axes[0, 3].tick_params(labelsize=5)
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    vmaks = max(float(r.max()) for r in data_visual["responses"])
    for i in range(GABOR_ORIENTASI):
        ax = axes[1 + i // 4, i % 4]
        sudut = int(i * 180 / GABOR_ORIENTASI)
        ax.imshow(data_visual["responses"][i], cmap='hot', vmin=0, vmax=vmaks)
        ax.set_title(f"Orientasi {i} ({sudut} derajat)", fontsize=9, fontweight='bold')
        ax.axis('off')
    for j in range(GABOR_ORIENTASI, n_baris_orientasi * 4):
        axes[1 + j // 4, j % 4].axis('off')

    os.makedirs(os.path.dirname(path_simpan), exist_ok=True)
    fig.savefig(path_simpan, dpi=150, bbox_inches='tight', facecolor='white')
    if tampilkan:
        print("  >>> Tutup jendela gambar untuk melanjutkan.")
        plt.show()
    plt.close(fig)
    print(f"  Gambar    : {path_simpan}")


# ======================================================================
# PROGRAM UTAMA
# ======================================================================
def main():
    print("=" * 60)
    print("  BAGIAN 2 (TEKSTUR) - EKSTRAKSI FingerCode [P2]")
    print("=" * 60)
    print(f"  Parameter [P2]: {GABOR_ORIENTASI} orientasi, "
          f"{P2_JUMLAH_PITA} pita x {P2_SEKTOR_PITA} sektor, "
          f"f={GABOR_FREKUENSI:.2f}, dimensi vektor {GABOR_DIMENSI}")

    daftar, keterangan = pilih_citra()
    if not daftar:
        print("Tidak ada citra untuk diproses.")
        return

    print(f"\n  {keterangan}")
    tampilkan = True
    if len(daftar) > 1:
        tampilkan = tanya_ya_tidak(
            "  Tampilkan jendela gambar untuk setiap citra "
            "(tetap disimpan bila tidak)")

    berhasil = 0
    for path_bmp in daftar:
        nama = nama_dasar(path_bmp)
        print("\n" + "-" * 60)
        print(f"  {nama}")
        print("-" * 60)

        vektor, data_visual, tabel = proses_satu_file(path_bmp)
        if vektor is None:
            continue

        simpan_ciri(path_bmp, vektor, tabel)
        path_png = os.path.join(FOLDER_HASIL_GABOR, nama + '.png')
        visualisasi_ekstraksi(os.path.basename(path_bmp), data_visual,
                              path_png, tampilkan=tampilkan)
        berhasil += 1

    print("\n" + "=" * 60)
    print(f"  SELESAI — {berhasil} citra terekstraksi.")
    print(f"  Ciri     : {FOLDER_GABOR_FITUR}")
    print(f"  Gambar   : {FOLDER_HASIL_GABOR}")
    print(" LANJUT MATCHING!")
    print("=" * 60)


if __name__ == "__main__":
    main()
