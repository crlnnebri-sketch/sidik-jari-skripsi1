import csv
import json

import matplotlib.pyplot as plt

from config import *


# ====================================================================
# PEMILIHAN CITRA
# ====================================================================
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
        print(f"\n  Belum ada citra yang tersimpan {label} di raw_images/.")
        print(f"  Ambil citra terlebih dahulu.")
        return [], ''

    subjek = sorted({subjek_dari(f) for f in semua})
    print(f"\n  Subjek {label.lower()} yang tersedia: {', '.join(subjek)}")
    for k in subjek:
        milik = cari(kode_subjek=k)
        print(f"    {k:<8} {len(milik)} citra  "
              f"({nama_dasar(milik[0])} ... {nama_dasar(milik[-1])})")
    print("\n  Masukkan kode subjek, atau ketik 'semua' untuk seluruh subjek.")

    while True:
        jawab = input("  Kode subjek: ").strip().lower()
        if jawab == 'semua':
            return semua, f'{label} seluruh subjek ({len(semua)} citra)'
        terpilih = cari(kode_subjek=jawab)
        if terpilih:
            return terpilih, f'{label} subjek {jawab} ({len(terpilih)} citra)'
        print(f"  Tidak ada berkas {label.lower()} untuk kode '{jawab}'. Coba lagi.")


# ====================================================================
# PEMROSESAN SATU CITRA
# ====================================================================
def proses_satu(path_bmp, tampilkan=True):
    nama = nama_dasar(path_bmp)
    info = info_berkas(path_bmp)

    print("\n" + "-" * 60)
    print(f"  {nama}   [{info['jenis']}, subjek {info['kode_subjek']}]")
    print("-" * 60)

    img_full = cv2.imread(path_bmp, cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        print("  Gagal membaca berkas.")
        return None

    img = siapkan_medan_dan_segmentasi(img_full)
    binary = preprocessing(img)
    skeleton = thinning(binary)
    endings_raw, bifurs_raw, skel_int = ekstraksi_minutiae(skeleton)
    endings, bifurs = filter_minutiae(endings_raw, bifurs_raw, skel_int)

    theta_e = [hitung_arah(skel_int, x, y) for x, y in endings]
    theta_b = [hitung_arah(skel_int, x, y) for x, y in bifurs]

    D_ridge = hitung_inter_ridge_width(skel_int)
    ridge_e = [bentuk_ridge(skel_int, x, y, th, D_ridge)
               for (x, y), th in zip(endings, theta_e)]
    ridge_b = [bentuk_ridge(skel_int, x, y, th, D_ridge)
               for (x, y), th in zip(bifurs, theta_b)]

    total_minutiae = len(endings) + len(bifurs)
    lolos_mutu = total_minutiae >= P1_MINUTIAE_MIN
    if not lolos_mutu:
        print(f"    PERINGATAN mutu [P1] Bab I-D: hanya {total_minutiae} "
              f"minutiae (< {P1_MINUTIAE_MIN}). Citra ini TIDAK layak "
              f"dijadikan template pendaftaran.")

    data = {
        'nama_file': os.path.basename(path_bmp),
        'nama': nama,
        'jenis': info['jenis'],
        'kode_subjek': info['kode_subjek'],
        'kode_jari': info['kode_jari'],
        'metode': 'minutiae [P1] Jain dkk. 1997, Proc. IEEE 85(9):1365-1388',
        'parameter': {
            'blok_orientasi_W': P1_BLOK_W,
            'filter_penghalus_D': P1_SMOOTH_D,
            'ambang_certainty_Ts': P1_AMBANG_CERTAINTY,   # [P1] Pers. 2 = T_s
            'ambang_konsistensi_Tc': P1_AMBANG_KONSISTENSI,  # [P1] Gbr. 8 lgk. 5
            'mask_ridge_LxH': f'{P1_MASK_L}x{P1_MASK_H}',
            'ambang_ridge_Tridge': P1_AMBANG_RIDGE,
            'spur_maks': P1_SPUR_MAKS,
            'jembatan_maks': P1_JEMBATAN_MAKS,
            'border_margin': BORDER_MARGIN,
            'inter_ridge_D': round(D_ridge, 2),
            'ridge_sampel': P1_RIDGE_SAMPEL,
        },
        'keterangan_theta': ('arah ridge dalam radian, 0 = ke kanan, '
                             'positif berlawanan arah jarum jam'),
        'endings': [],
        'bifurcations': [],
        'lolos_mutu_pendaftaran': bool(lolos_mutu),
        'ambang_mutu_minutiae': P1_MINUTIAE_MIN,
        'jumlah_ending': len(endings),
        'jumlah_bifurcation': len(bifurs),
        'total_minutiae': len(endings) + len(bifurs),
    }
    for (x, y), th, rg in zip(endings, theta_e, ridge_e):
        data['endings'].append({'x': int(x), 'y': int(y), 'tipe': 'E',
                                'nama_tipe': 'ridge ending',
                                'theta': float(th),
                                'theta_deg': round(math.degrees(th), 2),
                                'ridge': [round(float(v), 4) for v in rg[1]],
                                'ridge_x': [round(float(v), 4) for v in rg[0]]})
    for (x, y), th, rg in zip(bifurs, theta_b, ridge_b):
        data['bifurcations'].append({'x': int(x), 'y': int(y), 'tipe': 'B',
                                     'nama_tipe': 'bifurcation',
                                     'theta': float(th),
                                     'theta_deg': round(math.degrees(th), 2),
                                     'ridge': [round(float(v), 4) for v in rg[1]],
                                     'ridge_x': [round(float(v), 4)
                                                 for v in rg[0]]})

    json_path = os.path.join(FOLDER_MINUTIAE, f'{nama}_minutiae.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    csv_path = os.path.join(FOLDER_MINUTIAE, f'{nama}_minutiae.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['no', 'tipe', 'nama_tipe', 'x', 'y', 'theta_rad', 'theta_deg'])
        no = 0
        for m in data['endings'] + data['bifurcations']:
            no += 1
            w.writerow([no, m['tipe'], m['nama_tipe'], m['x'], m['y'],
                        round(m['theta'], 4), m['theta_deg']])

    img_overlay = gambar_overlay(img, endings, bifurs, skel_int, theta_e, theta_b)
    overlay_path = os.path.join(FOLDER_MINUTIAE, f'{nama}_overlay.bmp')
    cv2.imwrite(overlay_path, img_overlay)

    plot_path = os.path.join(FOLDER_MINUTIAE, f'{nama}_minutiae.png')
    fig = plt.figure(figsize=(15, 4.4))
    fig.suptitle(f'Ekstraksi Minutiae [P1] — {nama}   '
                 f'({len(endings)} ending + {len(bifurs)} bifurcation)',
                 fontsize=11, fontweight='bold')
    for i, (citra, judul, cmap, vmin, vmax) in enumerate([
            (img, 'Asli', 'gray', 0, 255),
            (binary, 'Peta ridge [P1] II-B', 'gray', 0, 255),
            (skeleton, 'Rangka tertipiskan', 'gray', 0, 255),
            (cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB), 'Minutiae',
             None, None, None)]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(citra, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(judul, fontsize=9)
        ax.axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(plot_path, dpi=150)
    if tampilkan:
        print("  >>> Tutup jendela gambar untuk melanjutkan.")
        plt.show()
    plt.close(fig)

    print(f"  JSON    : {json_path}")
    print(f"  CSV     : {csv_path}")
    print(f"  Overlay : {overlay_path}")
    print(f"  Gambar  : {plot_path}")
    return data


# ======================================================================
# PROGRAM UTAMA
# ======================================================================
if __name__ == '__main__':
    print("." * 60)
    print(" EKSTRAKSI CIRI MINUTIAE")
    print("." * 60)

    daftar, keterangan = pilih_citra()
    if not daftar:
        raise SystemExit("Tidak ada citra untuk diproses.")

    print(f"\n  {keterangan}")
    tampilkan = True
    if len(daftar) > 1:
        tampilkan = tanya_ya_tidak(
            "  Tampilkan jendela gambar untuk setiap citra "
            "(tetap disimpan bila tidak)")

    ringkas = []
    for path in daftar:
        hasil = proses_satu(path, tampilkan=tampilkan)
        if hasil:
            ringkas.append(hasil)

    print("\n" + "=" * 60)
    print("  RINGKASAN EKSTRAKSI MINUTIAE")
    print("=" * 60)
    print(f"  {'Berkas':<26}{'Ending':>8}{'Bifur':>8}{'Total':>8}")
    print("  " + "-" * 50)
    for r in ringkas:
        print(f"  {r['nama']:<26}{r['jumlah_ending']:>8}"
              f"{r['jumlah_bifurcation']:>8}{r['total_minutiae']:>8}")
    if ringkas:
        rata = sum(r['total_minutiae'] for r in ringkas) / len(ringkas)
        print("  " + "-" * 50)
        print(f"  Rata-rata minutiae per citra: {rata:.1f}")

    print(f"\n  SELURUH HASIL BERHASIL TERSIMPAN")
    print("  LANJUT MATCHING!")
