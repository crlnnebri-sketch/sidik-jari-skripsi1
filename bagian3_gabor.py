import os
import csv
import json
from datetime import datetime

from config import *

SKOR_THRESHOLD = SKOR_THRESHOLD_GABOR


# ----------------------------------------------------------------------
# Pemilihan citra uji
# ----------------------------------------------------------------------
def pilih_citra_uji(templates):
    """Mengembalikan (daftar_template_uji, keterangan)."""
    uji = sorted([t for t in templates if is_identifikasi(t['nama_dasar'])],
                 key=lambda t: t['nama_dasar'])
    if not uji:
        print("\n  Belum ada ciri Gabor untuk citra IDENTIFIKASI.")
        print("  Jalankan bagian1.py (menu 2) lalu bagian2_gabor.py.")
        return [], ''

    subjek = sorted({subjek_dari(t['nama_dasar']) for t in uji})
    print("\n" + "=" * 74)
    print("  CITRA UJI MANA YANG MAU DICOCOKKAN?")
    print("=" * 74)
    print(f"  {'Subjek':<10}{'Jumlah':>7}   Rentang berkas")
    print("  " + "-" * 62)
    for k in subjek:
        milik = [t for t in uji
                 if kode_banding(subjek_dari(t['nama_dasar'])) == kode_banding(k)]
        print(f"  {k:<10}{len(milik):>7}   {milik[0]['nama_dasar']} ... "
              f"{milik[-1]['nama_dasar']}")

    print("\n  Ketik  kode subjek        (contoh: cbs)")
    print("         semua               untuk seluruh subjek")
    print("         nama berkas lengkap untuk satu citra saja "
          "(contoh: fingerprint_cbs03)")

    while True:
        jawab = input("  Pilihan: ").strip().lower()

        if jawab == 'semua':
            return uji, f'seluruh subjek ({len(uji)} citra uji)'

        kb = kode_banding(jawab)
        milik = [t for t in uji
                 if kode_banding(subjek_dari(t['nama_dasar'])) == kb]
        if milik and re.fullmatch(r'[a-z]{2,6}', jawab):
            return milik, f'subjek {jawab} ({len(milik)} citra uji)'

        satu = [t for t in uji if t['nama_dasar'] == jawab]
        if satu:
            return satu, f'satu citra uji ({jawab})'

        print("  Tidak dikenali. Ketik kode subjek, 'semua', "
              "atau nama berkas lengkap.")


# ----------------------------------------------------------------------
# Pencocokan satu citra uji terhadap seluruh galeri
# ----------------------------------------------------------------------
def cocokkan_satu_uji(tmpl_uji, galeri, cetak_detail=True):
    nama_uji = tmpl_uji['nama_dasar']
    info_uji = info_berkas(nama_uji)

    if cetak_detail:
        print(f"\n  {'Citra galeri':<24}{'Jarak':>9}{'Skor':>9}"
              f"{'Keputusan':>13}{'Pasangan':>11}{'Kategori':>20}")
        print("  " + "-" * 86)

    tabel = []
    for t in galeri:
        nama_ref = t['nama_dasar']
        info_ref = info_berkas(nama_ref)

        d = jarak_ternormalisasi(tmpl_uji['vektor'], t['vektor'])
        skor = skor_kemiripan(tmpl_uji['vektor'], t['vektor'])
        cocok = skor >= SKOR_THRESHOLD
        genuine = pasangan_genuine(nama_uji, nama_ref)
        kategori, kode_kat = kategori_hasil(genuine, cocok)

        tabel.append({
            'citra_uji': nama_uji,
            'subjek_uji': info_uji['kode_subjek'],
            'citra_galeri': nama_ref,
            'subjek_galeri': info_ref['kode_subjek'],
            'dimensi_vektor': int(len(tmpl_uji['vektor'])),
            'jarak_ternormalisasi': round(d, 6),
            'skor': round(skor, 4),
            'ambang': SKOR_THRESHOLD,
            'keputusan': 'COCOK' if cocok else 'TIDAK COCOK',
            'jenis_pasangan': ('genuine' if genuine else 'impostor')
                              if genuine is not None else 'tidak diketahui',
            'kategori': kategori,
            'kode_kategori': kode_kat,
        })

        if cetak_detail:
            print(f"  {nama_ref:<24}{d:>9.4f}{skor:>8.1f}%"
                  f"{('COCOK' if cocok else 'TIDAK COCOK'):>13}"
                  f"{('genuine' if genuine else 'impostor'):>11}{kategori:>20}")

    if not tabel:
        return None

    terbaik = max(tabel, key=lambda r: r['skor'])
    hitung = {'TM': 0, 'FNM': 0, 'FM': 0, 'TN': 0}
    for r in tabel:
        if r['kode_kategori'] in hitung:
            hitung[r['kode_kategori']] += 1

    return {'nama_uji': nama_uji, 'info_uji': info_uji, 'tabel': tabel,
            'terbaik': terbaik, 'hitung': hitung,
            'rank1_benar': terbaik['jenis_pasangan'] == 'genuine'}


def simpan_hasil_satu_uji(h):
    nama_uji = h['nama_uji']
    tabel_urut = sorted(h['tabel'], key=lambda x: -x['skor'])

    kolom = ['citra_uji', 'subjek_uji', 'citra_galeri', 'subjek_galeri',
             'dimensi_vektor', 'jarak_ternormalisasi', 'skor', 'ambang',
             'keputusan', 'jenis_pasangan', 'kategori', 'kode_kategori']

    path_csv = os.path.join(FOLDER_MATCHING_GABOR, f'tabel_{nama_uji}.csv')
    with open(path_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction='ignore')
        w.writeheader()
        w.writerows(tabel_urut)

    path_json = os.path.join(FOLDER_MATCHING_GABOR, f'tabel_{nama_uji}.json')
    with open(path_json, 'w') as f:
        json.dump({
            'metode': 'FingerCode [P2] Jain dkk. 2000',
            'citra_uji': nama_uji,
            'subjek_uji': h['info_uji']['kode_subjek'],
            'waktu': datetime.now().isoformat(timespec='seconds'),
            'parameter': {
                'orientasi': GABOR_ORIENTASI,
                'grid': GABOR_GRID,
                'frekuensi': GABOR_FREKUENSI,
                'dimensi': GABOR_DIMENSI,
                'threshold_cocok_persen': SKOR_THRESHOLD,
            },
            'jumlah_galeri': len(h['tabel']),
            'ringkasan_kategori': h['hitung'],
            'paling_cocok': h['terbaik']['citra_galeri'],
            'skor_tertinggi': h['terbaik']['skor'],
            'status': h['terbaik']['keputusan'],
            'perbandingan': tabel_urut,
        }, f, indent=2)

    return path_csv, path_json


# ----------------------------------------------------------------------
# Program utama
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 74)
    print("  BAGIAN 3 (TEKSTUR) - PENCOCOKAN FingerCode [P2] (identifikasi 1:N)")
    print("=" * 74)

    templates = muat_template_gabor()
    if not templates:
        raise SystemExit("Belum ada berkas ciri di gabor_features/. "
                         "Jalankan bagian2_gabor.py dulu.")
    print(f"  {len(templates)} berkas ciri dimuat dari {FOLDER_GABOR_FITUR}")

    daftar_uji, keterangan = pilih_citra_uji(templates)
    if not daftar_uji:
        raise SystemExit("Dibatalkan.")

    galeri = sorted([t for t in templates if is_dataset(t['nama_dasar'])],
                    key=lambda t: t['nama_dasar'])
    if not galeri:
        raise SystemExit("Belum ada ciri Gabor citra DATASET.")

    print(f"\n  Terpilih    : {keterangan}")
    print(f"  Galeri      : {len(galeri)} citra dataset")
    print(f"  Dimensi     : {GABOR_DIMENSI} nilai "
          f"({GABOR_ORIENTASI} orientasi x {GABOR_GRID}x{GABOR_GRID} sel x 2)")
    print(f"  Ambang      : {SKOR_THRESHOLD}% skor kemiripan")
    print(f"  Perbandingan: {len(daftar_uji)} x {len(galeri)} = "
          f"{len(daftar_uji) * len(galeri)}")

    cetak_detail = True
    if len(daftar_uji) > 1:
        cetak_detail = tanya_ya_tidak(
            "\n  Tampilkan tabel rinci seluruh galeri untuk tiap citra uji "
            "(tetap disimpan bila tidak)")

    semua_hasil = []
    for tmpl in daftar_uji:
        print("\n" + "-" * 74)
        print(f"  {tmpl['nama_dasar']}   "
              f"(subjek {subjek_dari(tmpl['nama_dasar'])})")
        print("-" * 74)

        h = cocokkan_satu_uji(tmpl, galeri, cetak_detail=cetak_detail)
        if h is None:
            continue

        t = h['terbaik']
        print(f"  -> paling cocok: {t['citra_galeri']:<22} "
              f"skor {t['skor']:>5.1f}%   {t['keputusan']:<12} "
              f"rank-1 {'BENAR' if h['rank1_benar'] else 'SALAH'}")

        if cetak_detail:
            urut = sorted(h['tabel'], key=lambda r: -r['skor'])[:5]
            print("\n  Lima kandidat teratas:")
            print(f"  {'No':<4}{'Citra galeri':<24}{'Skor':>9}{'Keputusan':>14}")
            for i, r in enumerate(urut, 1):
                print(f"  {i:<4}{r['citra_galeri']:<24}{r['skor']:>8.1f}%"
                      f"{r['keputusan']:>14}")
            k = h['hitung']
            print(f"\n  Kategori: genuine {k['TM'] + k['FNM']} "
                  f"(benar cocok {k['TM']}, salah tidak cocok {k['FNM']}) | "
                  f"impostor {k['FM'] + k['TN']} "
                  f"(salah cocok {k['FM']}, benar tidak cocok {k['TN']})")

        simpan_hasil_satu_uji(h)
        semua_hasil.append(h)

    if not semua_hasil:
        raise SystemExit("Tidak ada pencocokan yang berhasil dihitung.")

    print("\n" + "=" * 78)
    print("  REKAP HASIL IDENTIFIKASI SELURUH CITRA UJI")
    print("=" * 78)
    print(f"  {'Citra uji':<22}{'Paling cocok':<20}{'Skor':>8}"
          f"{'Keputusan':>14}{'Rank-1':>9}")
    print("  " + "-" * 74)

    rekap = []
    total_kat = {'TM': 0, 'FNM': 0, 'FM': 0, 'TN': 0}
    for h in semua_hasil:
        t = h['terbaik']
        for kode in total_kat:
            total_kat[kode] += h['hitung'][kode]
        rekap.append({
            'citra_uji': h['nama_uji'],
            'subjek_uji': h['info_uji']['kode_subjek'],
            'paling_cocok': t['citra_galeri'],
            'subjek_galeri': t['subjek_galeri'],
            'skor': t['skor'],
            'ambang': SKOR_THRESHOLD,
            'keputusan': t['keputusan'],
            'rank1': 'BENAR' if h['rank1_benar'] else 'SALAH',
            'TM': h['hitung']['TM'], 'FNM': h['hitung']['FNM'],
            'FM': h['hitung']['FM'], 'TN': h['hitung']['TN'],
        })
        print(f"  {h['nama_uji']:<22}{t['citra_galeri']:<20}{t['skor']:>7.1f}%"
              f"{t['keputusan']:>14}"
              f"{('BENAR' if h['rank1_benar'] else 'SALAH'):>9}")

    benar = sum(1 for r in rekap if r['rank1'] == 'BENAR')
    n_gen = total_kat['TM'] + total_kat['FNM']
    n_imp = total_kat['FM'] + total_kat['TN']

    print("  " + "-" * 74)
    print(f"  Akurasi rank-1 : {benar}/{len(rekap)} = "
          f"{benar / len(rekap) * 100:.2f}%")
    print(f"\n  Kategori seluruh {n_gen + n_imp} perbandingan pada ambang "
          f"{SKOR_THRESHOLD}%:")
    print(f"    genuine  {n_gen:>5}  ->  benar cocok {total_kat['TM']}, "
          f"salah tidak cocok {total_kat['FNM']}")
    print(f"    impostor {n_imp:>5}  ->  salah cocok {total_kat['FM']}, "
          f"benar tidak cocok {total_kat['TN']}")
    if n_gen and n_imp:
        print(f"    FNMR = {total_kat['FNM']}/{n_gen} = "
              f"{total_kat['FNM'] / n_gen * 100:.2f}%   |   "
              f"FMR = {total_kat['FM']}/{n_imp} = "
              f"{total_kat['FM'] / n_imp * 100:.2f}%")
    print("\n Hasil perhitungan dari citra uji yang baru saja diproses telah selesai!")


    path_rekap = os.path.join(FOLDER_MATCHING_GABOR, 'rekap_matching.csv')
    with open(path_rekap, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rekap[0].keys()))
        w.writeheader()
        w.writerows(rekap)

# ------------------------------------------------------------------
# BELUM ADA BERKAS, PERLU JALANKAN ULANG PENGAMBILAN CITRA
# ------------------------------------------------------------------
    terakhir = rekap[-1]
    path_terakhir = os.path.join(FOLDER_MATCHING_GABOR, 'hasil_terakhir.json')
    with open(path_terakhir, 'w') as f:
        json.dump({
            'metode': 'gabor',
            'waktu': datetime.now().isoformat(timespec='seconds'),
            'citra_baru': terakhir['citra_uji'],
            'paling_cocok': terakhir['paling_cocok'],
            'skor': terakhir['skor'],
            'keputusan': terakhir['keputusan'],
            'rank1': terakhir['rank1'],
            'jumlah_citra_uji_sesi_ini': len(rekap),
            'seluruh_citra_uji_sesi_ini': [r['citra_uji'] for r in rekap],
        }, f, indent=2)
    print(f"  Catatan hasil terakhir : {path_terakhir}")


    print("\n  LANJUT VISUALISASI! ")
