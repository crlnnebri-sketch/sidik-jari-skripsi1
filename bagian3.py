import os
import csv
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from config import *


# ----------------------------------------------------------------------
# PENCOCOKAN PARALEL
# ----------------------------------------------------------------------
JUMLAH_PROSES = max(1, (os.cpu_count() or 1))

_CIRI_GALERI_PEKERJA = None


def _siapkan_pekerja(ciri_galeri):
    global _CIRI_GALERI_PEKERJA
    _CIRI_GALERI_PEKERJA = ciri_galeri


def _kerjakan_satu_pasangan(tugas):
    nama_ref, e_uji, b_uji = tugas
    e_ref, b_ref = _CIRI_GALERI_PEKERJA[nama_ref]
    jml, skor, pasangan, tot_uji, tot_ref = cocokkan_minutiae(
        e_uji, b_uji, e_ref, b_ref)
    return nama_ref, jml, skor, len(pasangan), tot_uji, tot_ref


# ----------------------------------------------------------------------
# MEMBACA CIRI MINUTIAE SIDIK JARI
# ----------------------------------------------------------------------
def ambil_minutiae(nama_bmp, path_bmp, dengan_arah=True):

    json_path = cari_json_untuk_bmp(nama_bmp)
    if json_path is not None:
        hasil = muat_minutiae_dari_json(json_path, dengan_arah=dengan_arah)
        if hasil is not None:
            return hasil

    print(f"    (JSON belum ada untuk {nama_bmp}, memproses ulang dari BMP)")
    hasil = ekstraksi_dari_file(path_bmp)
    if hasil is None:
        return None
    endings, bifurs, _, _, _, skel_int = hasil
    if dengan_arah:
        endings = [(x, y, hitung_arah(skel_int, x, y)) for x, y in endings]
        bifurs = [(x, y, hitung_arah(skel_int, x, y)) for x, y in bifurs]
    return endings, bifurs


def muat_ciri(daftar_path, label):

    ciri = {}
    for p in daftar_path:
        nama = nama_dasar(p)
        hasil = ambil_minutiae(os.path.basename(p), p)
        if hasil is None:
            print(f"    [LEWAT] ciri tidak terbaca: {nama}")
            continue
        ciri[nama] = hasil
    print(f"  {len(ciri)} ciri {label} dimuat.")
    return ciri


# ----------------------------------------------------------------------
# PEMILIHAN CITRA UJI
# ----------------------------------------------------------------------
def pilih_citra_uji():

    semua = daftar_identifikasi()
    if not semua:
        print("\n  Belum ada citra IDENTIFIKASI di raw_images/.")
        print("  Ambil citra dulu lalu ekstraksi!")
        return [], ''

    subjek = sorted({subjek_dari(f) for f in semua})
    print("\n" + "=" * 74)
    print("  CITRA UJI MANA YANG MAU DICOCOKKAN?")
    print("=" * 74)
    print(f"  {'Subjek':<10}{'Jumlah':>7}   {'Rentang berkas':<42}Ciri")
    print("  " + "-" * 70)
    for k in subjek:
        milik = daftar_identifikasi(kode_subjek=k)
        siap = sum(1 for f in milik if cari_json_untuk_bmp(f))
        ket = "lengkap" if siap == len(milik) else f"baru {siap}/{len(milik)}"
        rentang = f"{nama_dasar(milik[0])} ... {nama_dasar(milik[-1])}"
        print(f"  {k:<10}{len(milik):>7}   {rentang:<42}{ket}")

    print("\n  Ketik  kode subjek        (contoh: cbs)")
    print("         semua               untuk seluruh subjek")
    print("         nama berkas lengkap untuk satu citra saja "
          "(contoh: fingerprint_cbs03)")

    while True:
        jawab = input("  Pilihan: ").strip().lower()

        if jawab == 'semua':
            return semua, f'seluruh subjek ({len(semua)} citra uji)'

        terpilih = daftar_identifikasi(kode_subjek=jawab)
        if terpilih:
            return terpilih, f'subjek {jawab} ({len(terpilih)} citra uji)'

        satu = [f for f in semua if nama_dasar(f) == jawab]
        if satu:
            return satu, f'satu citra uji ({jawab})'

        print("  Tidak dikenali. Ketik kode subjek, 'semua', "
              "atau nama berkas lengkap.")


# ----------------------------------------------------------------------
# Pencocokan satu citra uji terhadap seluruh galeri
# ----------------------------------------------------------------------
def _kerjakan_satu_pasangan_lokal(tugas, ciri_galeri):
    """Versi satu proses. Dipakai bila multiproses dimatikan atau gagal."""
    nama_ref, e_uji, b_uji = tugas
    e_ref, b_ref = ciri_galeri[nama_ref]
    jml, skor, pasangan, tot_uji, tot_ref = cocokkan_minutiae(
        e_uji, b_uji, e_ref, b_ref)
    return nama_ref, jml, skor, len(pasangan), tot_uji, tot_ref


def cocokkan_satu_uji(nama_uji, ciri_uji, galeri_nama, ciri_galeri,
                      cetak_detail=True, kolam=None):
    """
    Mengembalikan dict berisi tabel perbandingan, kandidat terbaik, ringkasan.
    """
    e_uji, b_uji = ciri_uji
    info_uji = info_berkas(nama_uji)

    if cetak_detail:
        print(f"\n  {'Citra galeri':<24}{'Cocok':>7}{'Total':>7}{'Skor':>9}"
              f"{'Keputusan':>13}{'Pasangan':>11}{'Kategori':>20}")
        print("  " + "-" * 91)

    tugas = [(nama_ref, e_uji, b_uji) for nama_ref in galeri_nama]
    if kolam is not None:
        hasil_cocok = list(kolam.map(_kerjakan_satu_pasangan, tugas,
                                     chunksize=max(1, len(tugas) // (4 * JUMLAH_PROSES))))
    else:
        hasil_cocok = [_kerjakan_satu_pasangan_lokal(t_, ciri_galeri)
                       for t_ in tugas]

    tabel = []
    for nama_ref, jml, skor, n_pasangan, tot_uji, tot_ref in hasil_cocok:
        info_ref = info_berkas(nama_ref)

        cocok = skor >= THRESHOLD_COCOK
        genuine = pasangan_genuine(nama_uji, nama_ref)
        kategori, kode_kat = kategori_hasil(genuine, cocok)

        tabel.append({
            'citra_uji': nama_uji,
            'subjek_uji': info_uji['kode_subjek'],
            'citra_galeri': nama_ref,
            'subjek_galeri': info_ref['kode_subjek'],
            'minutiae_uji': tot_uji,
            'minutiae_galeri': tot_ref,
            'minutiae_cocok': jml,
            'pembagi': min(tot_uji, tot_ref),
            'skor': round(skor, 4),
            'ambang': THRESHOLD_COCOK,
            'keputusan': 'COCOK' if cocok else 'TIDAK COCOK',
            'jenis_pasangan': ('genuine' if genuine else 'impostor')
                              if genuine is not None else 'tidak diketahui',
            'kategori': kategori,
            'kode_kategori': kode_kat,
            'jumlah_pasangan_detail': n_pasangan,
        })

        if cetak_detail:
            print(f"  {nama_ref:<24}{jml:>7}{min(tot_uji, tot_ref):>7}"
                  f"{skor:>8.1f}%"
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
    """Menyimpan tabel_<uji>.csv dan tabel_<uji>.json."""
    nama_uji = h['nama_uji']
    tabel_urut = sorted(h['tabel'], key=lambda x: -x['skor'])

    kolom = ['citra_uji', 'subjek_uji', 'citra_galeri', 'subjek_galeri',
             'minutiae_uji', 'minutiae_galeri', 'minutiae_cocok', 'pembagi',
             'skor', 'ambang', 'keputusan', 'jenis_pasangan',
             'kategori', 'kode_kategori']

    path_csv = os.path.join(FOLDER_HASIL, f'tabel_{nama_uji}.csv')
    with open(path_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction='ignore')
        w.writeheader()
        w.writerows(tabel_urut)

    path_json = os.path.join(FOLDER_HASIL, f'tabel_{nama_uji}.json')
    with open(path_json, 'w') as f:
        json.dump({
            'metode': 'minutiae',
            'citra_uji': nama_uji,
            'subjek_uji': h['info_uji']['kode_subjek'],
            'waktu': datetime.now().isoformat(timespec='seconds'),
            'parameter': {
                'kotak_radius_delta_px': P1_KOTAK_R,
                'kotak_sudut_epsilon_deg': P1_KOTAK_E,
                'toleransi_arah_rho_deg': P1_TOLERANSI_SUDUT,
                'bobot_alpha_beta_gamma': [P1_BOBOT_R, P1_BOBOT_E,
                                           P1_BOBOT_THETA],
                'penalti_Omega': P1_PENALTI,
                'laju_belajar_eta': P1_LAJU_BELAJAR,
                'ambang_ridge_Tr': P1_AMBANG_RIDGE_SELARAS,
                'kandidat_penyelarasan': P1_KANDIDAT_SELARAS,
                'pakai_alignment': PAKAI_ALIGNMENT,
                'threshold_cocok': THRESHOLD_COCOK,
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
    print("  PENCOCOKAN MINUTIAE (identifikasi 1:N)")
    print("=" * 74)

    daftar_uji, keterangan = pilih_citra_uji()
    if not daftar_uji:
        raise SystemExit("Dibatalkan.")

    galeri = daftar_dataset()
    if not galeri:
        raise SystemExit("Belum ada citra DATASET di raw_images/.")

    print(f"\n  Terpilih   : {keterangan}")
    print(f"  Galeri     : {len(galeri)} citra dataset")
    print(f"  Kotak [P1] : delta {P1_KOTAK_R} px, epsilon {P1_KOTAK_E} deg, "
          f"rho {P1_TOLERANSI_SUDUT} deg, Omega {P1_PENALTI:.0f}")
    print(f"  Ambang     : {THRESHOLD_COCOK}")
    print(f"  Penyelarasan: "
          f"{'AKTIF (ridge matching, T_r = ' + str(P1_AMBANG_RIDGE_SELARAS) + ')' if PAKAI_ALIGNMENT else 'TIDAK AKTIF'}")
    print(f"  Perbandingan: {len(daftar_uji)} x {len(galeri)} = "
          f"{len(daftar_uji) * len(galeri)}")

    print("\n  Memuat ciri minutiae...")
    ciri_galeri = muat_ciri(galeri, 'galeri')
    ciri_uji = muat_ciri(daftar_uji, 'citra uji')
    galeri_nama = [n for n in (nama_dasar(p) for p in galeri) if n in ciri_galeri]
    if not galeri_nama or not ciri_uji:
        raise SystemExit("Ciri belum lengkap. Jalankan EKSTRAKSI terlebih dahulu.")

    cetak_detail = True
    if len(ciri_uji) > 1:
        cetak_detail = tanya_ya_tidak(
            "\n  Tampilkan tabel rinci seluruh galeri untuk tiap citra uji "
            "(tetap disimpan bila tidak)")

    pakai_paralel = JUMLAH_PROSES > 1 and len(galeri_nama) > 1
    if pakai_paralel:
        print(f"\n  Pencocokan paralel: {JUMLAH_PROSES} proses "
              f"({len(daftar_uji)} x {len(galeri_nama)} perbandingan)")
        kolam_ctx = ProcessPoolExecutor(max_workers=JUMLAH_PROSES,
                                        initializer=_siapkan_pekerja,
                                        initargs=(ciri_galeri,))
    else:
        print("\n  Pencocokan satu proses (hanya satu inti tersedia)")
        kolam_ctx = None

    semua_hasil = []
    for path_uji in daftar_uji:
        nama_uji = nama_dasar(path_uji)
        if nama_uji not in ciri_uji:
            continue

        e_uji, b_uji = ciri_uji[nama_uji]
        print("\n" + "-" * 74)
        print(f"  {nama_uji}   (subjek {subjek_dari(nama_uji)}, "
              f"{len(e_uji)}E + {len(b_uji)}B = "
              f"{len(e_uji) + len(b_uji)} minutiae)")
        print("-" * 74)

        h = cocokkan_satu_uji(nama_uji, ciri_uji[nama_uji], galeri_nama,
                              ciri_galeri, cetak_detail=cetak_detail,
                              kolam=kolam_ctx)
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

    if kolam_ctx is not None:
        kolam_ctx.shutdown(wait=True)

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
            'ambang': THRESHOLD_COCOK,
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
          f"{THRESHOLD_COCOK}%:")
    print(f"    genuine  {n_gen:>5}  ->  benar cocok {total_kat['TM']}, "
          f"salah tidak cocok {total_kat['FNM']}")
    print(f"    impostor {n_imp:>5}  ->  salah cocok {total_kat['FM']}, "
          f"benar tidak cocok {total_kat['TN']}")
    if n_gen and n_imp:
        print(f"    FNMR = {total_kat['FNM']}/{n_gen} = "
              f"{total_kat['FNM'] / n_gen * 100:.2f}%   |   "
              f"FMR = {total_kat['FM']}/{n_imp} = "
              f"{total_kat['FM'] / n_imp * 100:.2f}%")


    path_rekap = os.path.join(FOLDER_HASIL, 'rekap_matching.csv')
    with open(path_rekap, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rekap[0].keys()))
        w.writeheader()
        w.writerows(rekap)

# ------------------------------------------------------------------
# Sebelum ini berkas tersebut TIDAK PERNAH DITULIS, sehingga pilihan 1
# ------------------------------------------------------------------
    terakhir = rekap[-1]
    path_terakhir = os.path.join(FOLDER_HASIL, 'hasil_terakhir.json')
    with open(path_terakhir, 'w') as f:
        json.dump({
            'metode': 'minutiae',
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
