import os
import glob
import json
import math

import matplotlib.pyplot as plt

from config import *

SKOR_THRESHOLD = SKOR_THRESHOLD_GABOR
FOLDER_BMP = OUTPUT_DIR

_simpanan = {}


# ======================================================================
# PIPELINE SATU CITRA 
# ======================================================================
def proses_visualisasi(nama):
    if nama in _simpanan:
        return _simpanan[nama]

    path_bmp = os.path.join(FOLDER_BMP, nama + '.bmp')
    if not os.path.exists(path_bmp):
        print(f"  [LEWAT] berkas tidak ditemukan: {path_bmp}")
        return None

    gray = cv2.imread(path_bmp, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  [LEWAT] gagal membaca: {nama}")
        return None

    vektor, d = fingercode_dari_citra(gray)
    responses = [np.abs(r) for r in d['responses']]
    combined = np.mean(np.stack(responses, axis=0), axis=0)

    heatmap = np.zeros((P2_JUMLAH_PITA, P2_SEKTOR_PITA))
    for i in range(P2_JUMLAH_SEKTOR):
        heatmap[i // P2_SEKTOR_PITA, i % P2_SEKTOR_PITA] = float(
            np.mean([vektor[o * P2_JUMLAH_SEKTOR + i]
                     for o in range(GABOR_ORIENTASI)]))

    hasil = {"nama": nama, "roi": d['roi'].astype(np.uint8), "ternormalisasi": d['norm'],
             "combined": combined, "heatmap": heatmap,
             "peta": d['peta'], "pusat": d['pusat'], "vektor": vektor}
    _simpanan[nama] = hasil
    return hasil


def tabel_sel_beda(kontrib, top_n=5):
    """Sektor dengan sumbangan jarak terbesar. (pita, sektor, x, y, nilai)."""
    R = P2_JARI_LUAR
    daftar = []
    for p in range(P2_JUMLAH_PITA):
        for sgm in range(P2_SEKTOR_PITA):
            rm = P2_JARI_DALAM + (p + 0.5) * P2_LEBAR_PITA
            am = (sgm + 0.5) * (2 * math.pi / P2_SEKTOR_PITA)
            daftar.append((p, sgm,
                           int(R + rm * math.cos(am)),
                           int(R + rm * math.sin(am)),
                           float(kontrib[p, sgm])))
    daftar.sort(key=lambda x: -x[4])
    return daftar[:top_n]


# ======================================================================
# GAMBAR
# ======================================================================
def _gambar_grid_koordinat(ax, roi, top_cells, judul):
    """Tesselasi melingkar [P2] Gbr. 4 digambar di atas ROI."""
    R = P2_JARI_LUAR
    ax.imshow(roi, cmap='gray', vmin=0, vmax=255)
    for p in range(P2_JUMLAH_PITA + 1):
        rr = P2_JARI_DALAM + p * P2_LEBAR_PITA
        ax.add_patch(plt.Circle((R, R), rr, fill=False,
                                edgecolor='cyan', lw=0.5, alpha=0.7))
    ax.add_patch(plt.Circle((R, R), P2_JARI_DALAM, fill=False,
                            edgecolor='cyan', lw=0.8, alpha=0.9))
    for sgm in range(P2_SEKTOR_PITA):
        a = sgm * (2 * math.pi / P2_SEKTOR_PITA)
        ax.plot([R + P2_JARI_DALAM * math.cos(a), R + R * math.cos(a)],
                [R + P2_JARI_DALAM * math.sin(a), R + R * math.sin(a)],
                color='cyan', lw=0.4, alpha=0.6)
    ax.plot(R, R, 'y+', ms=9, mew=1.6)
    for p, sgm, cx, cy, k in top_cells:
        ax.plot(cx, cy, 'r+', ms=8, mew=1.6)
        ax.text(cx + 2, cy - 2, f"{p},{sgm}", color='red', fontsize=5)
    ax.set_title(judul, fontsize=7.5, fontweight='bold')
    ax.set_xticks(range(0, 2 * R + 1, max(20, R // 2)))
    ax.set_yticks(range(0, 2 * R + 1, max(20, R // 2)))
    ax.tick_params(labelsize=5)


_JUDUL_TAHAP = [
    "1. ROI di sekitar titik acuan\n([P2] Bab II-A)",
    "2. Normalisasi tiap sektor\n(M0=V0=100, [P2] Pers. 16)",
    "3. Respons Gabor\n(rata-rata 8 orientasi)",
    f"4. Peta ciri AAD\n({P2_JUMLAH_PITA} pita x {P2_SEKTOR_PITA} sektor)",
    "5. Tesselasi + sektor paling beda\n(tanda merah)",
]


def visualisasi_matching(query_nama, query_data, match_nama, match_data,
                         skor, jarak, status, kontrib, top_cells,
                         path_simpan=None, tampilkan=False):
    """Tata letak 2 baris x 6 kolom."""
    fig, axes = plt.subplots(2, 6, figsize=(21, 7.6), facecolor='white')
    fig.subplots_adjust(hspace=0.30, wspace=0.30, top=0.84, bottom=0.05,
                        left=0.025, right=0.985)

    status_text = "COCOK" if status else "TIDAK COCOK"
    warna = '#2e7d32' if status else '#c62828'

    fig.suptitle(
        f"Matching FingerCode [P2]: {skor:.1f}%  ->  {status_text}\n"
        f"Ambang {SKOR_THRESHOLD:.1f}%  |  jarak Euclidean {jarak:.2f}  |  "
        f"{GABOR_ORIENTASI} orientasi, {P2_JUMLAH_SEKTOR} sektor",
        fontsize=13, fontweight='bold', color=warna)

    _vmin_peta = min(query_data["heatmap"].min(),
                     match_data["heatmap"].min())
    _vmax_peta = max(query_data["heatmap"].max(),
                     match_data["heatmap"].max())

    _vmax_resp = max(float(query_data["combined"].max()),
                     float(match_data["combined"].max()))

    for row, data, label, berkas in [
            (0, query_data, "CITRA UJI", query_nama),
            (1, match_data, "PEMBANDING TERBAIK", match_nama)]:

        axes[row, 0].imshow(data["roi"], cmap='gray', vmin=0, vmax=255)
        axes[row, 0].set_title(f"{label}\n{_JUDUL_TAHAP[0]}",
                               fontsize=7.5, fontweight='bold')
        axes[row, 0].axis('off')
        axes[row, 0].text(0.5, -0.06, f"({berkas}.bmp)",
                          transform=axes[row, 0].transAxes,
                          ha='center', va='top', fontsize=6.5, style='italic')

        axes[row, 1].imshow(data["ternormalisasi"], cmap='gray', vmin=0, vmax=255)
        axes[row, 1].set_title(_JUDUL_TAHAP[1], fontsize=7.5, fontweight='bold')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(data["combined"], cmap='hot',
                            vmin=0, vmax=_vmax_resp)
        axes[row, 2].set_title(_JUDUL_TAHAP[2], fontsize=7.5, fontweight='bold')
        axes[row, 2].axis('off')

        im = axes[row, 3].imshow(data["heatmap"], cmap='YlOrRd',
                                 interpolation='nearest',
                                 vmin=_vmin_peta, vmax=_vmax_peta)
        axes[row, 3].set_title(_JUDUL_TAHAP[3], fontsize=7.5, fontweight='bold')
        axes[row, 3].set_xticks(range(0, P2_SEKTOR_PITA, 2))
        axes[row, 3].set_yticks(range(P2_JUMLAH_PITA))
        axes[row, 3].tick_params(labelsize=4.5)
        cb = fig.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)

        _gambar_grid_koordinat(axes[row, 4], data["roi"], top_cells,
                               _JUDUL_TAHAP[4])

    imk = axes[0, 5].imshow(kontrib, cmap='Reds', interpolation='nearest')
    axes[0, 5].set_title("6. Peta kontribusi jarak\n"
                         "(merah tua = paling berbeda)",
                         fontsize=7.5, fontweight='bold')
    axes[0, 5].set_xticks(range(0, P2_SEKTOR_PITA, 2))
    axes[0, 5].set_yticks(range(P2_JUMLAH_PITA))
    axes[0, 5].tick_params(labelsize=4.5)
    cbk = fig.colorbar(imk, ax=axes[0, 5], fraction=0.046, pad=0.04)
    cbk.ax.tick_params(labelsize=5)

    ax_info = axes[1, 5]
    ax_info.axis('off')
    baris_teks = [
        "HASIL PENCOCOKAN", "=" * 30,
        f"Skor kemiripan : {skor:5.1f} %",
        f"Ambang         : {SKOR_THRESHOLD:5.1f} %",
        f"Jarak Euclidean: {jarak:8.2f}",
        f"Keputusan      : {status_text}", "",
        "SEKTOR PALING BERBEDA", "-" * 30,
        "(pita,sektor)  (x,y)   kontrib",
    ]
    for r, c, cx, cy, k in top_cells:
        baris_teks.append(f"  ({r},{c})    ({cx:>3},{cy:>3})  {k:.4f}")
    ax_info.text(0.0, 0.97, "\n".join(baris_teks), va='top', ha='left',
                 family='monospace', fontsize=7.2,
                 transform=ax_info.transAxes)
    ax_info.add_patch(plt.Rectangle((-0.04, -0.02), 1.08, 1.03,
                                    transform=ax_info.transAxes, fill=False,
                                    edgecolor=warna, lw=2, clip_on=False))

    if path_simpan:
        os.makedirs(os.path.dirname(path_simpan), exist_ok=True)
        fig.savefig(path_simpan, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"    Gambar : {path_simpan}")
    if tampilkan:
        plt.show()
    plt.close(fig)


# ======================================================================
# PEMROSESAN SATU PASANGAN
# ======================================================================
def proses_pasangan(nama_uji, nama_ref, templates, tampilkan=False, rinci=True):
    if nama_uji not in templates or nama_ref not in templates:
        print(f"  [LEWAT] ciri Gabor belum ada untuk {nama_uji} atau {nama_ref}")
        return None

    v_uji = templates[nama_uji]['vektor']
    v_ref = templates[nama_ref]['vektor']

    skor = skor_kemiripan(v_uji, v_ref)
    jarak = jarak_ternormalisasi(v_uji, v_ref)
    status = skor >= SKOR_THRESHOLD
    kontrib = peta_kontribusi(v_uji, v_ref)
    top_cells = tabel_sel_beda(kontrib, top_n=5)

    if rinci:
        print("\n  " + "=" * 60)
        print("  PARAMETER MATCHING (GABOR / TEXTURE-BASED)")
        print("  " + "=" * 60)
        print(f"    Citra uji      : {nama_uji}")
        print(f"    Paling cocok   : {nama_ref}")
        print(f"    Skor kemiripan : {skor:.1f} %   "
              f"(ambang {SKOR_THRESHOLD:.1f} %)")
        print(f"    Jarak Euclidean: {jarak:.2f}")
        print(f"    Keputusan      : {'COCOK' if status else 'TIDAK COCOK'}")
        print("  " + "-" * 60)
        print("    SEKTOR PALING BERBEDA")
        print("    Catatan: ini titik tengah SEKTOR tesselasi, bukan minutiae.")
        print(f"    {'baris':>5}{'kolom':>6}{'x_pusat':>9}{'y_pusat':>9}"
              f"{'kontribusi':>12}")
        for r, c, cx, cy, k in top_cells:
            print(f"    {r:>5}{c:>6}{cx:>9}{cy:>9}{k:>12.4f}")
        print("  " + "=" * 60)
    else:
        print(f"  {nama_uji:<22} vs {nama_ref:<20} {skor:>6.1f}%  "
              f"{'COCOK' if status else 'TIDAK COCOK'}")

    d1 = proses_visualisasi(nama_uji)
    d2 = proses_visualisasi(nama_ref)
    if d1 is None or d2 is None:
        return None

    path_simpan = os.path.join(FOLDER_MATCHING_GABOR,
                               f"gabor_matching_{nama_uji}.png")
    visualisasi_matching(nama_uji, d1, nama_ref, d2, skor, jarak, status,
                         kontrib, top_cells, path_simpan=path_simpan,
                         tampilkan=tampilkan)

    return {'citra_uji': nama_uji, 'citra_galeri': nama_ref,
            'skor': round(skor, 4), 'jarak': round(jarak, 6),
            'status': 'COCOK' if status else 'TIDAK COCOK'}


# ======================================================================
# SUMBER PASANGAN YANG AKAN DIGAMBAR
# ======================================================================
def dari_hasil_terakhir():
    """Pasangan citra uji terakhir yang dicocokkan."""
    path = os.path.join(FOLDER_MATCHING_GABOR, 'hasil_terakhir.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                h = json.load(f)
            return [(nama_dasar(h['citra_baru']), nama_dasar(h['paling_cocok']))]
        except (json.JSONDecodeError, KeyError, IOError, OSError):
            print("  hasil_terakhir.json rusak, memakai tabel terbaru.")

    tabel = glob.glob(os.path.join(FOLDER_MATCHING_GABOR, 'tabel_*.json'))
    if not tabel:
        print("  Belum ada hasil pencocokan sama sekali.")
        print("  Jalankan bagian3_gabor.py terlebih dahulu.")
        return []
    baru = max(tabel, key=os.path.getmtime)
    try:
        with open(baru) as f:
            d = json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        print(f"  {os.path.basename(baru)} tidak terbaca.")
        return []
    if not d.get('citra_uji') or not d.get('paling_cocok'):
        print(f"  {os.path.basename(baru)} tidak memuat pasangan yang sah.")
        return []
    print(f"  hasil_terakhir.json belum ada; memakai berkas terbaru: "
          f"{os.path.basename(baru)}")
    return [(nama_dasar(d['citra_uji']), nama_dasar(d['paling_cocok']))]


def dari_semua_tabel():
    pasangan = []
    for path in sorted(glob.glob(os.path.join(FOLDER_MATCHING_GABOR,
                                              'tabel_*.json'))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        if 'citra_uji' in d and d.get('paling_cocok'):
            pasangan.append((d['citra_uji'], d['paling_cocok']))
    if not pasangan:
        print("  Belum ada berkas tabel_*.json di hasil_matching_gabor/.")
        print("  Jalankan MATCHING (pilih 'semua') terlebih dahulu.")
    return pasangan


# ======================================================================
# PROGRAM UTAMA
# ======================================================================
def main():
    print("=" * 74)
    print("  TEKSUR (GABOR) - VISUALISASI HASIL PENCOCOKAN")
    print("=" * 74)

    templates = {t['nama_dasar']: t for t in muat_template_gabor()}
    if not templates:
        print("  Belum ada berkas ciri di gabor_features/. "
              "Jalankan EKSTRAKSI dulu.")
        return

    tersedia = len(glob.glob(os.path.join(FOLDER_MATCHING_GABOR,
                                          'tabel_*.json')))
    print("\n  Mau menggambar yang mana?")
    print("   1 = Hasil terakhir dari bagian3_gabor.py  (satu citra uji saja)")
    print(f"   2 = Semua citra uji yang sudah dicocokkan  "
          f"({tersedia} berkas tersedia)")
    pilih = tanya_pilihan("  Pilihan Anda", ['1', '2'])

    daftar = dari_hasil_terakhir() if pilih == '1' else dari_semua_tabel()
    if not daftar:
        return

    rinci = True
    tampilkan = True
    if len(daftar) > 1:
        print(f"\n  {len(daftar)} citra uji akan digambar.")
        rinci = tanya_ya_tidak("  Cetak parameter matching untuk "
                               "tiap citra uji")
        tampilkan = tanya_ya_tidak("  Tampilkan jendela gambar tiap citra uji "
                                   "(tetap disimpan bila tidak)")

    hasil = []
    for i, (nama_uji, nama_ref) in enumerate(daftar, 1):
        print("\n" + "-" * 74)
        print(f"  [{i}/{len(daftar)}]  {nama_uji}  vs  {nama_ref}")
        print("-" * 74)
        r = proses_pasangan(nama_uji, nama_ref, templates,
                            tampilkan=tampilkan, rinci=rinci)
        if r:
            hasil.append(r)

    if not hasil:
        print("Tidak ada gambar yang berhasil dibuat.")
        return

    print("\n" + "=" * 74)
    print(f"  {len(hasil)} citra uji selesai divisualisasikan.")
    print("=" * 74)
    if len(hasil) > 1:
        print(f"  {'Citra uji':<22}{'Paling cocok':<20}{'Skor':>8}"
              f"{'Keputusan':>14}")
        print("  " + "-" * 64)
        for r in hasil:
            print(f"  {r['citra_uji']:<22}{r['citra_galeri']:<20}"
                  f"{r['skor']:>7.1f}%{r['status']:>14}")
    print(f"\n  Gambar : {FOLDER_MATCHING_GABOR}/gabor_matching_<nama_uji>.png")
    print("\n  Langkah berikutnya: evaluasi.py (EER, FMR, FNMR).")


if __name__ == "__main__":
    main()
