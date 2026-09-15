import os
import glob
import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from config import *
from config import _rapikan          
from bagian3 import ambil_minutiae

GAMBAR_GARIS_PASANGAN = True
GAMBAR_PENYELARASAN   = False  


_simpanan = {}


def pipeline_citra(nama):

    if nama in _simpanan:
        return _simpanan[nama]

    path = os.path.join(OUTPUT_DIR, nama + '.bmp')
    if not os.path.exists(path):
        print(f"  [LEWAT] berkas tidak ditemukan: {path}")
        return None

    data = ekstraksi_dari_file(path)
    if data is None:
        print(f"  [LEWAT] gagal membaca: {nama}")
        return None
    e_xy, b_xy, img, binr, skel, si = data

    ciri = ambil_minutiae(nama + '.bmp', path)
    if ciri is None:
        ciri = ([(x, y, hitung_arah(si, x, y)) for x, y in e_xy],
                [(x, y, hitung_arah(si, x, y)) for x, y in b_xy])

    hasil = {'nama': nama, 'img': img, 'bin': binr, 'skel': skel, 'si': si,
             'e': ciri[0], 'b': ciri[1]}
    _simpanan[nama] = hasil
    return hasil


def matching_detail(e1, b1, e2, b2):
    """
    Alias pendek untuk cocokkan_minutiae(), dipakai di beberapa tempat
    pada berkas ini agar barisnya tidak terlalu panjang.
    """
    return cocokkan_minutiae(e1, b1, e2, b2)


def _xy(daftar):
    return [(m[0], m[1]) for m in daftar]


def _theta(daftar):
    return [m[2] for m in daftar] if daftar and len(daftar[0]) > 2 else None


# ======================================================================
# CETAK DETAIL KE TERMINAL
# ======================================================================
def cetak_detail(d1, d2, jml_cocok, skor, pasangan, tot1, tot2, status,
                 align=None):
    print("\n" + "=" * 78)
    print("  DETAIL PENCOCOKAN")
    print("=" * 78)
    print("\n  Parameter [P1] yang digunakan (hlm. 1380):")
    print(f"    delta   kotak radius              : +-{P1_KOTAK_R} piksel")
    print(f"    epsilon kotak sudut radial        : +-{P1_KOTAK_E} derajat")
    print(f"    rho     toleransi arah minutia    : {P1_TOLERANSI_SUDUT} derajat")
    print(f"    alpha, beta, gamma                : {P1_BOBOT_R}, {P1_BOBOT_E}, {P1_BOBOT_THETA}")
    print(f"    Omega   penalti tak berpasangan   : {P1_PENALTI:.0f}")
    print(f"    eta     laju belajar kotak        : {P1_LAJU_BELAJAR}")
    print(f"    T_r     ambang kemiripan ridge    : {P1_AMBANG_RIDGE_SELARAS}")
    print(f"    Ambang keputusan (THRESHOLD_COCOK): {THRESHOLD_COCOK}")
    print(f"    Border margin (BORDER_MARGIN)     : {BORDER_MARGIN} piksel")
    print(f"    Penyelarasan (PAKAI_ALIGNMENT)    : "
          f"{'aktif' if PAKAI_ALIGNMENT else 'tidak aktif'}")

    print(f"\n  Citra uji ({d1['nama']}):")
    print(f"    Ridge ending   : {len(d1['e'])}")
    print(f"    Bifurcation    : {len(d1['b'])}")
    print(f"    Total minutiae : {len(d1['e']) + len(d1['b'])}")

    print(f"\n  Citra referensi ({d2['nama']}):")
    print(f"    Ridge ending   : {len(d2['e'])}")
    print(f"    Bifurcation    : {len(d2['b'])}")
    print(f"    Total minutiae : {len(d2['e']) + len(d2['b'])}")

    if PAKAI_ALIGNMENT and align is not None:
        print("\n  Penyelarasan yang terpilih:")
        if align['suara'] == 0:
            print("    Tanpa transformasi (tidak ada kandidat yang lebih baik).")
        else:
            print(f"    Perputaran   : {align['rotasi_deg']:+.1f} derajat")
            print(f"    Pergeseran x : {align['geser_x']:+.1f} piksel")
            print(f"    Pergeseran y : {align['geser_y']:+.1f} piksel")
            print(f"    Didukung oleh: {align['suara']} suara pasangan minutiae")

    print("\n  Rumus skor:")
    print("    Skor = (minutiae cocok / min(total uji, total referensi)) x 100%")
    print(f"    Skor = ({jml_cocok} / min({tot1}, {tot2})) x 100%")
    print(f"    Skor = ({jml_cocok} / {min(tot1, tot2)}) x 100%")
    print(f"    Skor = {skor:.1f}%")

    print("\n  " + "-" * 76)
    print(f"  {'No':>3}  {'Tipe':<14} {'Uji (x,y)':>11} {'Ref (x,y)':>11} "
          f"{'Jarak':>8} {'d.sudut':>9}")
    print("  " + "-" * 76)
    n_e = n_b = 0
    for p in pasangan:
        if p['tipe_kode'] == 'E':
            n_e += 1
        else:
            n_b += 1
        print(f"  {p['no']:>3}  {p['tipe']:<14} "
              f"({p['x1']:>3},{p['y1']:>3}) ({p['x2']:>3},{p['y2']:>3}) "
              f"{p['jarak']:>8.2f} {p['selisih_sudut_deg']:>8.1f}d")
    print("  " + "-" * 76)
    print(f"  Total cocok: {jml_cocok} minutiae "
          f"({n_e} ending + {n_b} bifurcation)")

    print("\n  " + "=" * 56)
    if skor >= THRESHOLD_COCOK:
        print("  IDENTIFIKASI BERHASIL")
        print(f"  Skor {skor:.1f}% >= ambang {THRESHOLD_COCOK}%")
        print(f"  Sidik jari COCOK dengan {d2['nama']}")
    else:
        print("  IDENTIFIKASI GAGAL")
        print(f"  Skor {skor:.1f}% < ambang {THRESHOLD_COCOK}%")
        print("  Sidik jari TIDAK COCOK")
    print("  " + "=" * 56)


# ======================================================================
# GAMBAR 2x4 (susunan lama dipertahankan)
# ======================================================================
def gambar_pipeline(d1, d2, jml_cocok, skor, status, path_simpan,
                    tampilkan=False):
    ov1 = gambar_overlay(d1['img'], _xy(d1['e']), _xy(d1['b']), d1['si'],
                         _theta(d1['e']), _theta(d1['b']))
    ov2 = gambar_overlay(d2['img'], _xy(d2['e']), _xy(d2['b']), d2['si'],
                         _theta(d2['e']), _theta(d2['b']))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Matching: {skor:.1f}% -> {status}\n'
                 f'({jml_cocok} minutiae cocok)',
                 fontsize=14, fontweight='bold')

    axes[0, 0].imshow(d1['img'], cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Asli (citra uji)\n({d1["nama"]}.bmp)', fontsize=8)
    axes[0, 1].imshow(d1['bin'], cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Biner')
    axes[0, 2].imshow(d1['skel'], cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('Skeleton')
    axes[0, 3].imshow(cv2.cvtColor(ov1, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title(f'Minutiae ({len(d1["e"])}E + {len(d1["b"])}B)')

    axes[1, 0].imshow(d2['img'], cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Asli (dataset)\n({d2["nama"]}.bmp)', fontsize=8)
    axes[1, 1].imshow(d2['bin'], cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Biner')
    axes[1, 2].imshow(d2['skel'], cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title('Skeleton')
    axes[1, 3].imshow(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title(f'Minutiae ({len(d2["e"])}E + {len(d2["b"])}B)')

    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=150)
    if tampilkan:
        plt.show()
    plt.close(fig)
    print(f"    Gambar 2x4   : {path_simpan}")


# ======================================================================
# GAMBAR GARIS PENGHUBUNG MINUTIAE YANG COCOK
# ======================================================================
def gambar_garis_pasangan(d1, d2, pasangan, jml_cocok, skor, status,
                          path_simpan, tampilkan=False):

    lebar = d1['img'].shape[1]
    gabung = np.hstack([d1['img'], d2['img']])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(gabung, cmap='gray', vmin=0, vmax=255)
    ax.axvline(lebar - 0.5, color='white', lw=1.2)

    for m in d1['e'] + d1['b']:
        ax.plot(m[0], m[1], 'x', color='#00e5ff', ms=5, mew=1.3)
    for m in d2['e'] + d2['b']:
        ax.plot(m[0] + lebar, m[1], 'x', color='#00e5ff', ms=5, mew=1.3)

    for p in pasangan:
        warna = '#d32f2f' if p['tipe_kode'] == 'E' else '#1565c0'
        ax.plot([p['x1'], p['x2'] + lebar], [p['y1'], p['y2']],
                '-', color=warna, lw=0.9, alpha=0.85)
        ax.plot(p['x1'], p['y1'], 'o', color=warna, ms=7,
                markerfacecolor='none', mew=1.6)
        ax.plot(p['x2'] + lebar, p['y2'], 'o', color=warna, ms=7,
                markerfacecolor='none', mew=1.6)
        ax.text(p['x1'] + 4, p['y1'] - 4, str(p['no']), color='white',
                fontsize=6.5, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.8, foreground=warna)])

    ax.set_title(
        f"Pasangan minutiae yang cocok — {d1['nama']} (kiri) vs "
        f"{d2['nama']} (kanan)\n"
        f"{jml_cocok} pasangan cocok, skor {skor:.1f}% -> {status}\n"
        f"garis merah = ridge ending, garis biru = bifurcation, "
        f"silang cyan tanpa lingkaran = minutiae yang TIDAK berpasangan",
        fontsize=9, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path_simpan, dpi=170, bbox_inches='tight')
    if tampilkan:
        plt.show()
    plt.close(fig)
    print(f"    Gambar garis : {path_simpan}")


# ======================================================================
# GAMBAR SEBARAN MINUTIAE SEBELUM DAN SESUDAH DISELARASKAN
# ======================================================================
def gambar_penyelarasan(d1, d2, align, jml_cocok, path_simpan,
                        tampilkan=False):
    m1 = _rapikan(d1['e'], 'E') + _rapikan(d1['b'], 'B')
    m2 = _rapikan(d2['e'], 'E') + _rapikan(d2['b'], 'B')
    m1_selaras = transformasi_minutiae(m1, math.radians(align['rotasi_deg']),
                                       align['geser_x'], align['geser_y'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    for k, (mm, jd) in enumerate(zip((m1, m1_selaras),
                                     ('Sebelum penyelarasan',
                                      'Sesudah penyelarasan'))):
        ax[k].scatter([m[0] for m in m2], [m[1] for m in m2], s=45,
                      facecolors='none', edgecolors='tab:blue',
                      label=f"galeri ({d2['nama']})")
        ax[k].scatter([m[0] for m in mm], [m[1] for m in mm], s=20,
                      c='tab:red', marker='x', label=f"uji ({d1['nama']})")
        ax[k].set_title(jd, fontsize=11)
        ax[k].set_xlim(0, WIDTH)
        ax[k].set_ylim(HEIGHT, 0)
        ax[k].set_aspect('equal')
        ax[k].grid(alpha=0.25)
        ax[k].set_xlabel('x (piksel)')
        ax[k].set_ylabel('y (piksel)')
    ax[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f"Penyelarasan: rotasi {align['rotasi_deg']:+.1f} derajat, "
                 f"geseran ({align['geser_x']:+.1f}, {align['geser_y']:+.1f}) "
                 f"piksel — {jml_cocok} minutiae cocok",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(path_simpan, dpi=200)
    if tampilkan:
        plt.show()
    plt.close(fig)
    print(f"    Gambar selaras: {path_simpan}")


# ======================================================================
# PEMROSESAN SATU PASANGAN
# ======================================================================
def proses_pasangan(nama_uji, nama_ref, tampilkan=False, rinci=True):
    d1 = pipeline_citra(nama_uji)
    d2 = pipeline_citra(nama_ref)
    if d1 is None or d2 is None:
        return None

    h = cocokkan_minutiae_lengkap(d1['e'], d1['b'], d2['e'], d2['b'])
    jml, skor, pasangan = h['jumlah_cocok'], h['skor'], h['pasangan']
    tot1, tot2, align = h['total_uji'], h['total_galeri'], h['alignment']
    status = 'COCOK' if skor >= THRESHOLD_COCOK else 'TIDAK COCOK'

    if rinci:
        cetak_detail(d1, d2, jml, skor, pasangan, tot1, tot2, status,
                     align=align)
    else:
        print(f"  {nama_uji:<22} vs {nama_ref:<20} {skor:>6.1f}%  "
              f"{status:<12} {jml} minutiae cocok")

    path_2x4 = os.path.join(FOLDER_HASIL, f'matching_{nama_uji}.png')
    gambar_pipeline(d1, d2, jml, skor, status, path_2x4, tampilkan=tampilkan)

    if GAMBAR_GARIS_PASANGAN:
        path_garis = os.path.join(FOLDER_HASIL, f'pasangan_{nama_uji}.png')
        gambar_garis_pasangan(d1, d2, pasangan, jml, skor, status,
                              path_garis, tampilkan=tampilkan)

    if GAMBAR_PENYELARASAN and PAKAI_ALIGNMENT:
        path_selaras = os.path.join(FOLDER_HASIL, f'selaras_{nama_uji}.png')
        gambar_penyelarasan(d1, d2, align, jml, path_selaras,
                            tampilkan=tampilkan)

    return {'citra_uji': nama_uji, 'citra_galeri': nama_ref,
            'skor': round(skor, 4), 'minutiae_cocok': jml, 'status': status}


# ======================================================================
# SUMBER PASANGAN YANG AKAN DIGAMBAR
# ======================================================================
def dari_hasil_terakhir():
    """Pasangan citra uji terakhir yang dicocokkan."""
    path = os.path.join(FOLDER_HASIL, 'hasil_terakhir.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                h = json.load(f)
            return [(nama_dasar(h['citra_baru']), nama_dasar(h['paling_cocok']))]
        except (json.JSONDecodeError, KeyError, IOError, OSError):
            print("  hasil_terakhir.json rusak, memakai tabel terbaru.")

    tabel = glob.glob(os.path.join(FOLDER_HASIL, 'tabel_*.json'))
    if not tabel:
        print("  Belum ada hasil pencocokan sama sekali.")
        print("  Jalankan bagian3.py terlebih dahulu.")
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
    for path in sorted(glob.glob(os.path.join(FOLDER_HASIL, 'tabel_*.json'))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        if 'citra_uji' in d and d.get('paling_cocok'):
            pasangan.append((d['citra_uji'], d['paling_cocok']))
    if not pasangan:
        print("  Belum ada berkas tabel_*.json di hasil_matching/.")
        print("  Jalankan MATCHING terlebih dahulu.")
    return pasangan


# ======================================================================
# PROGRAM UTAMA
# ======================================================================
if __name__ == '__main__':
    print("=" * 74)
    print("  VISUALISASI HASIL PENCOCOKAN MINUTIAE")
    print("=" * 74)

    tersedia = len(glob.glob(os.path.join(FOLDER_HASIL, 'tabel_*.json')))
    print("\n  Mau tampilkan visualisasi yang mana?")
    print("   1 = Hasil terakhir dari bagian3.py  (satu citra uji saja)")
    print(f"   2 = Semua citra uji yang sudah dicocokkan  "
          f"({tersedia} berkas tersedia)")
    pilih = tanya_pilihan("  Pilihan Anda", ['1', '2'])

    daftar = dari_hasil_terakhir() if pilih == '1' else dari_semua_tabel()
    if not daftar:
        raise SystemExit("Tidak ada yang bisa divisualisasikan.")

    rinci = True
    tampilkan = True
    if len(daftar) > 1:
        print(f"\n  {len(daftar)} citra uji akan digambar.")
        rinci = tanya_ya_tidak("  Cetak tabel pasangan minutiae untuk "
                               "tiap citra uji")
        tampilkan = tanya_ya_tidak("  Tampilkan jendela gambar tiap citra uji "
                                   "(tetap disimpan bila tidak)")

    hasil = []
    for i, (nama_uji, nama_ref) in enumerate(daftar, 1):
        print("\n" + "-" * 74)
        print(f"  [{i}/{len(daftar)}]  {nama_uji}  vs  {nama_ref}")
        print("-" * 74)
        r = proses_pasangan(nama_uji, nama_ref, tampilkan=tampilkan, rinci=rinci)
        if r:
            hasil.append(r)

    if not hasil:
        raise SystemExit("Tidak ada visualisasi yang berhasil dibuat.")

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
    print(f"\n  Gambar 2x4    : {FOLDER_HASIL}/matching_<nama_uji>.png")
    if GAMBAR_GARIS_PASANGAN:
        print(f"  Gambar garis  : {FOLDER_HASIL}/pasangan_<nama_uji>.png")
    if GAMBAR_PENYELARASAN and PAKAI_ALIGNMENT:
        print(f"  Gambar selaras: {FOLDER_HASIL}/selaras_<nama_uji>.png")
    print("\n  LANJUT EVALUASI (EER, FMR, FNMR).")
