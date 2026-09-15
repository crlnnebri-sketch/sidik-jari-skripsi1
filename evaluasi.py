import csv
import json
import re
import shutil
import sys
import collections
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import *

# ======================================================================
# Pengaturan tampilan
# ======================================================================
WARNA_METODE = {'minutiae': '#6a1b9a', 'gabor': '#ef6c00'}
NAMA_TAMPIL = {'minutiae': 'Minutiae', 'gabor': 'Gabor'}
NOMOR_GAMBAR = {'minutiae': {'distribusi': 1, 'kurva': 3},
                'gabor':    {'distribusi': 2, 'kurva': 4}}
WARNA_GENUINE = '#1565c0'
WARNA_IMPOSTOR = '#c62828'
DPI = 200

LANGKAH_CSV   = 1.0
LANGKAH_LAYAR = 5.0

LEBAR = 74          # lebar garis pemisah di terminal

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.25,
})


def garis(ch='-'):
    print("  " + ch * (LEBAR - 2))


def judul(teks, ch='='):
    print("\n" + ch * LEBAR)
    print("  " + teks)
    print(ch * LEBAR)


def kunci_urut(nama):
    return [int(p) if p.isdigit() else p
            for p in re.split(r'(\d+)', nama or '')]


# ======================================================================
# 1. PENGUMPULAN DATA PERBANDINGAN
# ======================================================================
def kumpulkan_minutiae():

    uji = daftar_identifikasi()
    galeri = daftar_dataset()
    if not uji or not galeri:
        return []

    ciri = {}
    for path in uji + galeri:
        nama = nama_dasar(path)
        jp = cari_json_untuk_bmp(path)
        hasil = muat_minutiae_dari_json(jp, dengan_arah=True) if jp else None
        if hasil is None:
            print(f"  [LEWAT] ciri minutiae belum ada: {nama}")
            continue
        ciri[nama] = hasil

    baris = []
    for pu in sorted(uji, key=lambda p: kunci_urut(nama_dasar(p))):
        nu = nama_dasar(pu)
        if nu not in ciri:
            continue
        e1, b1 = ciri[nu]
        for pg in sorted(galeri, key=lambda p: kunci_urut(nama_dasar(p))):
            ng = nama_dasar(pg)
            if ng not in ciri:
                continue
            e2, b2 = ciri[ng]
            jml, skor, _, tot1, tot2 = cocokkan_minutiae(e1, b1, e2, b2)
            baris.append({
                'citra_uji': nu, 'subjek_uji': subjek_dari(nu),
                'citra_galeri': ng, 'subjek_galeri': subjek_dari(ng),
                'minutiae_uji': tot1, 'minutiae_galeri': tot2,
                'minutiae_cocok': jml, 'pembagi': min(tot1, tot2),
                'skor': round(skor, 4),
                'jenis_pasangan': 'genuine' if pasangan_genuine(nu, ng)
                                  else 'impostor',
                'genuine': bool(pasangan_genuine(nu, ng)),
            })
        print(f"\r  menghitung ... {len(baris)} perbandingan", end='', flush=True)
    print()
    return baris


def kumpulkan_gabor():
    
    templates = muat_template_gabor()
    uji = [t for t in templates if is_identifikasi(t['nama_dasar'])]
    galeri = [t for t in templates if is_dataset(t['nama_dasar'])]
    if not uji or not galeri:
        return []

    baris = []
    for tu in sorted(uji, key=lambda t: kunci_urut(t['nama_dasar'])):
        nu = tu['nama_dasar']
        for tg in sorted(galeri, key=lambda t: kunci_urut(t['nama_dasar'])):
            ng = tg['nama_dasar']
            baris.append({
                'citra_uji': nu, 'subjek_uji': subjek_dari(nu),
                'citra_galeri': ng, 'subjek_galeri': subjek_dari(ng),
                'jarak_ternormalisasi': round(
                    jarak_ternormalisasi(tu['vektor'], tg['vektor']), 6),
                'skor': round(skor_kemiripan(tu['vektor'], tg['vektor']), 4),
                'jenis_pasangan': 'genuine' if pasangan_genuine(nu, ng)
                                  else 'impostor',
                'genuine': bool(pasangan_genuine(nu, ng)),
            })
    print(f"  menghitung ... {len(baris)} perbandingan")
    return baris


def kumpulkan_dari_tabel(folder, ambang_sekarang=None):

    baris = []
    ambang_tersimpan = set()
    ada = set()
    for path in sorted(glob.glob(os.path.join(folder, 'tabel_*.json'))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        amb = data.get('parameter', {}).get('threshold_cocok_persen')
        if amb is not None:
            ambang_tersimpan.add(round(float(amb), 4))
        if data.get('citra_uji'):
            ada.add(data['citra_uji'])
        for r in data.get('perbandingan', []):
            r = dict(r)
            r['genuine'] = (r.get('jenis_pasangan') == 'genuine')
            baris.append(r)

    seharusnya = {nama_dasar(p) for p in daftar_identifikasi()}
    kurang = sorted(seharusnya - ada, key=kunci_urut)
    if kurang:
        print("\n  " + "!" * (LEBAR - 8))
        print(f"  PERINGATAN: {len(kurang)} citra uji belum punya berkas tabel.")
        print(f"  Terbaca {len(ada)} dari {len(seharusnya)} citra uji.")
        print("  Yang belum ada:")
        for n in kurang[:10]:
            print(f"    {n}")
        if len(kurang) > 10:
            print(f"    ... dan {len(kurang) - 10} lainnya")
        print("  AKIBATNYA seluruh metrik di bawah ini TIDAK LENGKAP dan")
        print("  berbeda dengan pilihan 1. Jalankan pencocokan ulang")
        print("  " + "!" * (LEBAR - 8) + "\n")

    if ambang_sekarang is not None and ambang_tersimpan:
        beda = [a for a in ambang_tersimpan
                if abs(a - float(ambang_sekarang)) > 1e-6]
        if beda:
            print(f"  [CATATAN] Berkas tabel memakai ambang {sorted(beda)}, "
                  f"config.py sekarang {ambang_sekarang}.")
            print("            Nilai EER tidak terpengaruh ambang, tetapi "
                  "kolom keputusan sudah kedaluwarsa.")
    return baris


# ======================================================================
# 2. PERHITUNGAN METRIK
# ======================================================================
def pisah_skor(baris):
    g = np.array([r['skor'] for r in baris if r['genuine']], dtype=float)
    i = np.array([r['skor'] for r in baris if not r['genuine']], dtype=float)
    return g, i


def kurva_fmr_fnmr(skor_genuine, skor_impostor):

    g = np.sort(skor_genuine)
    i = np.sort(skor_impostor)
    titik = np.unique(np.concatenate([g, i]))
    ambang = np.unique(np.concatenate([[titik[0] - 1e-9], titik,
                                       [titik[-1] + 1e-9]]))
    fmr = 1.0 - np.searchsorted(i, ambang, side='left') / len(i)
    fnmr = np.searchsorted(g, ambang, side='left') / len(g)
    return ambang, fmr, fnmr


def hitung_eer(ambang, fmr, fnmr):

    d = fmr - fnmr
    idx = np.where(d <= 0)[0]
    if idx.size == 0:
        k = len(d) - 1
        return float(ambang[k]), float((fmr[k] + fnmr[k]) / 2), \
            float(fmr[k]), float(fnmr[k])
    k = int(idx[0])
    if k == 0:
        return float(ambang[0]), float((fmr[0] + fnmr[0]) / 2), \
            float(fmr[0]), float(fnmr[0])
    d0, d1 = d[k - 1], d[k]
    w = float(d0 / (d0 - d1)) if d0 != d1 else 0.0
    t = float(ambang[k - 1] + w * (ambang[k] - ambang[k - 1]))
    fm = float(fmr[k - 1] + w * (fmr[k] - fmr[k - 1]))
    fn = float(fnmr[k - 1] + w * (fnmr[k] - fnmr[k - 1]))
    return t, (fm + fn) / 2.0, fm, fn


def hitung_eer_rinci(ambang, fmr, fnmr):
    """
    Sama seperti hitung_eer, tetapi juga mengembalikan DUA ambang yang
    mengapit titik potong. Dipakai agar perhitungan interpolasi dapat
    ditampilkan apa adanya di layar, bukan hanya hasil akhirnya.
    """
    d = fmr - fnmr
    idx = np.where(d <= 0)[0]
    t, eer, fm, fn = hitung_eer(ambang, fmr, fnmr)
    dasar = {'t': t, 'eer': eer, 'fmr': fm, 'fnmr': fn, 'terapit': False}
    if idx.size == 0 or int(idx[0]) == 0:
        return dasar
    k = int(idx[0])
    dasar.update({
        'terapit': True,
        't1': float(ambang[k - 1]), 'fmr1': float(fmr[k - 1]),
        'fnmr1': float(fnmr[k - 1]),
        't2': float(ambang[k]), 'fmr2': float(fmr[k]),
        'fnmr2': float(fnmr[k]),
    })
    return dasar


def hitung_pada_ambang(baris, ambang):

    h = {'TM': 0, 'FNM': 0, 'FM': 0, 'TN': 0}
    for r in baris:
        _, kode = kategori_hasil(r['genuine'], r['skor'] >= ambang)
        h[kode] += 1
    n_gen, n_imp = h['TM'] + h['FNM'], h['FM'] + h['TN']
    h['n_genuine'], h['n_impostor'] = n_gen, n_imp
    h['FMR'] = h['FM'] / n_imp if n_imp else 0.0
    h['FNMR'] = h['FNM'] / n_gen if n_gen else 0.0
    h['akurasi_keputusan'] = ((h['TM'] + h['TN']) / len(baris)) if baris else 0.0
    return h


def eksperimen_ambang(baris, langkah=LANGKAH_CSV):
    """
    Eksperimen ambang dari 0 sampai 100 dengan langkah tetap, lalu mencatat
    FMR, FNMR, jumlah tiap kategori, dan akurasi keputusan.
    """
    hasil = []
    jumlah_langkah = int(round(100.0 / langkah))
    for k in range(jumlah_langkah + 1):
        t = round(k * langkah, 4)
        h = hitung_pada_ambang(baris, t)
        hasil.append({
            'ambang_persen': t,
            'TM': h['TM'], 'FNM': h['FNM'], 'FM': h['FM'], 'TN': h['TN'],
            'n_genuine': h['n_genuine'], 'n_impostor': h['n_impostor'],
            'FMR_persen': round(h['FMR'] * 100, 4),
            'FNMR_persen': round(h['FNMR'] * 100, 4),
            'selisih_persen': round(abs(h['FMR'] - h['FNMR']) * 100, 4),
            'akurasi_keputusan_persen': round(h['akurasi_keputusan'] * 100, 4),
        })
    return hasil


def akurasi_rank1(baris):

    per_uji = collections.defaultdict(list)
    for r in baris:
        per_uji[r['citra_uji']].append(r)

    benar, rincian = 0, []
    for nama_uji in sorted(per_uji, key=kunci_urut):
        daftar = per_uji[nama_uji]
        top = max(daftar, key=lambda x: x['skor'])
        cocok = bool(top['genuine'])
        benar += int(cocok)
        rincian.append({'citra_uji': nama_uji,
                        'subjek_uji': top.get('subjek_uji',
                                              subjek_dari(nama_uji)),
                        'rank1_galeri': top['citra_galeri'],
                        'subjek_galeri': top.get('subjek_galeri',
                                                 subjek_dari(top['citra_galeri'])),
                        'skor': top['skor'],
                        'benar': cocok})
    return (benar / len(per_uji) if per_uji else 0.0), rincian


def titik_operasi_fvc(skor_genuine, skor_impostor):

    amb, fmr, fnmr = kurva_fmr_fnmr(skor_genuine, skor_impostor)
    t_eer, eer, _, _ = hitung_eer(amb, fmr, fnmr)
    hasil = {'EER': {'ambang': t_eer, 'fmr': eer * 100, 'fnmr': eer * 100}}
    for nama, target in [('FMR100', 0.01), ('FMR1000', 0.001),
                         ('ZeroFMR', 0.0)]:
        idx = np.where(fmr <= target)[0]
        if idx.size:
            k = int(idx[0])
            hasil[nama] = {'ambang': float(amb[k]), 'fmr': float(fmr[k] * 100),
                           'fnmr': float(fnmr[k] * 100)}
        else:
            hasil[nama] = None
    return hasil


def ringkas_metode(nama_metode, baris, ambang_terpasang):
    g, i = pisah_skor(baris)
    if len(g) == 0 or len(i) == 0:
        print(f"  [{nama_metode}] genuine atau impostor kosong.")
        return None

    ambang, fmr, fnmr = kurva_fmr_fnmr(g, i)
    t_eer, eer, fmr_e, fnmr_e = hitung_eer(ambang, fmr, fnmr)
    rank1, rincian = akurasi_rank1(baris)

    return {
        'metode': nama_metode,
        'jumlah_perbandingan': len(baris),
        'jumlah_citra_uji': len({r['citra_uji'] for r in baris}),
        'jumlah_citra_galeri': len({r['citra_galeri'] for r in baris}),
        'jumlah_subjek': len({r['subjek_uji'] for r in baris}),
        'jumlah_genuine': int(len(g)),
        'jumlah_impostor': int(len(i)),
        'skor_genuine': {'min': float(g.min()), 'maks': float(g.max()),
                         'rata2': float(g.mean()),
                         'std': float(g.std(ddof=1)) if len(g) > 1 else 0.0},
        'skor_impostor': {'min': float(i.min()), 'maks': float(i.max()),
                          'rata2': float(i.mean()),
                          'std': float(i.std(ddof=1)) if len(i) > 1 else 0.0},
        'eer': eer, 'eer_persen': eer * 100, 'ambang_eer': t_eer,
        'fmr_di_eer': fmr_e, 'fnmr_di_eer': fnmr_e,
        'rincian_eer': hitung_eer_rinci(ambang, fmr, fnmr),
        'kategori_di_ambang_eer': hitung_pada_ambang(baris, t_eer),
        'ambang_terpasang': ambang_terpasang,
        'kategori_di_ambang_terpasang': hitung_pada_ambang(baris,
                                                           ambang_terpasang),
        'akurasi_rank1': rank1, 'rincian_rank1': rincian,
        'titik_operasi': titik_operasi_fvc(g, i),
        'sapuan': eksperimen_ambang(baris, LANGKAH_CSV),
        '_kurva': (ambang, fmr, fnmr),
        '_skor': (g, i),
    }


# ======================================================================
# 3. KELUARAN CSV
# ======================================================================
def simpan_tabel_lengkap(baris, nama_metode, ambang, folder):
    """Seluruh perbandingan, satu baris per pasangan."""
    per_uji = collections.defaultdict(list)
    for r in baris:
        per_uji[r['citra_uji']].append(r)
    peringkat = {}
    for nu, daftar in per_uji.items():
        for k, r in enumerate(sorted(daftar, key=lambda x: -x['skor']), start=1):
            peringkat[(nu, r['citra_galeri'])] = k

    lengkap = []
    for r in baris:
        cocok = r['skor'] >= ambang
        kategori, kode = kategori_hasil(r['genuine'], cocok)
        baru = {k: v for k, v in r.items() if k != 'genuine'}
        baru.update({'ambang': round(ambang, 4),
                     'keputusan': 'COCOK' if cocok else 'TIDAK COCOK',
                     'seharusnya': 'COCOK' if r['genuine'] else 'TIDAK COCOK',
                     'kategori': kategori, 'kode_kategori': kode,
                     'benar_salah': 'BENAR' if kode in ('TM', 'TN') else 'SALAH',
                     'peringkat_skor': peringkat[(r['citra_uji'],
                                                  r['citra_galeri'])]})
        lengkap.append(baru)

    lengkap.sort(key=lambda x: (kunci_urut(x['citra_uji']),
                                kunci_urut(x['citra_galeri'])))

    path = os.path.join(folder,
                        f'lampiran_seluruh_perbandingan_{nama_metode}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lengkap[0].keys()))
        w.writeheader()
        w.writerows(lengkap)
    return path


def simpan_eksperimen_ambang(h, folder):
    """Tabel FMR & FNMR pada ambang 0..100 (langkah LANGKAH_CSV)."""
    path = os.path.join(folder, f"eksperimen_ambang_{h['metode']}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(h['sapuan'][0].keys()))
        w.writeheader()
        w.writerows(h['sapuan'])
    return path


def simpan_rank1(h, folder):
    """Daftar rank-1 tiap citra uji (bahan lampiran)."""
    path = os.path.join(folder, f"lampiran_rank1_{h['metode']}.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(h['rincian_rank1'][0].keys()))
        w.writeheader()
        w.writerows(h['rincian_rank1'])
    return path


# ======================================================================
# 4. GAMBAR — SEBARAN SKOR
# ======================================================================
def gambar_distribusi(h, folder):
    """Sebaran skor genuine dan impostor, beserta letak ambang EER."""
    g, i = h['_skor']
    warna = WARNA_METODE.get(h['metode'], '#333333')
    nama = NAMA_TAMPIL[h['metode']]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    tepi = np.linspace(0, 100, 61)          # lebar batang 100/60 = 1.67%
    ax.hist(i, bins=tepi, alpha=0.65, density=True, color=WARNA_IMPOSTOR,
            label=f"impostor (n = {h['jumlah_impostor']})")
    ax.hist(g, bins=tepi, alpha=0.65, density=True, color=WARNA_GENUINE,
            label=f"genuine (n = {h['jumlah_genuine']})")

    ax.axvline(h['ambang_eer'], color='black', ls='--', lw=1.4,
               label=f"ambang EER = {h['ambang_eer']:.2f}%")
    if abs(h['ambang_terpasang'] - h['ambang_eer']) > 1e-6:
        ax.axvline(h['ambang_terpasang'], color='#555555', ls=':', lw=1.4,
                   label=f"ambang di config.py = {h['ambang_terpasang']:.2f}%")

    sg, si = h['skor_genuine'], h['skor_impostor']
    ax.text(0.985, 0.97,
            f"genuine  : {sg['rata2']:.2f} +- {sg['std']:.2f}\n"
            f"impostor : {si['rata2']:.2f} +- {si['std']:.2f}\n"
            f"selisih  : {sg['rata2'] - si['rata2']:.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='#bbbbbb', alpha=0.9))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)   # ruang untuk legenda & kotak teks
    ax.set_xlabel('skor kemiripan (%)')
    ax.set_ylabel('kerapatan')
    ax.set_title(f"Sebaran skor genuine dan impostor — metode {nama}\n"
                 f"EER {h['eer_persen']:.2f}%, "
                 f"akurasi rank-1 {h['akurasi_rank1'] * 100:.2f}%",
                 color=warna, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()

    nomor = NOMOR_GAMBAR[h['metode']]['distribusi']
    path = os.path.join(folder, f"gambar_{nomor}_distribusi_{h['metode']}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ======================================================================
# 5. GAMBAR — KURVA FMR & FNMR
# ======================================================================
def gambar_kurva(h, folder):
    """
    Kiri  : FMR dan FNMR untuk seluruh ambang 0..100 (hasil eksperimen).
    Kanan : perbesaran di sekitar titik potong, supaya EER terlihat jelas.
    """
    sap = h['sapuan']
    t = np.array([r['ambang_persen'] for r in sap])
    fmr = np.array([r['FMR_persen'] for r in sap])
    fnmr = np.array([r['FNMR_persen'] for r in sap])

    t_eer = h['ambang_eer']
    eer = h['eer_persen']
    nama = NAMA_TAMPIL[h['metode']]
    warna = WARNA_METODE.get(h['metode'], '#333333')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax in (ax1, ax2):
        ax.plot(t, fmr, '-', color=WARNA_IMPOSTOR, lw=2, label='FMR (salah cocok)')
        ax.plot(t, fnmr, '-', color=WARNA_GENUINE, lw=2,
                label='FNMR (salah tidak cocok)')
        ax.plot([t_eer], [eer], 'o', color='black', ms=8, zorder=5,
                label=f'EER = {eer:.2f}% pada ambang {t_eer:.2f}%')
        ax.axvline(t_eer, color='black', ls='--', lw=1, alpha=0.6)
        ax.axhline(eer, color='black', ls='--', lw=1, alpha=0.6)
        ax.set_xlabel('ambang keputusan (%)')
        ax.set_ylabel('laju kesalahan (%)')

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-2, 102)
    ax1.set_title('Seluruh rentang ambang 0 - 100%')
    ax1.legend(loc='center right', framealpha=0.9)

    lebar = 15
    ax2.set_xlim(max(0, t_eer - lebar), min(100, t_eer + lebar))
    batas = min(100, max(eer * 3, 10))
    ax2.set_ylim(-1, batas)
    ax2.set_title(f'Perbesaran di sekitar titik EER '
                  f'({max(0, t_eer - lebar):.0f} - {min(100, t_eer + lebar):.0f}%)')

    fig.suptitle(f'Kurva FMR dan FNMR — metode {nama}',
                 fontsize=12, fontweight='bold', color=warna)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    nomor = NOMOR_GAMBAR[h['metode']]['kurva']
    path = os.path.join(folder, f"gambar_{nomor}_kurva_{h['metode']}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ======================================================================
# 6. GAMBAR — DIAGRAM BATANG RINGKAS
# ======================================================================
def gambar_batang_ringkas(daftar, folder):
    """Diagram batang EER dan rank-1"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    nama = [NAMA_TAMPIL[h['metode']] for h in daftar]
    warna = [WARNA_METODE.get(h['metode']) for h in daftar]

    v1 = [h['eer_persen'] for h in daftar]
    ax1.bar(nama, v1, color=warna, alpha=0.9, width=0.55)
    for x, v in zip(nama, v1):
        ax1.text(x, v + 1.2, f"{v:.2f}%", ha='center', fontweight='bold')
    ax1.set_ylabel('EER (%)')
    ax1.set_title('Equal Error Rate\n(makin kecil makin baik)')
    ax1.set_ylim(0, max(max(v1) * 1.28, 1.0))

    v2 = [h['akurasi_rank1'] * 100 for h in daftar]
    ax2.bar(nama, v2, color=warna, alpha=0.9, width=0.55)
    for x, v in zip(nama, v2):
        ax2.text(x, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    ax2.set_ylabel('Akurasi rank-1 (%)')
    ax2.set_title('Akurasi identifikasi rank-1\n(makin besar makin baik)')
    ax2.set_ylim(0, 108)

    fig.tight_layout()
    path = os.path.join(folder, 'gambar_5_perbandingan_ringkas.png')
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ======================================================================
# 6b. KURVA ROC, AUC, DAN GAR PADA TITIK FMR BAKU
#
# Rujukan penyajian: Jain dkk. 2000, IEEE TIP 9(5):846-859 [P2],
# yang menyatakan hasilnya dalam bentuk kedudukan kurva ROC.
#
# Kesepakatan yang dipakai:
#   FMR(t) = proporsi pasangan impostor yang skornya >= t
#   GAR(t) = proporsi pasangan genuine  yang skornya >= t = 1 - FNMR(t)
# Kurva ROC adalah himpunan titik (FMR(t), GAR(t)) untuk seluruh t.
# ======================================================================
TITIK_FMR_BAKU = (0.001, 0.01, 0.10)     # 0,1% ; 1% ; 10%
FMR_LANTAI = 5e-4                        # 0,05% - batas kiri sumbu log


def hitung_roc(h):
    """
    Kurva ROC satu metode, dihitung dari hasil ringkas_metode().

    Sengaja memakai kembali kurva FMR/FNMR yang SUDAH dihitung
    ringkas_metode() dan disimpan pada kunci '_kurva'. Jadi ambang, FMR,
    dan FNMR yang dipakai ROC persis sama dengan yang dipakai EER dan
    titik operasi FVC, sehingga Tabel ROC dan Tabel FVC pada naskah
    dijamin tidak saling bertentangan.
    """
    ambang, fmr, fnmr = h['_kurva']
    gar = 1.0 - fnmr
    g, i = h['_skor']

    # AUC dihitung dengan statistik Mann-Whitney U, bukan luas trapesium.
    # Alasannya: skor sensor ini banyak yang kembar (ties) akibat kuantisasi
    # aras keabuan, dan U menangani nilai kembar secara tepat sedangkan
    # trapesium hanya menghampiri. Tafsirannya langsung: AUC adalah peluang
    # satu skor genuine acak melebihi satu skor impostor acak.
    gabung = np.concatenate([g, i])
    urut = np.argsort(gabung, kind='mergesort')
    peringkat = np.empty(len(gabung), dtype=float)
    peringkat[urut] = np.arange(1, len(gabung) + 1, dtype=float)
    nilai_urut = gabung[urut]
    mulai = 0
    for k in range(1, len(nilai_urut) + 1):        # rerata untuk yang kembar
        if k == len(nilai_urut) or nilai_urut[k] != nilai_urut[mulai]:
            if k - mulai > 1:
                peringkat[urut[mulai:k]] = peringkat[urut[mulai:k]].mean()
            mulai = k
    auc = (peringkat[:len(g)].sum() - len(g) * (len(g) + 1) / 2.0) \
        / (len(g) * len(i))

    return {'ambang': ambang, 'fmr': fmr, 'gar': gar, 'auc': float(auc)}


def gar_pada_fmr(roc, target):
    """
    GAR pada satu titik FMR baku, memakai kesepakatan FVC: diambil ambang
    TERKECIL yang FMR-nya masih <= target, lalu dibaca GAR di ambang itu.
    Kesepakatan ini sama persis dengan titik_operasi_fvc(), sehingga GAR di
    sini selalu = 100 - FNMR di sana.

    Sengaja TIDAK memakai interpolasi linier, karena sebaran skor pada basis
    data ini diskret; nilai hasil interpolasi akan menunjuk ambang yang tidak
    pernah benar-benar ada, sehingga tidak dapat dipertanggungjawabkan.
    """
    idx = np.where(roc['fmr'] <= target)[0]
    if idx.size == 0:
        return None
    k = int(idx[0])
    return {'target_fmr_persen': target * 100,
            'ambang': float(roc['ambang'][k]),
            'fmr_persen': float(roc['fmr'][k] * 100),
            'gar_persen': float(roc['gar'][k] * 100)}


def dominasi_roc(roc_a, roc_b, n_titik=2000):
    """
    Berapa persen titik FMR yang kurva A-nya berada di ATAS kurva B.
    Dicuplik merata pada skala logaritmik karena sumbu mendatar gambarnya
    juga logaritmik.
    """
    grid = np.logspace(np.log10(FMR_LANTAI), 0.0, n_titik)
    a = np.interp(grid, roc_a['fmr'][::-1], roc_a['gar'][::-1])
    b = np.interp(grid, roc_b['fmr'][::-1], roc_b['gar'][::-1])
    return float((a > b).mean() * 100), float((a >= b).mean() * 100)


def simpan_tabel_roc(daftar, folder):
    """Dua CSV: seluruh titik kurva, dan ringkasan AUC + GAR."""
    path_kurva = os.path.join(folder, 'roc_titik.csv')
    with open(path_kurva, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metode', 'ambang', 'FMR_persen', 'GAR_persen',
                    'FNMR_persen'])
        for h in daftar:
            r = h['roc']
            for t, fm, ga in zip(r['ambang'], r['fmr'], r['gar']):
                w.writerow([h['metode'], round(float(t), 6),
                            round(float(fm) * 100, 6),
                            round(float(ga) * 100, 6),
                            round((1 - float(ga)) * 100, 6)])

    path_ring = os.path.join(folder, 'roc_ringkas.csv')
    with open(path_ring, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metode', 'AUC', 'target_FMR_persen', 'ambang',
                    'FMR_tercapai_persen', 'GAR_persen'])
        for h in daftar:
            for target in TITIK_FMR_BAKU:
                t = gar_pada_fmr(h['roc'], target)
                if t is None:
                    w.writerow([h['metode'], round(h['roc']['auc'], 4),
                                target * 100, '', '', 'tidak tercapai'])
                else:
                    w.writerow([h['metode'], round(h['roc']['auc'], 4),
                                t['target_fmr_persen'], round(t['ambang'], 4),
                                round(t['fmr_persen'], 4),
                                round(t['gar_persen'], 2)])
    return path_kurva, path_ring


def gambar_roc(daftar, folder):
    """
    Kurva ROC kedua metode pada satu sumbu. Sumbu mendatar FMR logaritmik,
    sumbu tegak GAR = 100 - FNMR, mengikuti cara penyajian [P2].
    Tanda bintang = titik EER.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for h in daftar:
        r = h['roc']
        nama = NAMA_TAMPIL[h['metode']]
        warna = WARNA_METODE.get(h['metode'], '#333333')

        # FMR = 0 tidak dapat digambar pada sumbu logaritmik; titik tersebut
        # digeser ke FMR_LANTAI dan dinyatakan pada keterangan sumbu.
        fmr = np.clip(r['fmr'], FMR_LANTAI, None) * 100
        ax.plot(fmr, r['gar'] * 100, '-', color=warna, lw=2,
                label=f"{nama}  (AUC = {r['auc']:.4f}, "
                      f"EER = {h['eer_persen']:.2f}%)")

        fmr_eer = max(h['eer_persen'] / 100.0, FMR_LANTAI) * 100
        ax.plot([fmr_eer], [100 - h['eer_persen']], '*', color=warna,
                ms=15, zorder=5, markeredgecolor='black', markeredgewidth=0.6)

    for target in TITIK_FMR_BAKU:
        ax.axvline(target * 100, color='gray', ls=':', lw=0.9, alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlim(FMR_LANTAI * 100, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.05, 0.1, 1, 10, 100])
    ax.set_xticklabels(['0,05', '0,1', '1', '10', '100'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel('FMR (%) - skala logaritmik')
    ax.set_ylabel('GAR (%) = 100% - FNMR')
    ax.set_title('Kurva ROC kedua metode pada basis citra yang sama',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.92)
    ax.grid(True, which='both', alpha=0.25)

    fig.tight_layout()
    path = os.path.join(folder, 'gambar_6_kurva_roc.png')
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def cetak_roc(daftar):
    """Cetak angka ROC ke layar supaya dapat langsung disalin ke naskah."""
    judul('KURVA ROC - ANGKA UNTUK NASKAH', '=')
    print(f"  {'Metode':<12}{'AUC':>10}")
    print("  " + "-" * (LEBAR - 2))
    for h in daftar:
        print(f"  {NAMA_TAMPIL[h['metode']]:<12}{h['roc']['auc']:>10.4f}")

    print()
    print(f"  {'FMR target':>11}", end='')
    for h in daftar:
        print(f"{'GAR ' + NAMA_TAMPIL[h['metode']]:>16}", end='')
    print(f"{'Selisih':>12}")
    print("  " + "-" * (LEBAR - 2))
    for target in TITIK_FMR_BAKU:
        print(f"  {target * 100:>10.1f}%", end='')
        nilai = []
        for h in daftar:
            t = gar_pada_fmr(h['roc'], target)
            nilai.append(t['gar_persen'] if t else float('nan'))
            print(f"{nilai[-1]:>15.2f}%", end='')
        print(f"{nilai[-1] - nilai[0]:>11.2f} poin")
    print("  " + "-" * (LEBAR - 2))

    if len(daftar) == 2:
        ketat, longgar = dominasi_roc(daftar[1]['roc'], daftar[0]['roc'])
        print(f"  Kurva {NAMA_TAMPIL[daftar[1]['metode']]} berada DI ATAS "
              f"kurva {NAMA_TAMPIL[daftar[0]['metode']]} pada "
              f"{ketat:.1f}% titik FMR")
        print(f"  (di atas atau sama pada {longgar:.1f}% titik FMR; "
              f"cuplikan {FMR_LANTAI * 100:.2f}% - 100%, skala logaritmik)")




def simpan_ringkasan_metrik(daftar_hasil, folder):
    """Menulis SELURUH angka pokok ke satu berkas ringkasan_metrik.json."""
    ringkas = {
        'dibuat': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sumber_metode': {
            'minutiae': 'Jain dkk. 1997, Proc. IEEE 85(9):1365-1388',
            'gabor': 'Jain dkk. 2000, IEEE TIP 9(5):846-859',
        },
        'parameter_minutiae': {
            'blok_orientasi_W': P1_BLOK_W, 'filter_penghalus_D': P1_SMOOTH_D,
            'ambang_certainty_Ts': P1_AMBANG_CERTAINTY,
            'ambang_konsistensi_Tc': P1_AMBANG_KONSISTENSI,
            'mask_ridge': f'{P1_MASK_L}x{P1_MASK_H}',
            'ambang_ridge_Tridge': P1_AMBANG_RIDGE,
            'toleransi_sudut_deg': P1_TOLERANSI_SUDUT,
            'kandidat_selaras': P1_KANDIDAT_SELARAS,
            'kandidat_dp': P1_KANDIDAT_DP,
        },
        'parameter_gabor': {
            'orientasi': P2_ORIENTASI, 'jumlah_pita_B': P2_JUMLAH_PITA,
            'sektor_per_pita_k': P2_SEKTOR_PITA,
            'lebar_pita_b': P2_LEBAR_PITA, 'jari_dalam_r0': P2_JARI_DALAM,
            'frekuensi_f': round(GABOR_FREKUENSI, 4),
            'delta_x_y': P2_GABOR_DELTA, 'mask': P2_GABOR_MASK,
            'normalisasi_M0': P2_M0, 'normalisasi_V0': P2_V0,
            'dimensi_fingercode': GABOR_DIMENSI,
        },
        'metode': {},
    }
    for h in daftar_hasil:
        ringkas['metode'][h['metode']] = {
            'nama_tampil': NAMA_TAMPIL[h['metode']],
            'jumlah_perbandingan': h['jumlah_perbandingan'],
            'jumlah_citra_uji': h['jumlah_citra_uji'],
            'jumlah_citra_galeri': h['jumlah_citra_galeri'],
            'jumlah_subjek': h['jumlah_subjek'],
            'jumlah_genuine': h['jumlah_genuine'],
            'jumlah_impostor': h['jumlah_impostor'],
            'eer_persen': round(h['eer_persen'], 4),
            'ambang_eer': round(h['ambang_eer'], 4),
            'fmr_di_eer_persen': round(h['fmr_di_eer'] * 100, 4),
            'fnmr_di_eer_persen': round(h['fnmr_di_eer'] * 100, 4),
            'akurasi_rank1_persen': round(h['akurasi_rank1'] * 100, 4),
            'skor_genuine': {k: round(v, 4)
                             for k, v in h['skor_genuine'].items()},
            'skor_impostor': {k: round(v, 4)
                              for k, v in h['skor_impostor'].items()},
            'titik_operasi': h['titik_operasi'],
            'ambang_terpasang': h['ambang_terpasang'],
            'kategori_di_ambang_eer': h['kategori_di_ambang_eer'],
            'kategori_di_ambang_terpasang': h['kategori_di_ambang_terpasang'],
        }

    for h in daftar_hasil:
        sg, si = h['skor_genuine'], h['skor_impostor']
        penyebut = ((sg['std'] ** 2 + si['std'] ** 2) / 2.0) ** 0.5
        ringkas['metode'][h['metode']]['d_prime'] = (
            round((sg['rata2'] - si['rata2']) / penyebut, 4)
            if penyebut > 0 else None)

    if len(daftar_hasil) > 1:
        a, b = daftar_hasil[0], daftar_hasil[1]
        ringkas['perbandingan'] = {
            'selisih_eer_poin_persen': round(
                a['eer_persen'] - b['eer_persen'], 4),
            'selisih_rank1_poin_persen': round(
                (a['akurasi_rank1'] - b['akurasi_rank1']) * 100, 4),
            'eer_terkecil': min(daftar_hasil,
                                key=lambda x: x['eer_persen'])['metode'],
            'rank1_tertinggi': max(daftar_hasil,
                                   key=lambda x: x['akurasi_rank1'])['metode'],
        }

    path = os.path.join(folder, 'ringkasan_metrik.json')
    with open(path, 'w') as f:
        json.dump(ringkas, f, indent=2, ensure_ascii=False)
    return path


# ======================================================================
# 7. PENULISAN AMBANG KE config.py
# ======================================================================
NAMA_VARIABEL_AMBANG = {'minutiae': 'THRESHOLD_COCOK',
                        'gabor': 'SKOR_THRESHOLD_GABOR'}


def tulis_ambang_ke_config(daftar_hasil, path_config=None):
    """
    Menuliskan ambang pada titik EER ke config.py.
    config.py disalin dulu ke config_cadangan.py, hanya baris ambang yang
    diganti, dan hasilnya diperiksa sintaksnya sebelum ditulis.
    """
    path_config = path_config or os.path.join(BASE_DIR, 'config.py')
    if not os.path.exists(path_config):
        print("  config.py tidak ditemukan.")
        return False
    with open(path_config) as f:
        isi = f.read()

    stempel = datetime.now().strftime('%Y-%m-%d %H:%M')
    diubah = []
    for h in daftar_hasil:
        nama_var = NAMA_VARIABEL_AMBANG.get(h['metode'])
        if not nama_var:
            continue
        pola = re.compile(rf'^({nama_var}\s*=\s*)([-+0-9.eE]+)(.*)$', re.M)
        m = pola.search(isi)
        if not m:
            print(f"  [LEWAT] baris {nama_var} tidak ditemukan.")
            continue
        baru = f"{h['ambang_eer']:.4f}"
        komentar = (f"  # <<< SESUAIKAN  ambang pada titik EER, ditulis "
                    f"otomatis {stempel} (EER {h['eer_persen']:.2f}%)")
        isi = pola.sub(lambda mm: mm.group(1) + baru + komentar, isi, count=1)
        diubah.append((nama_var, m.group(2), baru, h['eer_persen']))

    if not diubah:
        return False
    try:
        compile(isi, path_config, 'exec')
    except SyntaxError as e:
        print(f"  Hasil penggantian tidak valid ({e}). config.py TIDAK diubah.")
        return False

    shutil.copy(path_config, os.path.join(BASE_DIR, 'config_cadangan.py'))
    with open(path_config, 'w') as f:
        f.write(isi)

    print("\n  Ambang ditulis ke config.py:")
    print(f"    {'Variabel':<24}{'Lama':>10}{'Baru':>10}{'EER':>10}")
    print("    " + "-" * 54)
    for nama_var, lama, baru, eer in diubah:
        print(f"    {nama_var:<24}{lama:>10}{baru:>10}{eer:>9.2f}%")
    print("\n  Jalankan ulang bagian3.py dan bagian3_gabor.py agar kolom")
    print("  keputusan pada seluruh tabel memakai ambang yang baru.")
    return True


# ======================================================================
# 8. CETAK RINGKASAN KE LAYAR (disusun bertahap, lengkap dengan hitungannya)
# ======================================================================
def cetak_ringkasan(h, urutan=None, total=None):
    n = NAMA_TAMPIL[h['metode']].upper()
    kepala = f"METODE {n}" if urutan is None else f"[{urutan}/{total}]  METODE {n}"
    judul(kepala, '=')

    print("\n  A. DATA PERBANDINGAN")
    print(f"     Citra uji                 : {h['jumlah_citra_uji']}")
    print(f"     Citra galeri              : {h['jumlah_citra_galeri']}")
    print(f"     Jumlah subjek             : {h['jumlah_subjek']}")
    print(f"     Total perbandingan        : {h['jumlah_citra_uji']} x "
          f"{h['jumlah_citra_galeri']} = {h['jumlah_perbandingan']}")
    print(f"       pasangan genuine        : {h['jumlah_genuine']:>6}   "
          f"(subjek uji = subjek galeri)")
    print(f"       pasangan impostor       : {h['jumlah_impostor']:>6}   "
          f"(subjek berbeda)")
    print(f"       pemeriksaan             : {h['jumlah_genuine']} + "
          f"{h['jumlah_impostor']} = "
          f"{h['jumlah_genuine'] + h['jumlah_impostor']}"
          f"{'  (cocok)' if h['jumlah_genuine'] + h['jumlah_impostor'] == h['jumlah_perbandingan'] else '  (TIDAK COCOK, periksa data)'}")

    sg, si = h['skor_genuine'], h['skor_impostor']
    print("\n  B. SEBARAN SKOR (%)")
    print(f"     {'':<12}{'min':>9}{'maks':>9}{'rata2':>9}{'sd':>9}")
    print(f"     {'genuine':<12}{sg['min']:>9.2f}{sg['maks']:>9.2f}"
          f"{sg['rata2']:>9.2f}{sg['std']:>9.2f}")
    print(f"     {'impostor':<12}{si['min']:>9.2f}{si['maks']:>9.2f}"
          f"{si['rata2']:>9.2f}{si['std']:>9.2f}")
    print(f"     Jarak rata-rata = {sg['rata2']:.2f} - {si['rata2']:.2f} "
          f"= {sg['rata2'] - si['rata2']:.2f}")
    print("     Makin jauh jaraknya, makin mudah kedua kelompok dipisahkan.")

    k = h['kategori_di_ambang_eer']
    re_ = h['rincian_eer']
    print("\n  C. PERHITUNGAN EER")
    print("     Skor bernilai diskret, sehingga kurva FMR dan FNMR berbentuk")
    print("     tangga dan keduanya hampir tidak pernah persis sama. Titik")
    print("     potongnya dicari dengan interpolasi linear di antara dua")
    print("     ambang berurutan yang mengapit perpotongan.")
    if re_['terapit']:
        print(f"       {'':<16}{'ambang':>10}{'FMR':>10}{'FNMR':>10}")
        print("       " + "-" * 46)
        print(f"       {'sebelum potong':<16}{re_['t1']:>9.2f}%"
              f"{re_['fmr1'] * 100:>9.2f}%{re_['fnmr1'] * 100:>9.2f}%")
        print(f"       {'sesudah potong':<16}{re_['t2']:>9.2f}%"
              f"{re_['fmr2'] * 100:>9.2f}%{re_['fnmr2'] * 100:>9.2f}%")
        print(f"       {'titik potong':<16}{re_['t']:>9.2f}%"
              f"{re_['fmr'] * 100:>9.2f}%{re_['fnmr'] * 100:>9.2f}%")
        print("       " + "-" * 46)
    print(f"       EER = (FMR + FNMR) / 2 = "
          f"({h['fmr_di_eer'] * 100:.2f}% + {h['fnmr_di_eer'] * 100:.2f}%) / 2 "
          f"= {h['eer_persen']:.2f}%")
    print(f"       Ambang pada titik EER = {h['ambang_eer']:.2f}%")

    print(f"\n  D. EMPAT KATEGORI PADA AMBANG EER (t = {h['ambang_eer']:.2f}%)")
    print("     Cacahan di bawah dihitung pada ambang itu APA ADANYA, jadi")
    print("     FMR dan FNMR-nya sedikit berbeda dari nilai interpolasi di C.")
    print(f"     {'benar cocok':<22}TM  = {k['TM']:>6}")
    print(f"     {'salah tidak cocok':<22}FNM = {k['FNM']:>6}   "
          f"FNMR = {k['FNM']}/{k['n_genuine']} = {k['FNMR'] * 100:.2f}%")
    print(f"     {'salah cocok':<22}FM  = {k['FM']:>6}   "
          f"FMR  = {k['FM']}/{k['n_impostor']} = {k['FMR'] * 100:.2f}%")
    print(f"     {'benar tidak cocok':<22}TN  = {k['TN']:>6}")
    print(f"     akurasi keputusan = (TM + TN) / total = "
          f"({k['TM']} + {k['TN']}) / {h['jumlah_perbandingan']} = "
          f"{k['akurasi_keputusan'] * 100:.2f}%")

    kt = h['kategori_di_ambang_terpasang']
    print(f"\n  E. PADA AMBANG YANG TERPASANG DI config.py "
          f"(t = {h['ambang_terpasang']:.2f}%)")
    print(f"     TM {kt['TM']}, FNM {kt['FNM']}, FM {kt['FM']}, TN {kt['TN']}")
    print(f"     FNMR = {kt['FNM']}/{kt['n_genuine']} = "
          f"{kt['FNMR'] * 100:.2f}%   |   "
          f"FMR = {kt['FM']}/{kt['n_impostor']} = {kt['FMR'] * 100:.2f}%")
    if abs(h['ambang_terpasang'] - h['ambang_eer']) > 0.005:
        print("     Ambang ini BERBEDA dengan ambang EER. Isi config.py dengan")
        print("     ambang EER bila ingin kedua angka itu sama.")

    benar = sum(1 for r in h['rincian_rank1'] if r['benar'])
    total_uji = len(h['rincian_rank1'])
    print("\n  F. AKURASI IDENTIFIKASI RANK-1")
    print("     Untuk tiap citra uji diambil citra galeri berskor tertinggi.")
    print(f"       benar                   : {benar} dari {total_uji} citra uji")
    print(f"       akurasi rank-1          : {benar} / {total_uji} x 100% = "
          f"{h['akurasi_rank1'] * 100:.2f}%")
    salah = [r for r in h['rincian_rank1'] if not r['benar']]
    if salah:
        print(f"       citra uji yang salah    : {len(salah)}")
        for r in salah[:5]:
            print(f"         {r['citra_uji']:<24} -> {r['rank1_galeri']:<24}"
                  f"{r['skor']:>7.2f}%")
        if len(salah) > 5:
            print(f"         ... dan {len(salah) - 5} lainnya "
                  f"(lihat lampiran_rank1_{h['metode']}.csv)")

    print("\n  G. TITIK OPERASI BAKU (protokol FVC)")
    print(f"     {'Titik':<12}{'Ambang':>10}{'FMR':>10}{'FNMR':>10}")
    print("     " + "-" * 42)
    for nama in ('EER', 'FMR100', 'FMR1000', 'ZeroFMR'):
        v = h['titik_operasi'].get(nama)
        if v is None:
            print(f"     {nama:<12}{'tidak tercapai':>30}")
        else:
            print(f"     {nama:<12}{v['ambang']:>9.2f}%{v['fmr']:>9.2f}%"
                  f"{v['fnmr']:>9.2f}%")
    print("     FMR100  = FNMR terendah ketika FMR <= 1%")
    print("     FMR1000 = FNMR terendah ketika FMR <= 0,1%")
    print("     ZeroFMR = FNMR terendah ketika FMR = 0%")

    cetak_sapuan(h)


def cetak_sapuan(h):
    """Tabel FMR & FNMR pada ambang 0..100, ditampilkan berlangkah kasar."""
    print(f"\n  H. EKSPERIMEN AMBANG 0 - 100% "
          f"(layar berlangkah {LANGKAH_LAYAR:.0f}%, "
          f"berkas CSV berlangkah {LANGKAH_CSV:.0f}%)")
    print(f"     {'Ambang':>8}{'TM':>7}{'FNM':>7}{'FM':>8}{'TN':>8}"
          f"{'FMR':>9}{'FNMR':>9}{'|beda|':>9}{'Akurasi':>9}")
    print("     " + "-" * 74)

    i_dekat = min(range(len(h['sapuan'])),
                  key=lambda k: h['sapuan'][k]['selisih_persen'])

    langkah_tampil = max(1, int(round(LANGKAH_LAYAR / LANGKAH_CSV)))
    tampil = set(range(0, len(h['sapuan']), langkah_tampil))
    tampil.add(len(h['sapuan']) - 1)
    tampil.add(i_dekat)

    for k in sorted(tampil):
        r = h['sapuan'][k]
        tanda = ' <- paling dekat EER' if k == i_dekat else ''
        print(f"     {r['ambang_persen']:>7.0f}%{r['TM']:>7}{r['FNM']:>7}"
              f"{r['FM']:>8}{r['TN']:>8}"
              f"{r['FMR_persen']:>8.2f}%{r['FNMR_persen']:>8.2f}%"
              f"{r['selisih_persen']:>8.2f}%"
              f"{r['akurasi_keputusan_persen']:>8.2f}%{tanda}")
    print("     " + "-" * 74)
    print("     Ambang naik  -> FMR turun, FNMR naik. EER = titik seimbangnya.")
    print(f"     Tabel ini berlangkah {LANGKAH_CSV:.0f}%, jadi titik potongnya "
          f"kasar. EER resmi")
    print(f"     ({h['eer_persen']:.2f}% pada ambang {h['ambang_eer']:.2f}%) "
          f"dihitung dari seluruh nilai skor")
    print("     yang benar-benar muncul, memakai interpolasi linear.")


# ======================================================================
# 9. PROGRAM UTAMA
# ======================================================================
def jalankan(sumber='hitung_ulang', tulis=None):
    folder = os.path.join(FOLDER_EVALUASI, 'PERHITUNGAN')
    os.makedirs(folder, exist_ok=True)

    daftar_metode = [
        ('minutiae', kumpulkan_minutiae, FOLDER_HASIL, THRESHOLD_COCOK),
        ('gabor', kumpulkan_gabor, FOLDER_MATCHING_GABOR, SKOR_THRESHOLD_GABOR),
    ]

    hasil_semua = []
    keluaran = []

    for nomor, (nama_metode, fungsi, folder_tabel, ambang) in enumerate(
            daftar_metode, start=1):

        judul(f"[{nomor}/{len(daftar_metode)}]  MENGUMPULKAN DATA — "
              f"METODE {NAMA_TAMPIL[nama_metode].upper()}", '-')

        baris = (fungsi() if sumber == 'hitung_ulang'
                 else kumpulkan_dari_tabel(folder_tabel, ambang))
        if not baris:
            print(f"  Tidak ada data. Metode {nama_metode} dilewati.")
            continue

        h = ringkas_metode(nama_metode, baris, ambang)
        if h is None:
            continue

        cetak_ringkasan(h, urutan=nomor, total=len(daftar_metode))

        nm = NAMA_TAMPIL[nama_metode]
        keluaran.append((f"Seluruh perbandingan {nm} (CSV)",
                         simpan_tabel_lengkap(baris, nama_metode,
                                              h['ambang_eer'], folder)))
        keluaran.append((f"Eksperimen ambang 0-100 {nm} (CSV)",
                         simpan_eksperimen_ambang(h, folder)))
        keluaran.append((f"Rank-1 tiap citra uji {nm} (CSV)",
                         simpan_rank1(h, folder)))
        keluaran.append((f"Sebaran skor {nm} (gambar)",
                         gambar_distribusi(h, folder)))
        keluaran.append((f"Kurva FMR & FNMR {nm} (gambar)",
                         gambar_kurva(h, folder)))
        hasil_semua.append(h)

    if not hasil_semua:
        print("\nTidak ada metode yang dapat dievaluasi.")
        return

    jumlah = {h['jumlah_perbandingan'] for h in hasil_semua}
    if len(jumlah) > 1:
        print("\n  " + "!" * (LEBAR - 8))
        print("  PERINGATAN: jumlah perbandingan kedua metode BERBEDA "
              f"{sorted(jumlah)}.")
        print("  Kedua metode harus diuji pada citra yang sama agar dapat")
        print("  dibandingkan. Pastikan bagian2.py dan bagian2_gabor.py sudah")
        print("  dijalankan untuk SELURUH citra.")
        print("  " + "!" * (LEBAR - 8))

    if len(hasil_semua) > 1:
        keluaran.append(('Diagram batang EER & rank-1 (gambar)',
                         gambar_batang_ringkas(hasil_semua, folder)))
    else:
        print("\n  [CATATAN] Diagram batang perbandingan tidak dibuat karena")
        print("            hanya satu metode yang berhasil dievaluasi.")

    # --- kurva ROC ---
    for h in hasil_semua:
        h['roc'] = hitung_roc(h)
    p1, p2 = simpan_tabel_roc(hasil_semua, folder)
    keluaran.append(('Seluruh titik kurva ROC (CSV)', p1))
    keluaran.append(('Ringkasan AUC & GAR (CSV)', p2))
    if len(hasil_semua) > 1:
        keluaran.append(('Kurva ROC kedua metode (gambar)',
                         gambar_roc(hasil_semua, folder)))
    cetak_roc(hasil_semua)

    judul("ANGKA POKOK UNTUK HASIL", '=')
    print(f"  {'Metode':<12}{'EER':>10}{'Ambang EER':>13}{'Rank-1':>10}"
          f"{'Genuine':>10}{'Impostor':>10}")
    print("  " + "-" * (LEBAR - 2))
    for h in hasil_semua:
        sg, si = h['skor_genuine'], h['skor_impostor']
        print(f"  {NAMA_TAMPIL[h['metode']]:<12}{h['eer_persen']:>9.2f}%"
              f"{h['ambang_eer']:>12.2f}%{h['akurasi_rank1'] * 100:>9.2f}%"
              f"{sg['rata2']:>10.2f}{si['rata2']:>10.2f}")
    print("  " + "-" * (LEBAR - 2))
    if len(hasil_semua) > 1:
        a, b = hasil_semua[0], hasil_semua[1]
        print(f"  Selisih EER    : {a['eer_persen']:.2f}% - "
              f"{b['eer_persen']:.2f}% = "
              f"{a['eer_persen'] - b['eer_persen']:+.2f} poin persen")
        print(f"  Selisih rank-1 : {a['akurasi_rank1'] * 100:.2f}% - "
              f"{b['akurasi_rank1'] * 100:.2f}% = "
              f"{(a['akurasi_rank1'] - b['akurasi_rank1']) * 100:+.2f} "
              f"poin persen")
        unggul = min(hasil_semua, key=lambda x: x['eer_persen'])
        print(f"  EER terkecil   : {NAMA_TAMPIL[unggul['metode']]}")

    keluaran.append(('Ringkasan seluruh angka pokok (JSON)',
                     simpan_ringkasan_metrik(hasil_semua, folder)))

    judul("BERKAS KELUARAN", '=')
    for judul_berkas, path in keluaran:
        print(f"  {judul_berkas:<42} {os.path.basename(path)}")
    print(f"\n  Seluruh berkas ada di: {folder}")

    judul("AMBANG PADA TITIK EER", '=')
    for h in hasil_semua:
        print(f"  {NAMA_VARIABEL_AMBANG[h['metode']]:<24}"
              f"= {h['ambang_eer']:.2f}    "
              f"(EER {h['eer_persen']:.2f}%)")
    if tulis is None:
        tulis = tanya_ya_tidak("\n  Tulis ambang tersebut ke config.py")
    if tulis:
        tulis_ambang_ke_config(hasil_semua)
    else:
        print("\n  config.py tidak diubah.")


if __name__ == '__main__':
    print("=" * LEBAR)
    print("  EVALUASI UNJUK KERJA — FMR, FNMR, EER, DAN AKURASI RANK-1")
    print("=" * LEBAR)

    tulis = True if '--tulis-ambang' in sys.argv else (
        False if '--tanpa-tulis' in sys.argv else None)

    if '--auto' in sys.argv:
        jalankan('hitung_ulang', tulis=tulis)
    else:
        print("\n  Sumber data perbandingan:")
        print("   1 = Hitung ulang dari berkas ciri  (DISARANKAN, selalu lengkap)")
        print("   2 = Kumpulkan dari berkas tabel_*.json hasil bagian3")
        print("       (hanya benar bila bagian3 sudah dijalankan untuk")
        print("        SELURUH citra uji; bila belum, hasilnya berbeda)")
        pilih = tanya_pilihan("  Pilihan Anda", ['1', '2'])
        jalankan('hitung_ulang' if pilih == '1' else 'dari_tabel', tulis=tulis)