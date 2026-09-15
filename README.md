# sidik-jari-skripsi

## Isi repo

- `config.py`
- `bagian1.py`
- `bagian2.py`
- `bagian2_gabor.py`
- `bagian3.py`
- `bagian3_gabor.py`
- `bagian4.py`
- `bagian4_gabor.py`
- `evaluasi.py`
- `raw_images/` — data input (satu-satunya folder output/data yang ikut di-commit)

Folder hasil proses (`evaluasi/`, `gabor_features/`, `hasil_gabor/`, `hasil_matching/`, `hasil_matching_gabor/`, `minutiae/`) sengaja **tidak** dimasukkan ke git — lihat `.gitignore`. Folder-folder itu akan dibuat ulang otomatis saat script dijalankan.

## Setup (venv)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan

Jalankan script sesuai urutan bagian (bagian1 -> bagian2/bagian2_gabor -> bagian3/bagian3_gabor -> bagian4/bagian4_gabor -> evaluasi), dari dalam venv yang sudah aktif.
