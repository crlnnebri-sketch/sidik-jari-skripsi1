import time

import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt

from config import *


# ======================================================================
# KOMUNIKASI SENSOR
# ======================================================================
def tampilkan_port_tersedia():
    ports = serial.tools.list_ports.comports()
    print("Port tersedia:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    if not ports:
        print("  Tidak ada port. Colokkan FT232RL terlebih dahulu.")
    return ports


def verifikasi_sensor(port, baudrate):
    CMD_VERIFY_PASSWORD = bytes([0xEF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
                                 0x01, 0x00, 0x07, 0x13,
                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x1B])
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(0.1)
        print("Mengirim perintah verifikasi...")
        ser.write(CMD_VERIFY_PASSWORD)
        time.sleep(0.5)
        response = ser.read(12)
        ser.close()

        if len(response) >= 12 and response[9] == 0x00:
            print("Sensor AS608 terdeteksi dan berfungsi.")
            return True
        if len(response) >= 12:
            print(f"Sensor merespons tetapi ada galat. Kode: {response[9]}")
        else:
            print(f"Respons tidak lengkap: {response.hex() if response else 'kosong'}")
        return False
    except serial.SerialException as e:
        print(f"Galat membuka port: {e}")
        return False


def capture_fingerprint(port, baudrate):
    CMD_GEN_IMG = bytes([0xEF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
                         0x01, 0x00, 0x03, 0x01, 0x00, 0x05])
    CMD_UP_IMAGE = bytes([0xEF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
                          0x01, 0x00, 0x03, 0x0A, 0x00, 0x0E])
    try:
        ser = serial.Serial(port, baudrate, timeout=2)

        print("\n  Letakkan jari di sensor...")
        while True:
            ser.write(CMD_GEN_IMG)
            time.sleep(0.1)
            response = ser.read(12)
            if len(response) >= 10 and response[9] == 0x00:
                print("  Jari terdeteksi.")
                break
            time.sleep(0.2)

        ser.write(CMD_UP_IMAGE)
        time.sleep(0.1)

        header = ser.read(12)
        if len(header) < 12 or header[9] != 0x00:
            print("  Gagal memulai unggahan citra.")
            ser.close()
            return None

        print("  Mengunduh citra dari sensor...")
        image_data = bytearray()
        total_expected = (WIDTH * HEIGHT) // 2

        while True:
            packet = ser.read(139)
            if len(packet) < 9:
                break
            pkg_id = packet[6]
            data_len = (packet[7] << 8) | packet[8]
            if data_len > 2:
                image_data.extend(packet[9:9 + data_len - 2])
            persen = len(image_data) * 100 // total_expected
            print(f"    {len(image_data)}/{total_expected} bytes ({persen}%)", end='\r')
            if pkg_id == 0x08:
                break

        ser.close()
        print(f"\n  Data mentah diterima: {len(image_data)} bytes")

        if len(image_data) == 0:
            print("  Tidak ada data citra yang diterima.")
            return None
        return image_data

    except serial.SerialException as e:
        print(f"  Galat serial: {e}")
        return None


# ======================================================================
# PENYIMPANAN BERKAS
# ======================================================================
def simpan_raw(image_data, output_dir, nama):
    path = os.path.join(output_dir, f"{nama}.raw")
    with open(path, 'wb') as f:
        f.write(image_data)
    print(f"    RAW : {path}")
    return path


def data_ke_piksel(image_data):
    pixels = bytearray()
    for byte in image_data[:(WIDTH * HEIGHT) // 2]:
        pixels.append((byte >> 4) * 17)
        pixels.append((byte & 0x0F) * 17)
    return pixels


def konversi_raw_ke_bmp(image_data, output_dir, nama):
    pixels = data_ke_piksel(image_data)

    row_size = (WIDTH + 3) // 4 * 4
    padding = row_size - WIDTH

    pixel_data = bytearray()
    for y in range(HEIGHT - 1, -1, -1):
        row_start = y * WIDTH
        pixel_data.extend(pixels[row_start:row_start + WIDTH])
        pixel_data.extend(bytes(padding))

    file_size = 54 + 256 * 4 + len(pixel_data)
    bmp = bytearray()
    bmp.extend(b'BM')
    bmp.extend(file_size.to_bytes(4, 'little'))
    bmp.extend((0).to_bytes(4, 'little'))
    bmp.extend((54 + 256 * 4).to_bytes(4, 'little'))
    bmp.extend((40).to_bytes(4, 'little'))
    bmp.extend(WIDTH.to_bytes(4, 'little'))
    bmp.extend(HEIGHT.to_bytes(4, 'little'))
    bmp.extend((1).to_bytes(2, 'little'))
    bmp.extend((8).to_bytes(2, 'little'))
    bmp.extend((0).to_bytes(4, 'little'))
    bmp.extend(len(pixel_data).to_bytes(4, 'little'))
    bmp.extend((2835).to_bytes(4, 'little'))
    bmp.extend((2835).to_bytes(4, 'little'))
    bmp.extend((256).to_bytes(4, 'little'))
    bmp.extend((256).to_bytes(4, 'little'))
    for i in range(256):
        bmp.extend(bytes([i, i, i, 0]))
    bmp.extend(pixel_data)

    path = os.path.join(output_dir, f"{nama}.bmp")
    with open(path, 'wb') as f:
        f.write(bmp)
    print(f"    BMP : {path}  ({WIDTH}x{HEIGHT} piksel, grayscale)")
    return path


def pratinjau(image_data, judul):
    pixels = data_ke_piksel(image_data)
    arr = np.frombuffer(bytes(pixels), dtype=np.uint8).reshape(HEIGHT, WIDTH)

    plt.figure(figsize=(5, 5.6))
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.title(judul, fontsize=9)
    plt.axis('off')
    plt.tight_layout()
    print("\n  >>> Jendela pratinjau terbuka. TUTUP jendelanya untuk melanjutkan.")
    plt.show()
    plt.close('all')
    print(f"      Tingkat keabuan yang terpakai: {len(set(pixels))}")


# ======================================================================
# PEREKAMAN BERURUTAN
# ======================================================================
def rencana_perekaman():

    print("\n" + "=" * 60)
    print("  CITRA INI UNTUK APA?")
    print("=" * 60)
    print(f"  1 = INPUT DATASET   (citra galeri, {JUMLAH_DATASET} citra, "
          f"jari {NAMA_JARI[KODE_JARI_DATASET]})")
    print(f"  2 = IDENTIFIKASI    (citra uji,    {JUMLAH_IDENTIFIKASI} citra)")
    pilih = tanya_pilihan("Pilihan Anda", ['1', '2'])

    jenis = 'dataset' if pilih == '1' else 'identifikasi'
    target = JUMLAH_DATASET if jenis == 'dataset' else JUMLAH_IDENTIFIKASI

    print(f"\n--- {'INPUT DATASET' if jenis == 'dataset' else 'IDENTIFIKASI'} ---")
    kode_subjek = tanya_kode_subjek()

    if jenis == 'dataset':
        sudah = daftar_dataset(kode_subjek=kode_subjek)
        mulai = nomor_berikutnya_dataset(kode_subjek)
        contoh = nama_dataset(kode_subjek, 1)
    else:
        sudah = daftar_identifikasi(kode_subjek=kode_subjek)
        mulai = nomor_berikutnya_identifikasi(kode_subjek)
        contoh = nama_identifikasi(kode_subjek, 1)

    print(f"\n  Berkas subjek '{kode_subjek}' yang sudah ada : {len(sudah)}")
    for f in sudah:
        print(f"    - {nama_dasar(f)}")

    if mulai > target:
        print(f"\n  [PERHATIAN] Nomor urut sudah melewati {target:02d}.")
        if not tanya_ya_tidak(f"  Tetap lanjut merekam mulai nomor {mulai:02d}"):
            return None
        sisa = tanya_angka("  Berapa citra yang mau direkam", 1, 50)
    else:
        sisa = target - (mulai - 1)
        print(f"\n  Contoh nama berkas : {contoh}")
        print(f"  Mulai dari nomor   : {mulai:02d}")
        print(f"  Sisa yang direkam  : {sisa} citra")
        if not tanya_ya_tidak("  Lanjutkan"):
            return None

    return {'jenis': jenis, 'kode_subjek': kode_subjek,
            'nomor_mulai': mulai, 'sisa': sisa}


def susun_nama(rencana, nomor):
    if rencana['jenis'] == 'dataset':
        return nama_dataset(rencana['kode_subjek'], nomor)
    return nama_identifikasi(rencana['kode_subjek'], nomor)


def rekam_berurutan(rencana):

    tersimpan = []
    nomor = rencana['nomor_mulai']
    sisa = rencana['sisa']
    urutan_ke = 1

    while sisa > 0:
        nama = susun_nama(rencana, nomor)

        print("\n" + "=" * 60)
        print(f"  CITRA {urutan_ke} dari {rencana['sisa']}   ->   {nama}")
        print("=" * 60)

        if os.path.exists(os.path.join(OUTPUT_DIR, nama + '.bmp')):
            print(f"  [PERHATIAN] Berkas {nama}.bmp sudah ada.")
            if not tanya_ya_tidak("  Timpa berkas lama"):
                nomor += 1
                continue

        image_data = capture_fingerprint(PORT, BAUDRATE)
        if image_data is None:
            if tanya_ya_tidak("\n  Pengambilan gagal. Coba lagi"):
                continue
            print("\n  Perekaman dihentikan.")
            break

        pratinjau(image_data, f"{nama}\nCitra {urutan_ke}/{rencana['sisa']}")

        if not tanya_ya_tidak("\n  Citra sudah sesuai dan mau disimpan"):
            print("  Baik, citra dibuang. Ambil ulang untuk nomor yang sama.")
            continue

        print()
        simpan_raw(image_data, OUTPUT_DIR, nama)
        konversi_raw_ke_bmp(image_data, OUTPUT_DIR, nama)
        tersimpan.append(nama)

        nomor += 1
        urutan_ke += 1
        sisa -= 1

        if sisa > 0:
            print(f"\n  Tersimpan {len(tersimpan)} citra. Sisa {sisa} lagi.")
            if not tanya_ya_tidak("  Lanjut ke citra berikutnya"):
                print("  Perekaman dihentikan atas permintaan pengguna.")
                break

    return tersimpan


if __name__ == '__main__':
    print("=" * 60)
    print("AKUISISI CITRA SIDIK JARI (AS608)")
    print("=" * 60)

    tampilkan_port_tersedia()
    if not verifikasi_sensor(PORT, BAUDRATE):
        raise SystemExit("Sensor tidak siap. Program dihentikan.")

    rencana = rencana_perekaman()
    if rencana is None:
        raise SystemExit("Dibatalkan. Tidak ada berkas yang disimpan.")

    hasil = rekam_berurutan(rencana)

    print("\n" + "=" * 60)
    print(f"  SELESAI — {len(hasil)} citra tersimpan di {OUTPUT_DIR}")
    print("=" * 60)
    for n in hasil:
        print(f"  - {n}.bmp / {n}.raw")

    if hasil:
        lanjut = ("LANJUT EKSTRAKSI!")
        print(f"\n  Langkah berikutnya: {lanjut}")
