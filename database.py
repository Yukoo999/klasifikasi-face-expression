import sqlite3
from datetime import datetime

DB_NAME = "data_ekspresi.db"

def init_db():
    """Membuat tabel log data dan log gradcam jika belum ada"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Tabel Log Statistik (Visitor/Happy/Sad)
    c.execute('''CREATE TABLE IF NOT EXISTS log_ekspresi
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  waktu TIMESTAMP,
                  total_visitor INTEGER,
                  total_happy INTEGER,
                  total_sad INTEGER)''')

    # Tabel Log GradCAM (Menyimpan path gambar)
    c.execute('''CREATE TABLE IF NOT EXISTS log_gradcam
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  waktu TIMESTAMP,
                  object_id INTEGER,
                  prediction TEXT,
                  image_path TEXT)''')
                  
    conn.commit()
    conn.close()

def simpan_data(visitor, happy, sad):
    """Menyimpan data statistik (interval 3 menit)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO log_ekspresi (waktu, total_visitor, total_happy, total_sad) VALUES (?, ?, ?, ?)",
              (waktu_sekarang, visitor, happy, sad))
    conn.commit()
    conn.close()
    print(f"[DATABASE] Statistik tersimpan: {waktu_sekarang}")

def simpan_gradcam_db(obj_id, label, img_path):
    """Menyimpan record GradCAM ke database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO log_gradcam (waktu, object_id, prediction, image_path) VALUES (?, ?, ?, ?)",
              (waktu_sekarang, obj_id, label, img_path))
    conn.commit()
    conn.close()
    print(f"[DATABASE] GradCAM tersimpan untuk ID {obj_id}")

def ambil_data_terakhir():
    """Mengambil 10 data statistik terakhir"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM log_ekspresi ORDER BY id DESC LIMIT 10")
    data = c.fetchall()
    conn.close()
    return data

def ambil_gradcam_terakhir():
    """Mengambil 5 data GradCAM terakhir untuk ditampilkan di web"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM log_gradcam ORDER BY id DESC LIMIT 5")
    data = c.fetchall()
    conn.close()
    return data