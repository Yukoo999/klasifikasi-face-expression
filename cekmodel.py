import tensorflow as tf
import os

# Kita skip print versi keras karena bikin error di TF 2.15
print(f"✅ TensorFlow Terdeteksi: {tf.__version__}")

path_h5 = "model/final_adam_happy_sad_model.h5"

if os.path.exists(path_h5):
    try:
        print(f"📂 Mencoba load model dari: {path_h5}")
        
        # compile=False agar tidak perlu load optimizer lama
        model = tf.keras.models.load_model(path_h5, compile=False)
        
        print("=========================================")
        print("🎉 SUKSES BESAR! Model H5 berhasil dibaca.")
        print(f"Input Shape Model: {model.input_shape}")
        print("=========================================")
        print("Sekarang Anda aman untuk menjalankan 'streamlit run app.py'")
        
    except Exception as e:
        print("❌ GAGAL LOAD MODEL.")
        print("Pesan Error Detail:", e)
else:
    print("⚠️ File model tidak ditemukan! Cek nama folder/file.")