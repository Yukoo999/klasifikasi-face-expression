import tensorflow as tf
import os

MODEL_PATH = "model/final_adam_happy_sad_model.h5"

print(f"🔍 Sedang membedah model: {MODEL_PATH} ...")
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("\n" + "="*60)
        print("HASIL DIAGNOSA STRUKTUR MODEL")
        print("="*60)

        # 1. Cek Apakah ada Layer Bersarang (Nested Model)
        nested_layer = None
        for layer in model.layers:
            # Cek jika layer ini adalah Model/Functional (EfficientNet biasanya begini)
            if isinstance(layer, (tf.keras.Model, tf.keras.layers.Layer)) and hasattr(layer, 'layers'):
                print(f"📦 DITEMUKAN LAYER BERSARANG: '{layer.name}' (Tipe: {layer.__class__.__name__})")
                nested_layer = layer
                
                # Cek isi dalamnya
                last_inner_conv = None
                for inner_layer in layer.layers:
                    if 'conv' in inner_layer.name.lower():
                        last_inner_conv = inner_layer.name
                    if 'top_activation' in inner_layer.name: # Khas EfficientNet
                        last_inner_conv = inner_layer.name
                
                if last_inner_conv:
                    print(f"   👉 Layer Conv Terakhir di dalam '{layer.name}' adalah: '{last_inner_conv}'")
                    print(f"\n✅ KESIMPULAN: Anda menggunakan Model BERSARANG.")
                    print(f"   Nama Layer Luar : '{layer.name}'")
                    print(f"   Nama Layer Dalam: '{last_inner_conv}'")
                    break
        
        # 2. Jika Tidak Ada Layer Bersarang (Simple CNN)
        if not nested_layer:
            last_conv = None
            for layer in model.layers:
                if 'conv' in layer.name.lower():
                    last_conv = layer.name
            
            print("📦 DITEMUKAN STRUKTUR DATAR (Simple CNN).")
            if last_conv:
                print(f"✅ KESIMPULAN: Layer Conv Terakhir adalah '{last_conv}'")
            else:
                print("❌ TIDAK ADA LAYER CONV. Cek kembali model Anda.")

        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ ERROR SAAT LOAD: {e}")
else:
    print(f"❌ File tidak ditemukan: {MODEL_PATH}")