import tensorflow as tf
import os

# Ganti path ini sesuai lokasi model Anda
MODEL_PATH = "model/final_adam_happy_sad_model.h5" 

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("\n" + "="*40)
        print("DAFTAR LAYER DI MODEL ANDA:")
        print("="*40)
        
        found = False
        conv_layers = []
        for layer in model.layers:
            print(f"Name: {layer.name} \t| Type: {layer.__class__.__name__}")
            if "Conv" in layer.__class__.__name__:
                conv_layers.append(layer.name)
        
        print("="*40)
        if conv_layers:
            print(f"✅ CANDIDATE LAYER TERAKHIR: '{conv_layers[-1]}'")
            print("Silakan copy nama di atas ke dalam app.py")
        else:
            print("❌ TIDAK DITEMUKAN LAYER CONV2D! Apakah ini model CNN?")
        print("="*40 + "\n")
            
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"File model tidak ditemukan di: {MODEL_PATH}")