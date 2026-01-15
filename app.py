import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from scipy.spatial import distance as dist
import time
import threading
import os
import av
import sqlite3
import glob
from datetime import datetime
import pandas as pd

# ==========================================
# 1. KONFIGURASI SISTEM
# ==========================================
st.set_page_config(page_title="Final Multi-Model", layout="wide")

# Paksa CPU (Stabil untuk Threading)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

if 'session_cleared' not in st.session_state:
    tf.keras.backend.clear_session()
    st.session_state['session_cleared'] = True

DB_NAME = "data_ekspresi.db"
IMG_FOLDER = "gradcam_imgs"

if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS log_ekspresi
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, waktu TIMESTAMP,
                  total_visitor INTEGER, total_happy INTEGER, total_sad INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS log_gradcam
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, waktu TIMESTAMP,
                  object_id INTEGER, prediction TEXT, image_path TEXT)''')
    conn.commit()
    conn.close()

if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state['db_initialized'] = True

def wipe_all_data():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM log_ekspresi")
        c.execute("DELETE FROM log_gradcam")
        c.execute("DELETE FROM sqlite_sequence")
        conn.commit()
        conn.close()
    except: pass

    try:
        files = glob.glob(os.path.join(IMG_FOLDER, "*"))
        for f in files: os.remove(f)
    except: pass

    st.cache_resource.clear()
    st.rerun()

# ==========================================
# 2. LOAD MODEL (H5 & TFLITE)
# ==========================================
@st.cache_resource
def load_models_safe():
    debug_log = "--- START LOAD LOG ---\n"
    res = {}
    
    # 1. Face Detector
    try:
        res["face"] = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
        if res["face"].empty():
            res["face"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        debug_log += "✅ Face Cascade Loaded\n"
    except: debug_log += "❌ Face Cascade Failed\n"

    # 2. H5 Model (Support GradCAM)
    try:
        model_h5 = tf.keras.models.load_model("model/final_adam_happy_sad_model.h5", compile=False)
        res["h5"] = model_h5
        debug_log += "✅ Model H5 Loaded\n"
        
        # Build Grad-CAM logic
        inner_model = None
        target_layer_name = "top_activation" 
        for layer in model_h5.layers:
            if isinstance(layer, (tf.keras.models.Model, tf.keras.layers.Layer)) and "efficient" in layer.name.lower():
                inner_model = layer
                break
        
        if inner_model:
            try:
                last_conv_layer = inner_model.get_layer(target_layer_name)
                grad_model = tf.keras.models.Model(
                    inputs=inner_model.input, 
                    outputs=[last_conv_layer.output, inner_model.output]
                )
                res["grad_model"] = grad_model
                debug_log += "✅ Grad-CAM Ready.\n"
            except: res["grad_model"] = None
        else:
            try:
                last_conv = None
                for layer in reversed(model_h5.layers):
                    if len(layer.output.shape) == 4:
                        last_conv = layer
                        break
                if last_conv:
                    grad_model = tf.keras.models.Model(
                        inputs=model_h5.inputs,
                        outputs=[last_conv.output, model_h5.output]
                    )
                    res["grad_model"] = grad_model
                    debug_log += f"✅ Grad-CAM (Flat) Ready.\n"
            except: res["grad_model"] = None

    except Exception as e:
        debug_log += f"❌ H5 Load Error: {e}\n"
        res["h5"] = None
        res["grad_model"] = None

    # 3. TFLite Model (Faster, No GradCAM)
    try:
        tflite_path = "model/final_adam_happy_sad_model_int8.tflite"
        if os.path.exists(tflite_path):
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            res["tflite"] = interpreter
            res["input_details"] = interpreter.get_input_details()
            res["output_details"] = interpreter.get_output_details()
            debug_log += "✅ TFLite Model Loaded\n"
        else:
            res["tflite"] = None
            debug_log += "⚠️ TFLite file not found\n"
    except Exception as e:
        debug_log += f"❌ TFLite Error: {e}\n"
        res["tflite"] = None

    # Global Config untuk menyimpan pilihan User
    res["config"] = {"mode": "H5"} 
    res["lock"] = threading.Lock()
    return res, debug_log

MODELS, LOGS = load_models_safe()

# ==========================================
# 3. CORE LOGIC
# ==========================================
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            for col in unusedCols: self.register(inputCentroids[col])
        return self.objects

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=40)
        self.unique_ids = set()
        self.happy_count = 0
        self.sad_count = 0
        self.face_states = {}
        self.gradcam_done_ids = set()
        
        # Timer
        self.entry_times = {} 
        self.CAPTURE_DELAY = 2.0 
        self.last_save_time = time.time()
        
        # FPS & Display
        self.prev_frame_time = time.time()
        self.fps_display = "FPS: 0"
        self.fps_last_update_time = time.time()
        self.fps_update_interval = 0.5 

    def process_gradcam_fullframe(self, full_frame, face_coords, roi_input, objID, label, timestamp):
        # GradCAM hanya bisa jalan jika Model H5 ada
        grad_model = MODELS.get("grad_model")
        if grad_model is None: return
        
        try:
            (x, y, w, h) = face_coords
            with MODELS["lock"]:
                with tf.GradientTape() as tape:
                    inputs = tf.cast(roi_input, tf.float32)
                    outputs = grad_model(inputs, training=False)
                    conv_outputs, predictions = outputs[0], outputs[1]
                    loss = tf.reduce_max(predictions, axis=1)
                
                grads = tape.gradient(loss, conv_outputs)
                if grads is None: return

                output = conv_outputs[0]
                grads_val = grads[0]
                gate_f = tf.cast(output > 0, 'float32')
                gate_g = tf.cast(grads_val > 0, 'float32')
                weights = tf.reduce_mean(gate_f * gate_g * grads_val, axis=(0, 1))
                cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

                cam = cv2.resize(cam.numpy(), (96, 96))
                cam = np.maximum(cam, 0)
                heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            heatmap_resized = cv2.resize(heatmap_color, (w, h))
            full_frame_bgr = full_frame.copy()
            roi_bg = full_frame_bgr[y:y+h, x:x+w]
            blended_roi = cv2.addWeighted(roi_bg, 0.6, heatmap_resized, 0.4, 0)
            full_frame_bgr[y:y+h, x:x+w] = blended_roi
            
            cv2.rectangle(full_frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(full_frame_bgr, f"{label} (ID:{objID})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            fname = f"full_{objID}_{int(time.time())}.jpg"
            save_path = os.path.join(IMG_FOLDER, fname)
            cv2.imwrite(save_path, full_frame_bgr)
            
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO log_gradcam (waktu, object_id, prediction, image_path) VALUES (?,?,?,?)",
                      (timestamp, objID, label, fname))
            conn.commit()
            conn.close()

        except Exception as e: print(f"GC Error: {e}")

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            current_time = time.time()
            
            # --- CEK PILIHAN MODEL DARI SIDEBAR ---
            # Kita baca dari global variabel yg di-update via Streamlit sidebar
            current_mode = MODELS["config"].get("mode", "H5")

            # --- FPS LOGIC ---
            delta_time = current_time - self.prev_frame_time
            self.prev_frame_time = current_time
            fps_val = 1 / delta_time if delta_time > 0 else 0
            
            if current_time - self.fps_last_update_time > self.fps_update_interval:
                self.fps_display = f"FPS: {int(fps_val)}"
                self.fps_last_update_time = current_time
            
            cv2.putText(img, self.fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Label Model Aktif
            model_label = f"Model: {current_mode}"
            (fw, fh), _ = cv2.getTextSize(model_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(img, model_label, (img.shape[1] - fw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- DETEKSI WAJAH ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = MODELS["face"].detectMultiScale(gray, 1.3, 5)
            
            rects = []
            for (x, y, w, h) in faces: rects.append((x, y, w, h))
            objects = self.ct.update(rects)

            for (x, y, w, h) in faces:
                cX, cY = int(x + w/2), int(y + h/2)
                objID = None
                for id_num, centroid in objects.items():
                    if abs(centroid[0] - cX) < 20 and abs(centroid[1] - cY) < 20:
                        objID = id_num
                        break
                
                label = "..."
                color = (200, 200, 200)

                if objID is not None:
                    if objID not in self.entry_times:
                        self.entry_times[objID] = current_time
                    
                    duration_on_screen = current_time - self.entry_times[objID]
                    self.unique_ids.add(objID)
                    
                    roi = img[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_resized = cv2.resize(roi_rgb, (96, 96))
                    roi_input = np.expand_dims(roi_resized, axis=0).astype(np.float32)

                    score = 0.5
                    try:
                        with MODELS["lock"]:
                            # === PERCABANGAN MODEL ===
                            if current_mode == "H5":
                                if MODELS["h5"] is not None:
                                    # H5 Prediction
                                    pred = MODELS["h5"](roi_input, training=False)
                                    score = pred.numpy()[0][0]
                                else:
                                    label = "H5 Err"
                            
                            elif current_mode == "TFLite":
                                if MODELS["tflite"] is not None:
                                    # TFLite Prediction
                                    interpreter = MODELS["tflite"]
                                    input_details = MODELS["input_details"]
                                    output_details = MODELS["output_details"]
                                    
                                    interpreter.set_tensor(input_details[0]['index'], roi_input)
                                    interpreter.invoke()
                                    output_data = interpreter.get_tensor(output_details[0]['index'])
                                    score = output_data[0][0]
                                else:
                                    label = "Lite Err"
                        
                        # Tentukan Label
                        if "Err" not in label:
                            label = "Happy" if score < 0.5 else "Sad"
                        
                        color = (0, 255, 0) if label == "Happy" else (0, 0, 255)

                        # Update Statistik
                        if objID not in self.face_states:
                            self.face_states[objID] = label
                            if label == "Happy": self.happy_count += 1
                            else: self.sad_count += 1

                        # === LOGIKA GRAD-CAM ===
                        # Grad-CAM hanya aktif jika mode H5 (TFLite tidak support gradients)
                        if current_mode == "H5":
                            if (objID not in self.gradcam_done_ids) and (duration_on_screen > self.CAPTURE_DELAY):
                                self.gradcam_done_ids.add(objID)
                                t_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                coords = (x, y, w, h)
                                threading.Thread(target=self.process_gradcam_fullframe, 
                                                args=(img.copy(), coords, roi_input.copy(), objID, label, t_str)).start()
                            
                            if objID not in self.gradcam_done_ids:
                                bar_width = int(min(duration_on_screen / self.CAPTURE_DELAY, 1.0) * w)
                                cv2.rectangle(img, (x, y-15), (x+bar_width, y-12), (0, 255, 255), -1)
                        else:
                            # Mode TFLite
                             pass # Tidak ada loading bar / GradCAM

                    except Exception as e:
                        print(f"Pred Error: {e}")

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                txt = f"{label} {objID}" if objID else label
                cv2.putText(img, txt, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if time.time() - self.last_save_time > 180:
                self.save_stats()

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except: return frame

    def save_stats(self):
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO log_ekspresi (waktu, total_visitor, total_happy, total_sad) VALUES (?,?,?,?)",
                      (datetime.now(), len(self.unique_ids), self.happy_count, self.sad_count))
            conn.commit()
            conn.close()
            self.unique_ids = set()
            self.happy_count = 0
            self.sad_count = 0
            self.last_save_time = time.time()
            self.entry_times = {}
        except: pass

# ==========================================
# 4. UI STREAMLIT
# ==========================================
st.sidebar.title("🔧 Control Panel")

# --- DROPDOWN PILIHAN MODEL ---
# Pilihan user akan disimpan di variable global MODELS["config"]
model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ("H5", "TFLite"),
    help="H5 untuk Grad-CAM (Lebih Berat). TFLite untuk Kecepatan (Tanpa Grad-CAM)."
)
MODELS["config"]["mode"] = model_choice

st.sidebar.warning("Tombol di bawah akan menghapus semua data!")
if st.sidebar.button("HARD RESET & WIPE DATA"):
    wipe_all_data()

st.title(f"📹 Deteksi Ekspresi - Mode: {model_choice}")

ctx = webrtc_streamer(
    key="multi-model-fixed",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Log Data (Auto Update)")
    table_placeholder = st.empty() 

with col2:
    st.subheader("🖼️ Galeri (Auto Update - All)")
    if model_choice == "TFLite":
        st.warning("⚠️ Grad-CAM tidak tersedia di mode TFLite.")
    gallery_placeholder = st.empty() 

# LOOP UI UPDATE
if ctx.state.playing:
    while True:
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql("SELECT * FROM log_ekspresi ORDER BY id DESC LIMIT 10", conn)
            conn.close()
            with table_placeholder.container():
                st.dataframe(df, height=300, use_container_width=True)
        except: pass

        try:
            conn = sqlite3.connect(DB_NAME)
            rows = conn.execute("SELECT * FROM log_gradcam ORDER BY id DESC").fetchall()
            conn.close()
            
            with gallery_placeholder.container():
                if not rows: st.info("Belum ada capture Grad-CAM.")
                else:
                    cols_grid = st.columns(3)
                    for i, row in enumerate(rows):
                        fpath = os.path.join(IMG_FOLDER, row[4])
                        if os.path.exists(fpath):
                            with cols_grid[i % 3]:
                                st.image(fpath, caption=f"ID:{row[2]} | {row[3]}\n{row[1]}")
        except: pass
        
        time.sleep(2)
else:
    st.info("Nyalakan video untuk melihat update data secara real-time.")