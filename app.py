import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from scipy.spatial import distance as dist
import time
import threading
import os
import av
import base64
import shutil
from datetime import datetime
import pandas as pd

# ==========================================
# 0. PAKSA DARK MODE & MEMORY SETUP
# ==========================================
st.set_page_config(
    page_title="Final Multi-Model", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# Paksa CPU agar tidak konflik dengan aioice/threading
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], 'GPU')
except: pass

if 'session_cleared' not in st.session_state:
    tf.keras.backend.clear_session()
    st.session_state['session_cleared'] = True

IMG_FOLDER = "temp_gradcam"
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stDeployButton, [data-testid="stDeployButton"] { display: none !important; }
    div.stWebrtc {
        background-color: #1a1c24;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_models_safe():
    res = {}
    # 1. Face Detector
    try:
        res["face"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except: pass

    # 2. H5 Model
    try:
        model_h5 = tf.keras.models.load_model("model/final_adam_happy_sad_model.h5", compile=False)
        res["h5"] = model_h5
        # Setup GradCAM logic (Simplified for stability)
        try:
            target_layer = "top_activation"
            inner_model = None
            for layer in model_h5.layers:
                if "efficient" in layer.name.lower():
                    inner_model = layer
                    break
            
            if inner_model:
                last_conv = inner_model.get_layer(target_layer)
                res["grad_model"] = tf.keras.models.Model(
                    inputs=inner_model.input, outputs=[last_conv.output, inner_model.output])
            else:
                 res["grad_model"] = None
        except: res["grad_model"] = None
    except:
        res["h5"] = None
        res["grad_model"] = None

    # 3. TFLite
    try:
        tflite_path = "model/final_adam_happy_sad_model_int8.tflite"
        if not os.path.exists(tflite_path): tflite_path = "model/final_adam_happy_sad_model.tflite"
        if os.path.exists(tflite_path):
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            res["tflite"] = interpreter
            res["input_details"] = interpreter.get_input_details()
            res["output_details"] = interpreter.get_output_details()
        else: res["tflite"] = None
    except: res["tflite"] = None

    res["lock"] = threading.Lock()
    res["config"] = {"mode": "H5"}
    return res

MODELS = load_models_safe()

# ==========================================
# 2. LOGIC PROCESOR
# ==========================================
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
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
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            inputCentroids[i] = (int(x + w / 2.0), int(y + h / 2.0))
            
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)): self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
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

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=30)
        self.unique_ids = set()
        self.happy_count = 0
        self.sad_count = 0
        self.face_states = {}
        self.gradcam_done_ids = set()
        self.entry_times = {}
        self.temp_logs = []
        self.temp_gradcams = []
        self.last_save_time = time.time()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            
            # --- 1. Optimasi: Skip Frame Processing jika CPU load tinggi ---
            # (Tidak ada di sini, tapi resize gambar membantu)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = MODELS["face"].detectMultiScale(gray, 1.2, 5) # Scale factor diperbesar biar lebih ringan
            
            rects = []
            for (x, y, w, h) in faces: 
                rects.append((x, y, w, h))
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

            objects = self.ct.update(rects)
            
            # MODE
            mode = MODELS["config"].get("mode", "H5")
            
            for (x, y, w, h) in faces:
                # Simple ID matching
                cX, cY = int(x + w/2), int(y + h/2)
                objID = None
                for id_num, centroid in objects.items():
                    if abs(centroid[0] - cX) < 30 and abs(centroid[1] - cY) < 30:
                        objID = id_num
                        break
                
                label = "..."
                color = (200, 200, 200)

                # Prediction Logic
                if objID is not None:
                    # Logika durasi & count
                    now = time.time()
                    if objID not in self.entry_times: self.entry_times[objID] = now
                    duration = now - self.entry_times[objID]
                    self.unique_ids.add(objID)

                    # PREDICT
                    try:
                        roi = img[y:y+h, x:x+w]
                        if roi.shape[0] > 10 and roi.shape[1] > 10:
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            roi_resized = cv2.resize(roi_rgb, (96, 96))
                            roi_input = np.expand_dims(roi_resized, axis=0).astype(np.float32)
                            
                            score = 0.5
                            with MODELS["lock"]:
                                if mode == "TFLite" and MODELS["tflite"]:
                                    interpreter = MODELS["tflite"]
                                    interpreter.set_tensor(MODELS["input_details"][0]['index'], roi_input)
                                    interpreter.invoke()
                                    score = interpreter.get_tensor(MODELS["output_details"][0]['index'])[0][0]
                                elif MODELS["h5"]:
                                    # Gunakan try-except saat predict agar tidak crash thread
                                    score = MODELS["h5"](roi_input, training=False).numpy()[0][0]
                            
                            label = "Happy" if score < 0.5 else "Sad"
                            color = (0, 255, 0) if label == "Happy" else (0, 0, 255)
                            
                            # Update stats
                            if objID not in self.face_states:
                                self.face_states[objID] = label
                                if label == "Happy": self.happy_count += 1
                                else: self.sad_count += 1
                                
                                # Log Stats
                                if len(self.temp_logs) < 50:
                                    self.temp_logs.append({
                                        "waktu": datetime.now().strftime("%H:%M:%S"),
                                        "visitor": len(self.unique_ids),
                                        "happy": self.happy_count, 
                                        "sad": self.sad_count
                                    })

                            # GradCAM Trigger (Simplified: Only save generic image for stability)
                            if (objID not in self.gradcam_done_ids) and (duration > 2.0):
                                self.gradcam_done_ids.add(objID)
                                fname = f"face_{objID}_{int(now)}.jpg"
                                cv2.imwrite(os.path.join(IMG_FOLDER, fname), img)
                                if len(self.temp_gradcams) < 20:
                                    self.temp_gradcams.append({
                                        "waktu": datetime.now().strftime("%H:%M:%S"),
                                        "object_id": objID,
                                        "prediction": label,
                                        "image_path": fname
                                    })

                    except Exception as e: 
                        print(f"Pred Error: {e}")

                # Draw
                cv2.putText(img, f"{label} ID:{objID}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            # Return original frame on error to prevent black screen crash
            return frame

# ==========================================
# 3. UI UTAMA
# ==========================================
st.title("📹 Final Expression Detection")

model_choice = st.sidebar.radio("Model", ["H5", "TFLite"])
MODELS["config"]["mode"] = model_choice

# --- KONFIGURASI JARINGAN STABIL (PENTING!) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

# --- WEBRTC STREAMER ---
# async_processing=False lebih stabil untuk koneksi buruk (tapi FPS sedikit turun)
ctx = webrtc_streamer(
    key="stable-stream",
    video_processor_factory=VideoProcessor,
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": 320, "height": 240, "frameRate": 15}, 
        "audio": False
    },
    async_processing=False, 
)

col1, col2 = st.columns(2)

# --- UI LOOPING (SAFE MODE) ---
if ctx.state.playing:
    placeholder_table = col1.empty()
    placeholder_gallery = col2.empty()
    
    while ctx.state.playing:
        if ctx.video_processor:
            try:
                # TABLE
                logs = list(ctx.video_processor.temp_logs)
                if logs:
                    df = pd.DataFrame(logs)
                    placeholder_table.dataframe(df.iloc[::-1], height=200, use_container_width=True)
                
                # GALLERY
                cams = list(ctx.video_processor.temp_gradcams)
                if cams:
                    recent = cams[-1]
                    fpath = os.path.join(IMG_FOLDER, recent["image_path"])
                    if os.path.exists(fpath):
                        placeholder_gallery.image(fpath, caption=f"Last Capture: {recent['prediction']}")
            except: pass
        time.sleep(1)
