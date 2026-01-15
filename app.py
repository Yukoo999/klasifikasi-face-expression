import streamlit as st
# 1. Pastikan import ini lengkap
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
import pandas as pd
from datetime import datetime

# ==========================================
# 0. PAGE CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="Face Expression", layout="wide")

# Paksa CPU agar tidak crash di Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Buat folder temp jika belum ada
IMG_FOLDER = "temp_gradcam"
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# CSS Custom
st.markdown("""
    <style>
    div.stWebrtc {
        background-color: #1a1c24;
        border: 1px solid #444;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DEFINISI CONFIG WEBRTC (Ditaruh di atas)
# ==========================================
# Ini harus didefinisikan SEBELUM webrtc_streamer dipanggil
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_models_safe():
    res = {}
    # Face Detector
    try:
        res["face"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except: pass

    # H5 Model
    try:
        model_path = "model/final_adam_happy_sad_model.h5"
        if os.path.exists(model_path):
            res["h5"] = tf.keras.models.load_model(model_path, compile=False)
        else:
            res["h5"] = None
    except:
        res["h5"] = None

    # TFLite
    try:
        tflite_path = "model/final_adam_happy_sad_model_int8.tflite"
        if not os.path.exists(tflite_path): 
            tflite_path = "model/final_adam_happy_sad_model.tflite"
            
        if os.path.exists(tflite_path):
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            res["tflite"] = interpreter
            res["input_details"] = interpreter.get_input_details()
            res["output_details"] = interpreter.get_output_details()
        else: 
            res["tflite"] = None
    except: 
        res["tflite"] = None

    res["lock"] = threading.Lock()
    res["config"] = {"mode": "H5"}
    return res

MODELS = load_models_safe()

# ==========================================
# 3. CLASS LOGIC (Tracking & Processing)
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

# Pastikan class ini mewarisi VideoProcessorBase dengan benar
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=30)
        self.unique_ids = set()
        self.happy_count = 0
        self.sad_count = 0
        self.face_states = {}
        self.temp_logs = []
        self.entry_times = {}

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            
            # Deteksi Wajah
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = MODELS["face"].detectMultiScale(gray, 1.3, 5)
            
            rects = []
            for (x, y, w, h) in faces: 
                rects.append((x, y, w, h))

            objects = self.ct.update(rects)
            mode = MODELS["config"].get("mode", "H5")
            
            for (x, y, w, h) in faces:
                cX, cY = int(x + w/2), int(y + h/2)
                objID = None
                # Matching ID
                for id_num, centroid in objects.items():
                    if abs(centroid[0] - cX) < 50 and abs(centroid[1] - cY) < 50:
                        objID = id_num
                        break
                
                label = "..."
                color = (200, 200, 200)

                # Predict Logic
                if objID is not None:
                    self.unique_ids.add(objID)
                    roi = img[y:y+h, x:x+w]
                    
                    try:
                        # Preprocessing
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_resized = cv2.resize(roi_rgb, (96, 96))
                        roi_input = np.expand_dims(roi_resized, axis=0).astype(np.float32)
                        
                        score = 0.5
                        # Thread Lock untuk Model
                        with MODELS["lock"]:
                            if mode == "TFLite" and MODELS["tflite"]:
                                interpreter = MODELS["tflite"]
                                interpreter.set_tensor(MODELS["input_details"][0]['index'], roi_input)
                                interpreter.invoke()
                                score = interpreter.get_tensor(MODELS["output_details"][0]['index'])[0][0]
                            elif MODELS["h5"]:
                                score = MODELS["h5"](roi_input, training=False).numpy()[0][0]
                        
                        label = "Happy" if score < 0.5 else "Sad"
                        color = (0, 255, 0) if label == "Happy" else (0, 0, 255)
                        
                        # Logging sederhana
                        if objID not in self.face_states:
                            self.face_states[objID] = label
                            if label == "Happy": self.happy_count += 1
                            else: self.sad_count += 1
                            
                            self.temp_logs.append({
                                "waktu": datetime.now().strftime("%H:%M:%S"),
                                "id": objID,
                                "emosi": label
                            })

                    except Exception as e:
                        print(f"Error predict: {e}")

                cv2.putText(img, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Frame Error: {e}")
            return frame

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("📹 Final Expression Detection")

model_choice = st.sidebar.radio("Pilih Model", ["H5", "TFLite"])
MODELS["config"]["mode"] = model_choice

col1, col2 = st.columns([2, 1])

with col1:
    # PERHATIKAN: video_processor_factory=VideoProcessor (Tanpa tanda kutip, tanpa kurung)
    ctx = webrtc_streamer(
        key="detection-stream-fixed",
        video_processor_factory=VideoProcessor, 
        mode="sendrecv",
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.write("### Live Logs")
    log_placeholder = st.empty()

# UI Update Loop
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            try:
                logs = list(ctx.video_processor.temp_logs)
                if logs:
                    df = pd.DataFrame(logs)
                    log_placeholder.dataframe(df.iloc[::-1].head(10), use_container_width=True)
            except: pass
        time.sleep(1)
