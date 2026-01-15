import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from scipy.spatial import distance as dist
import time
import threading
import os
import av
import glob
from datetime import datetime
import pandas as pd
import shutil
import base64
from twilio.rest import Client

# ==========================================
# 0. PAKSA DARK MODE (AUTO-CONFIG)
# ==========================================
def setup_forced_dark_theme():
    try:
        if not os.path.exists(".streamlit"):
            os.makedirs(".streamlit")
        
        config_path = ".streamlit/config.toml"
        config_content = """
[theme]
base="dark"
primaryColor="#F63366"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"
        """
        
        write_file = False
        if not os.path.exists(config_path):
            write_file = True
        else:
            with open(config_path, "r") as f:
                if "base=\"dark\"" not in f.read():
                    write_file = True
        
        if write_file:
            with open(config_path, "w") as f:
                f.write(config_content)
    except Exception as e:
        print(f"Gagal set theme: {e}")

setup_forced_dark_theme()

# ==========================================
# 1. KONFIGURASI SISTEM
# ==========================================
st.set_page_config(
    page_title="Final Multi-Model", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- FUNGSI UNTUK MEMBACA GAMBAR LOKAL KE BASE64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

local_image_path = "bg.jpg" 
bin_str = get_base64_of_bin_file(local_image_path)

if bin_str:
    bg_image_css = f'url("data:image/jpeg;base64,{bin_str}")'
else:
    bg_image_css = 'url("https://www.transparenttextures.com/patterns/cubes.png")'

# --- CUSTOM CSS (FINAL FIXES: TOOLTIP & DEPLOY BUTTON) ---
custom_css = f"""
    <style>
    /* 1. SETUP DASAR & BACKGROUND */
    #MainMenu {{visibility: hidden;}} 
    footer {{visibility: hidden;}} 
    [data-testid="stDecoration"] {{display: none;}} 
    [data-testid="stHeader"] {{background-color: transparent;}}

    /* --- PERBAIKAN HAPUS TOMBOL DEPLOY --- */
    /* Menggunakan selector yang lebih kuat untuk menghilangkan tulisan Deploy */
    .stDeployButton, [data-testid="stDeployButton"] {{
        display: none !important;
        visibility: hidden !important;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-image: {bg_image_css};
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* 2. SIDEBAR PERMANEN & DARK MODE */
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    @media (min-width: 992px) {{
        section[data-testid="stSidebar"] {{
            display: block !important;
            visibility: visible !important;
            width: 300px !important;
            transform: none !important;
        }}
        .main .block-container {{
            margin-left: 20px !important; 
            max-width: 100% !important;
            padding-left: 2rem !important;
        }}
    }}

    [data-testid="stSidebar"] {{
        background-color: #1a1c24; 
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 99999;
    }}
    [data-testid="stSidebar"] * {{
        color: #e0e0e0 !important;
    }}

    /* 3. FIX WEBRTC BOX */
    div.stWebrtc {{
        background-color: rgba(30, 30, 30, 0.9) !important;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    div.stWebrtc > div {{ background-color: transparent !important; }}
    div.stWebrtc label, div.stWebrtc span, div.stWebrtc div {{ color: white !important; }}
    div.stWebrtc button {{
        background-color: #F63366 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }}
    
    /* --- PERBAIKAN TOOLTIP (TANDA TANYA) --- */
    /* 1. Paksa ikon tanda tanya menjadi putih terang */
    [data-testid="stTooltipIcon"] svg {{
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
    }}
    
    /* 2. Pastikan kotak popup yang muncul backgroundnya gelap dan teksnya putih */
    div[data-testid="stTooltipHoverTarget"] > div,
    .stTooltip {{
        background-color: #262730 !important; /* Background gelap */
        color: #ffffff !important; /* Teks putih */
        border: 1px solid rgba(255,255,255,0.2) !important;
        padding: 10px !important;
        width: auto !important;
        min-width: 300px !important;
        max-width: 500px !important;
    }}
    
    /* Pastikan teks di dalam popup benar-benar putih */
    div[role="tooltip"],
    div[role="tooltip"] p {{
        background-color: transparent !important;
        color: #ffffff !important;
    }}

    /* 5. COMPONENTS LAIN */
    [data-testid="stDataFrame"] {{
        background-color: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: #262730 !important;
        color: white !important;
        border-color: #4b4b4b !important;
    }}
    .stSelectbox svg {{ fill: white !important; }}
    .stAlert {{
        background-color: rgba(38, 39, 48, 0.9) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1);
    }}

    /* TEKS GLOBAL */
    h1, h2, h3, h4, h5, h6, p, li, span, div, label {{
        color: #FFFFFF !important;
    }}

    /* FOOTER */
    .custom-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(14, 17, 23, 0.95);
        color: #b0b0b0 !important;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 99999;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-family: sans-serif;
    }}
    .custom-footer b {{ color: #F63366 !important; }}
    .main .block-container {{ padding-bottom: 80px; }}
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Paksa CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

if 'session_cleared' not in st.session_state:
    tf.keras.backend.clear_session()
    st.session_state['session_cleared'] = True

IMG_FOLDER = "temp_gradcam"
def clean_and_init_folder():
    if os.path.exists(IMG_FOLDER):
        try: shutil.rmtree(IMG_FOLDER) 
        except: pass
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER) 

if 'init_done' not in st.session_state:
    clean_and_init_folder()
    st.session_state['init_done'] = True

# ==========================================
# 2. LOAD MODEL
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
    except: pass

    # 2. H5 Model
    try:
        model_h5 = tf.keras.models.load_model("model/final_adam_happy_sad_model.h5", compile=False)
        res["h5"] = model_h5
        
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
            except: res["grad_model"] = None
    except:
        res["h5"] = None
        res["grad_model"] = None

    # 3. TFLite Model
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

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=40)
        self.unique_ids = set()
        self.happy_count = 0
        self.sad_count = 0
        self.face_states = {}
        self.gradcam_done_ids = set()
        
        self.entry_times = {} 
        self.CAPTURE_DELAY = 2.0 
        self.last_save_time = time.time()
        
        self.prev_frame_time = time.time()
        self.fps_display = "FPS: 0"
        self.fps_last_update_time = time.time()
        self.fps_update_interval = 0.5 
        
        self.temp_logs = []       
        self.temp_gradcams = []   

    def process_gradcam_fullframe(self, full_frame, face_coords, roi_input, objID, label, timestamp):
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
            
            self.temp_gradcams.append({
                "waktu": timestamp,
                "object_id": objID,
                "prediction": label,
                "image_path": fname
            })

        except Exception as e: print(f"GC Error: {e}")

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            current_time = time.time()
            
            current_mode = MODELS["config"].get("mode", "H5")

            delta_time = current_time - self.prev_frame_time
            self.prev_frame_time = current_time
            fps_val = 1 / delta_time if delta_time > 0 else 0
            
            if current_time - self.fps_last_update_time > self.fps_update_interval:
                self.fps_display = f"FPS: {int(fps_val)}"
                self.fps_last_update_time = current_time
            
            cv2.putText(img, self.fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            model_label = f"Model: {current_mode}"
            (fw, fh), _ = cv2.getTextSize(model_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(img, model_label, (img.shape[1] - fw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                            used_model = False
                            if current_mode == "TFLite" and MODELS["tflite"] is not None:
                                interpreter = MODELS["tflite"]
                                input_details = MODELS["input_details"]
                                output_details = MODELS["output_details"]
                                interpreter.set_tensor(input_details[0]['index'], roi_input)
                                interpreter.invoke()
                                output_data = interpreter.get_tensor(output_details[0]['index'])
                                score = output_data[0][0]
                                used_model = True
                            elif MODELS["h5"] is not None:
                                pred = MODELS["h5"](roi_input, training=False)
                                score = pred.numpy()[0][0]
                                used_model = True

                        if used_model:
                            label = "Happy" if score < 0.5 else "Sad"
                            color = (0, 255, 0) if label == "Happy" else (0, 0, 255)

                            if objID not in self.face_states:
                                self.face_states[objID] = label
                                if label == "Happy": self.happy_count += 1
                                else: self.sad_count += 1

                            if (objID not in self.gradcam_done_ids) and (duration_on_screen > self.CAPTURE_DELAY):
                                if MODELS.get("grad_model") is not None:
                                    self.gradcam_done_ids.add(objID)
                                    t_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    coords = (x, y, w, h)
                                    threading.Thread(target=self.process_gradcam_fullframe, 
                                                    args=(img.copy(), coords, roi_input.copy(), objID, label, t_str)).start()
                            
                            if objID not in self.gradcam_done_ids:
                                bar_width = int(min(duration_on_screen / self.CAPTURE_DELAY, 1.0) * w)
                                cv2.rectangle(img, (x, y-15), (x+bar_width, y-12), (0, 255, 255), -1)

                    except: pass

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                txt = f"{label} {objID}" if objID else label
                cv2.putText(img, txt, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if time.time() - self.last_save_time > 180:
                self.save_stats()

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except: return frame

    def save_stats(self):
        self.temp_logs.append({
            "waktu": datetime.now(),
            "total_visitor": len(self.unique_ids),
            "total_happy": self.happy_count,
            "total_sad": self.sad_count
        })
        self.unique_ids = set()
        self.happy_count = 0
        self.sad_count = 0
        self.last_save_time = time.time()
        self.entry_times = {}

# ==========================================
# 4. UI STREAMLIT
# ==========================================
st.sidebar.title("🔧 Control Panel")

model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ("H5", "TFLite"),
    help="H5 untuk Grad-CAM. TFLite untuk Kecepatan."
)
MODELS["config"]["mode"] = model_choice

st.title(f"📹 Deteksi Ekspresi - Mode: {model_choice}")

def get_ice_servers():
    # Coba ambil dari Twilio jika ada di secrets
    try:
        account_sid = st.secrets["twilio"]["account_sid"]
        auth_token = st.secrets["twilio"]["auth_token"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        # Fallback ke Google STUN (Mungkin mental di Cloud, tapi jalan di localhost)
        st.warning(f"Menggunakan Google STUN (koneksi mungkin tidak stabil). Error Twilio: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": get_ice_servers()}
)

ctx = webrtc_streamer(
    key="multi-model-fixed",
    video_processor_factory=VideoProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"width": 320, "height": 240, "frameRate": 15}, 
        "audio": False
    },
    async_processing=False,
)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Log Data")
    table_placeholder = st.empty() 

with col2:
    st.subheader("🖼️ Galeri")
    if model_choice == "TFLite":
        st.warning("⚠️ Grad-CAM berjalan di background.")
    gallery_placeholder = st.empty() 

# --- FOOTER ---
st.markdown("""
    <div class="custom-footer">
        Created by <b>Bassam & Yukoo</b> | © 2026 Klasifikasi Face Expression
    </div>
""", unsafe_allow_html=True)
# --------------

# ==========================================
# FIX UTAMA: LOOPING YANG BENAR
# ==========================================
if ctx.state.playing:
    # Cek apakah processor sudah siap
    if ctx.video_processor:
        # Loop ini HANYA berjalan selama status playing = True
        # Jika stop ditekan, loop berhenti, script selesai, tidak hang.
        while ctx.state.playing:
            if not ctx.video_processor: break # Safety check double
            
            try:
                # Gunakan list() untuk meng-copy data agar aman dari threading conflict
                logs = list(ctx.video_processor.temp_logs)
                if logs:
                    df = pd.DataFrame(logs)
                    df = df.iloc[::-1] 
                    with table_placeholder.container():
                        st.dataframe(df, height=300, use_container_width=True)
                else:
                     with table_placeholder.container():
                        st.info("Belum ada data statistik.")
            except: pass

            try:
                rows = list(ctx.video_processor.temp_gradcams)
                if not rows:
                    with gallery_placeholder.container():
                         st.info("Belum ada capture Grad-CAM.")
                else:
                    rows_reversed = rows[::-1]
                    with gallery_placeholder.container():
                        cols_grid = st.columns(3)
                        for i, row in enumerate(rows_reversed):
                            fpath = os.path.join(IMG_FOLDER, row["image_path"])
                            if os.path.exists(fpath):
                                with cols_grid[i % 3]:
                                    st.image(fpath, caption=f"ID:{row['object_id']} | {row['prediction']}\n{row['waktu']}")
            except: pass
        
            time.sleep(1) # Refresh UI setiap 1 detik
