#!/usr/bin/env python
# app_insight_webrtc.py
"""
Face-Recognition berbasis InsightFace (ArcFace R100 512-D) + WebRTC
Tiga langkah:
1) pip install -r requirements.txt
2) streamlit run app_insight_webrtc.py
3) Klik Start Camera → Allow kamera
"""

import os, pickle, av, cv2, numpy as np, streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from insightface.app import FaceAnalysis

# ────────────────────────────────────────────────────────────
#  konfigurasi file database
# ────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
EMB_FILE = os.path.join(DATA, "embeddings.npy")
NAME_FILE = os.path.join(DATA, "names.pkl")

# ────────────────────────────────────────────────────────────
#  load database jika ada
# ────────────────────────────────────────────────────────────
if os.path.exists(EMB_FILE):
    db_emb = list(np.load(EMB_FILE))
    db_names = pickle.load(open(NAME_FILE, "rb"))
else:
    db_emb, db_names = [], []

# ────────────────────────────────────────────────────────────
#  load InsightFace sekali (cache)
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="💡 Memuat model ArcFace… (~70 MB)")
def load_model():
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))   # -1 ➜ CPU
    return app
insight = load_model()

# ────────────────────────────────────────────────────────────
#  UI Streamlit
# ────────────────────────────────────────────────────────────
st.title("🤖 Face Recognition – InsightFace WebRTC")
MODE = st.radio("Pilih mode", ["Daftarkan Wajah", "Recognize Wajah"])

if MODE == "Daftarkan Wajah":
    NAME = st.text_input("Nama lengkap / ID", "")
    TARGET = st.slider("Jumlah sampel", 20, 200, 80, 10)
    st.info("Tekan **Start Camera** lalu hadapkan wajah hingga progres mencapai target.")
else:
    NAME, TARGET = None, None
    st.info("Tekan **Start Camera** untuk mengenali wajah.")

# ────────────────────────────────────────────────────────────
#  VideoProcessor
# ────────────────────────────────────────────────────────────
def make_processor(mode, name, target):
    class FaceProcessor(VideoProcessorBase):
        def __init__(self):
            self.cnt = 0
            self.saved = False

        async def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                faces = insight.get(img)          # deteksi + alignment

                for f in faces:
                    x1, y1, x2, y2 = f.bbox.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    emb = f.normed_embedding      # 512-D

                    # ===== mode daftar ==================================================
                    if mode == "Daftarkan Wajah" and name:
                        if self.cnt < target:
                            db_emb.append(emb)
                            db_names.append(name)
                            self.cnt += 1

                        cv2.putText(img, f"{self.cnt}/{target}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                        # auto-save ketika cukup sampel
                        if self.cnt >= target and not self.saved:
                            np.save(EMB_FILE, np.asarray(db_emb))
                            pickle.dump(db_names, open(NAME_FILE, "wb"))
                            self.saved = True
                            st.toast(f"✅ {self.cnt} sampel tersimpan utk {name}")

                    # ===== mode recognize ===============================================
                    if mode == "Recognize Wajah" and db_emb:
                        emb_db = np.vstack(db_emb)            # (N,512)
                        sims = emb @ emb_db.T                 # cosine sim
                        idx = int(np.argmax(sims))
                        label = db_names[idx] if sims[idx] > 0.55 else "UNKNOWN"

                        cv2.putText(img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 128, 255), 1)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            except Exception as err:
                # log error dan teruskan frame asli supaya koneksi tidak putus
                print("processor error:", err)
                return frame
    return FaceProcessor

# ────────────────────────────────────────────────────────────
#  Tombol kamera
# ────────────────────────────────────────────────────────────
if st.button("Start Camera"):
    if MODE == "Daftarkan Wajah" and not NAME.strip():
        st.warning("Isi nama dulu ya.")
    else:
        webrtc_streamer(
             key="test-loop",
             async_processing=False,
             rtc_configuration={
                 "iceServers": [
                     {"urls": ["stun:stun.l.google.com:19302"]},      # STUN publik
                     {"urls": ["stun:global.stun.twilio.com:3478"]}   # STUN cadangan
                 ]
             },
             media_stream_constraints={
                 "video": {        # kurangi resolusi → handshake lebih cepat
                     "width": { "ideal": 640 },
                     "height": { "ideal": 480 }
                 },
                 "audio": False
             },
             video_html_attrs={
                 "autoPlay": True,
                 "playsInline": True,
                 "style": {"width": "100%"}
             },
         )

st.markdown("---")
st.caption("© 2025 • Demo InsightFace × Streamlit-WebRTC")


