#!/usr/bin/env python
# app_insight_webrtc.py
"""
Face-Recognition Demo (InsightFace + streamlit-webrtc)

▶ Cara pakai:
$ pip install -r requirements.txt
$ streamlit run app_insight_webrtc.py
"""

import os, pickle, av, cv2, numpy as np, streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from insightface.app import FaceAnalysis

# ───── Folder & file DB ───────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
EMB_FILE = os.path.join(DATA, "embeddings.npy")
NAME_FILE= os.path.join(DATA, "names.pkl")

# ───── Muat database kalau sudah ada ─────────────────────────────────────────
if os.path.exists(EMB_FILE):
    db_emb = list(np.load(EMB_FILE))
    db_names = pickle.load(open(NAME_FILE, "rb"))
else:
    db_emb, db_names = [], []

# ───── Inisialisasi model InsightFace (cache sekali)──────────────────────────
@st.cache_resource(show_spinner="Memuat model InsightFace...")
def load_model():
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))   # -1 = CPU
    return app
insight = load_model()

# ───── UI Streamlit ──────────────────────────────────────────────────────────
st.title("🤖 Face Recognition – InsightFace WebRTC")
MODE = st.radio("Pilih mode", ["Daftarkan Wajah", "Recognize Wajah"])

if MODE == "Daftarkan Wajah":
    NAME = st.text_input("Nama lengkap / ID", "")
    TARGET = st.slider("Jumlah sampel", 20, 200, 80, 10)
    st.info("Tekan **Start Camera** lalu hadapkan wajah hingga progres mencapai target.")
else:
    NAME, TARGET = None, None
    st.info("Tekan **Start Camera** untuk mengenali wajah.")

# ───── VideoProcessor ────────────────────────────────────────────────────────
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.cnt = 0              # counter sampel
        self.saved = False        # flag simpan DB agar sekali saja

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        faces = insight.get(img)

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            emb = f.normed_embedding  # 512-D L2-normalized vector

            # ---- Mode Pendaftaran -----------------------------------------
            if MODE == "Daftarkan Wajah" and NAME:
                if self.cnt < TARGET:
                    db_emb.append(emb)
                    db_names.append(NAME)
                    self.cnt += 1
                cv2.putText(img, f"{self.cnt}/{TARGET}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

                # Auto-save ketika memenuhi target
                if self.cnt >= TARGET and not self.saved:
                    np.save(EMB_FILE, np.asarray(db_emb))
                    pickle.dump(db_names, open(NAME_FILE, "wb"))
                    self.saved = True
                    st.success(f"Selesai menyimpan {self.cnt} sampel untuk {NAME}")

            # ---- Mode Pengenalan ------------------------------------------
            if MODE == "Recognize Wajah" and db_emb:
                emb_db = np.vstack(db_emb)               # (N,512)
                sims = emb @ emb_db.T                    # cosine similarity
                idx  = int(np.argmax(sims))
                label = db_names[idx] if sims[idx] > 0.55 else "UNKNOWN"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ───── Tombol kamera ────────────────────────────────────────────────────────
if st.button("Start Camera"):
    if MODE == "Daftarkan Wajah" and not NAME.strip():
        st.warning("Isi nama dulu ya.")
    else:
        webrtc_streamer(key="face-cam",
                        video_processor_factory=FaceProcessor,
                        media_stream_constraints={"video": True, "audio": False})

st.markdown("---")
st.caption("© 2025 • Demo InsightFace + Streamlit-WebRTC")
