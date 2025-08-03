# app_insight_webrtc.py
import streamlit as st, av, cv2, numpy as np, os, pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from insightface.app import FaceAnalysis

DATA = "data"
os.makedirs(DATA, exist_ok=True)

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320,320))
    return app
app_insight = load_model()

MODE = st.radio("Mode", ("Daftarkan Wajah", "Recognize Wajah"))
NAME = st.text_input("Nama", "") if MODE == "Daftarkan Wajah" else None
TARGET = 80

# ───── database load ─────
enc_file, name_file = f"{DATA}/emb.npy", f"{DATA}/names.pkl"
if os.path.exists(enc_file):
    db_emb = list(np.load(enc_file))
    db_names = pickle.load(open(name_file, "rb"))
else:
    db_emb, db_names = [], []

class Processor(VideoProcessorBase):
    def __init__(self):
        self.cnt = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = app_insight.get(img)

        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(int)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),1)

            emb = f.normed_embedding
            if MODE == "Daftarkan Wajah" and NAME and self.cnt < TARGET:
                db_emb.append(emb)
                db_names.append(NAME)
                self.cnt += 1
                cv2.putText(img, f"{self.cnt}/{TARGET}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)

            if MODE == "Recognize Wajah" and db_emb:
                sims = emb @ np.vstack(db_emb).T
                idx = int(np.argmax(sims))
                label = db_names[idx] if sims[idx] > 0.55 else "UNKNOWN"
                cv2.putText(img, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(0,128,255),1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if st.button("Start Camera"):
    webrtc_streamer(key="cam", video_processor_factory=Processor,
                    media_stream_constraints={"video": True, "audio": False})

# ───── simpan DB saat sesi selesai ─────
def save_db():
    if db_emb:
        np.save(enc_file, np.asarray(db_emb))
        pickle.dump(db_names, open(name_file,"wb"))
st.on_session_end(save_db)
