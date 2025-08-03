#!/usr/bin/env python
"""
Daftarkan wajah → simpan embedding ArcFace (512-D) + nama.
"""

import argparse, os, pickle, cv2, numpy as np
from insightface.app import FaceAnalysis

# ─── argparser ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Nama orang yang didaftarkan")
parser.add_argument("--samples", type=int, default=80, help="Jumlah sampel")
args = parser.parse_args()
name = args.name.strip()

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

# ─── insightface init ─────────────────────────────────────────────────────────
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320)) # -1 = CPU, 0 = GPU pertama         # detektor RetinaFace + alignment

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "Webcam tidak terbuka!"

encodings, names = [], []
collected, total_target = 0, args.samples

while collected < total_target:
    ret, frame = cap.read()
    if not ret: continue

    faces = app.get(frame)
    for f in faces:
        emb  = f.normed_embedding          # (512,)
        x1,y1,x2,y2 = f.bbox.astype(int)

        encodings.append(emb)
        names.append(name)
        collected += 1

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{collected}/{total_target}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)

    cv2.imshow("Face Registration", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release(); cv2.destroyAllWindows()

# ─── simpan ──────────────────────────────────────────────────────────────────
enc_file  = os.path.join(DATA, "embeddings.npy")
names_file= os.path.join(DATA, "names.pkl")

if os.path.exists(enc_file):
    encodings_prev = np.load(enc_file)
    encodings      = np.vstack([encodings_prev, encodings])
np.save(enc_file, np.asarray(encodings))

if os.path.exists(names_file):
    names_prev = pickle.load(open(names_file, "rb"))
    names      = names_prev + names
pickle.dump(names, open(names_file, "wb"))

print(f"[INFO] {collected} embedding tersimpan utk {name}")

