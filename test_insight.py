#!/usr/bin/env python
"""
Real-time pengenalan wajah memakai InsightFace embedding 512-D + jarak kosinus.
"""

import os, pickle, cv2, numpy as np
from insightface.app import FaceAnalysis

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
emb_db = np.load(os.path.join(DATA, "embeddings.npy"))
names  = pickle.load(open(os.path.join(DATA, "names.pkl"), "rb"))

TOLERANCE = 0.45       # makin kecil = makin ketat
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320,320))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "Webcam tidak terbuka!"

while True:
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)
    for f in faces:
        emb  = f.normed_embedding
        x1,y1,x2,y2 = f.bbox.astype(int)

        # cos-sim → jarak
        sims = emb @ emb_db.T
        idx  = np.argmax(sims)
        label = names[idx] if sims[idx] > (1 - TOLERANCE) else "UNKNOWN"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,128,255), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,128,255), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release(); cv2.destroyAllWindows()
