# face_app.py (v2 – InsightFace)
"""
Streamlit UI (web) untuk aplikasi Face Recognition berbasis **InsightFace + ONNXRuntime**.

📌 Fitur
• Landing page dengan dua mode — *Daftarkan Wajah* & *Recognize Wajah*  
• Nama untuk pendaftaran diinput lewat UI (bukan console)  
• Menjalankan skrip back‑end sebagai proses terpisah, sehingga Streamlit tetap responsif.

Cara jalan lokal:
$ pip install -r requirements.txt
$ streamlit run face_app.py
"""

import os
import pathlib
import subprocess
import sys
import streamlit as st

SCRIPT_REGISTER = "add_faces_insight.py"   # script pendaftaran
SCRIPT_RECOGN   = "test_insight.py"        # script pengenalan

st.set_page_config(page_title="Face Recognition DL",
                   page_icon="🤖",
                   layout="centered")

st.title("🤖 Face Recognition (InsightFace)")
mode = st.radio("Pilih Mode", ("Daftarkan Wajah", "Recognize Wajah"))

if mode == "Daftarkan Wajah":
    name = st.text_input("Nama lengkap / ID", "")
else:
    name = None

if st.button("Mulai Kamera"):
    if mode == "Daftarkan Wajah" and not name.strip():
        st.warning("Silakan isi nama terlebih dahulu.")
        st.stop()

    script = SCRIPT_REGISTER if mode == "Daftarkan Wajah" else SCRIPT_RECOGN
    script_path = pathlib.Path(__file__).with_name(script)

    if not script_path.exists():
        st.error(f"File {script} tidak ditemukan di folder yang sama.")
        st.stop()

    cmd = [sys.executable, str(script_path)]
    if name:
        cmd += ["--name", name.strip()]

    st.info("Menjalankan proses… Jendela OpenCV akan terbuka. Tekan q untuk keluar.")
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
        if proc.returncode == 0:
            st.success("Proses selesai tanpa error.")
        else:
            st.error(f"Script berhenti dengan kode {proc.returncode}")
    except Exception as err:
        st.exception(err)

st.markdown("---")

