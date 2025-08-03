# debug_webrtc.py  ▶ jalankan lokal **dan** di Streamlit Cloud
import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("🔎 WebRTC Loopback Test")
st.write("Klik Start → seharusnya tampak video webcam.\n"
         "Jika kotak hilang ➜ lihat Console browser (F12) & Streamlit logs.")

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
