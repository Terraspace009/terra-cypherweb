import streamlit as st

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(page_title="TERRA-CYPHER", layout="wide")

import torch
import cv2
import numpy as np
import time
from emotion_utils_fixed import load_model, predict_emotion_with_smoothing
from radar_utils_fixed import plot_emotion_radar
from matrix_renderer import frame_to_matrix_dots
from matplotlib import font_manager as fm

# --- Styling ---
st.markdown(f"""
    <style>
    @font-face {{
        font-family: Cyber;
        src: url('fonts/Cyber-Bold.ttf');
    }}
    h1 {{
        margin-top: 20px;
        font-family: Cyber;
        font-size: 48px;
        color: #C0C0C0;
        text-shadow: 0px 0px 15px #B0B0B0, 0px 0px 25px #FFFFFF;
    }}
    .emotion-box {{
        font-family: Cyber;
        font-size: 34px;
        text-align: center;
        color: #C0C0C0;
        border: 2px solid #C0C0C0;
        padding: 12px;
        text-shadow: 0px 0px 12px #B0B0B0;
        background-color: rgba(20, 20, 20, 0.35);
        border-radius: 8px;
    }}
    .stImage > img {{
        transform: scale(0.8);
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>TERRA-CYPHER</h1>", unsafe_allow_html=True)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "terra_emotion_model_vgg13.pt"
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
model = load_model(model_path, device)
history = []

# --- Face Detection ---
def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return frame[y:y + h, x:x + w]
    return frame

prev_gray = None
mode = st.radio("Select Mode", ("Webcam", "Image Upload"))

# -------------------- WEBCAM --------------------
if mode == "Webcam":
    cap = cv2.VideoCapture(0)
    col1, col2, col3 = st.columns([1.0, 1.1, 1.0])
    frame_placeholder = col1.empty()
    matrix_placeholder = col2.empty()
    radar_placeholder = col3.empty()
    emotion_placeholder = st.empty()

    current_emotion = ""
    current_confidence = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Emotion detection (cropped face) ---
        face_for_emotion = detect_and_crop_face(frame_rgb)
        emotion, confidence, smoothed_probs = predict_emotion_with_smoothing(
            model, face_for_emotion, emotion_labels, history, device
        )

        if emotion != current_emotion and confidence < 0.5:
            emotion = current_emotion

        current_emotion = emotion
        current_confidence = (current_confidence * 0.7) + (confidence * 0.3)

        # --- ASCII rendering (unchanged) ---
        dot_image, prev_gray = frame_to_matrix_dots(
            frame_rgb, cols=80, dot_size=5, prev_gray=prev_gray, emotion=current_emotion
        )

        radar_fig = plot_emotion_radar(emotion_labels, smoothed_probs)

        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        emotion_placeholder.markdown(
            f"<div class='emotion-box'>{current_emotion.upper()} ({current_confidence * 100:.1f}%)</div>",
            unsafe_allow_html=True
        )
        matrix_placeholder.image(dot_image, use_container_width=True)
        radar_placeholder.pyplot(radar_fig, clear_figure=True)

        frame_count += 1
        time.sleep(0.02)

# -------------------- IMAGE UPLOAD --------------------
else:
    col1, col2, col3 = st.columns([0.9, 0.9, 0.5])
    uploaded = col1.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    matrix_placeholder = col2.empty()
    emotion_placeholder = st.empty()
    radar_placeholder = col3.empty()

    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_for_emotion = detect_and_crop_face(image_rgb)
        emotion, confidence, smoothed_probs = predict_emotion_with_smoothing(
            model, face_for_emotion, emotion_labels, history, device
        )

        dot_image, _ = frame_to_matrix_dots(
            image_rgb, cols=80, dot_size=5, emotion=emotion
        )

        radar_fig = plot_emotion_radar(emotion_labels, smoothed_probs)

        col1.image(image_rgb, channels="RGB", width=400)
        matrix_placeholder.image(dot_image, use_container_width=True)
        emotion_placeholder.markdown(
            f"<div class='emotion-box'>{emotion.upper()} ({confidence * 100:.1f}%)</div>",
            unsafe_allow_html=True
        )
        radar_placeholder.pyplot(radar_fig, clear_figure=True)
