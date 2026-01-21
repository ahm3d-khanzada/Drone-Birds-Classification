import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import streamlit.components.v1 as components
import time
import os
import base64
import requests  # For real API calls

# CONFIG
MODEL_PATH = r"D:\radar_drone_bird\models\radar_model_scripted.pt"
ALARM_FILE = "beep-03.wav"  # Ensure this file is in the same folder as the script
SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_ENDPOINT = "http://example-radar-api/get_segment"  # Replace with real API URL

# SESSION STATE INIT
if "index" not in st.session_state:
    st.session_state.index = 0

if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False

if "alarm_start_time" not in st.session_state:
    st.session_state.alarm_start_time = None

if "started" not in st.session_state:
    st.session_state.started = False

if "continuous_mode" not in st.session_state:
    st.session_state.continuous_mode = False

if "radar_data" not in st.session_state:
    st.session_state.radar_data = None

# LABEL
def get_label_name(pred):
    return "🚁 DRONE" if pred == 1 else "🐦 BIRD"

# LOAD MODEL
@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

model = load_model()

# PREPROCESS
def preprocess_segment(segment):
    if segment.ndim == 1:
        segment = segment.reshape(SEQ_LEN, 256)

    frames = []
    for _ in range(SEQ_LEN):
        frame = np.abs(segment)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32)
    return torch.from_numpy(frames).unsqueeze(0).to(DEVICE)

# MOCK API FETCH (Replace with real API)
def fetch_radar_segment_from_api():
    # Mock: Generate random complex data simulating radar segment (shape: 1280)
    # In real: Use requests to fetch from API
    try:
        # Real API example (uncomment and adjust):
        # response = requests.get(API_ENDPOINT, headers={"Authorization": "YOUR_API_KEY"})
        # if response.status_code == 200:
        #     data = np.frombuffer(response.content, dtype=np.complex128)
        #     return data
        # else:
        #     st.error("API fetch failed!")
        #     return None

        # Mock data for demo
        mock_data = np.random.random(1280) + 1j * np.random.random(1280)  # Complex radar-like data
        return mock_data
    except Exception as e:
        st.error(f"API error: {e}")
        return None

# ALARM (10 SEC)
def play_alarm_10_sec():
    if not os.path.exists(ALARM_FILE):
        st.error("Alarm sound file missing! Please ensure 'beep-03.wav' is in the same folder.")
        return

    with open(ALARM_FILE, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio id="alarm" autoplay loop>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
    </audio>
    <script>
        let alarmAudio = document.getElementById("alarm");
        alarmAudio.play().catch(error => console.log("Autoplay prevented: " + error));
        setTimeout(() => {{
            alarmAudio.pause();
            alarmAudio.currentTime = 0;
        }}, 10000);
    </script>
    """
    components.html(audio_html, height=0)

# PAUSE ALARM
def pause_alarm():
    pause_html = """
    <script>
        let alarmAudio = document.getElementById("alarm");
        if (alarmAudio) {
            alarmAudio.pause();
            alarmAudio.currentTime = 0;
        }
    </script>
    """
    components.html(pause_html, height=0)

# UI
st.title("FMCW Radar Drone vs Bird Classifier")

# Option for batch or continuous
mode = st.radio("Select Mode", ["Batch Upload", "Continuous API Monitoring"])

if mode == "Batch Upload":
    uploaded_file = st.file_uploader("Upload Radar Segment (.npy)", type=["npy"])

    if uploaded_file and not st.session_state.started:
        if st.button("Start Prediction", type="primary"):
            try:
                st.session_state.radar_data = np.load(uploaded_file, allow_pickle=True)
                if st.session_state.radar_data.ndim != 2 or st.session_state.radar_data.shape[1] != 1280:
                    st.error("Invalid file! Radar data must have shape (N, 1280).")
                    st.session_state.radar_data = None
                else:
                    st.session_state.started = True
                    st.session_state.continuous_mode = False
                    st.session_state.index = 0
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Batch logic (same as before)
    if st.session_state.started and st.session_state.radar_data is not None and not st.session_state.continuous_mode:
        radar_data = st.session_state.radar_data
        total_segments = radar_data.shape[0]

        if st.session_state.index >= total_segments:
            st.success("All segments processed successfully!")
            if st.button("Reset"):
                st.session_state.started = False
                st.session_state.index = 0
                st.session_state.radar_data = None
                st.session_state.alarm_active = False
                st.session_state.alarm_start_time = None
                st.rerun()
            st.stop()

        current_segment = radar_data[st.session_state.index]

        progress = (st.session_state.index + 1) / total_segments
        st.progress(progress)
        st.caption(f"Processing segment {st.session_state.index + 1} of {total_segments}")

        input_tensor = preprocess_segment(current_segment)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred_class].item() * 100

        label = get_label_name(pred_class)

        if pred_class == 1:  # DRONE
            st.error(f"{label} detected! Confidence: {confidence:.2f}%")

            if not st.session_state.alarm_active:
                pause_alarm()  # Ensure any previous alarm is fully stopped before starting new
                st.session_state.alarm_active = True
                st.session_state.alarm_start_time = time.time()
                play_alarm_10_sec()

            if time.time() - st.session_state.alarm_start_time > 10:
                pause_alarm()
                st.session_state.index += 1
                st.session_state.alarm_active = False
                st.session_state.alarm_start_time = None
                st.rerun()
            else:
                st.warning("Alarm active. Auto-advancing in 10 sec.")
                if st.button("Continue", type="primary"):
                    pause_alarm()
                    st.session_state.index += 1
                    st.session_state.alarm_active = False
                    st.session_state.alarm_start_time = None
                    st.rerun()

                components.html("""
                <script>
                setTimeout(function() {
                    location.reload();
                }, 1000);
                </script>
                """, height=0)

        else:  # BIRD
            st.success(f"{label} detected. Confidence: {confidence:.2f}%")
            time.sleep(1)
            st.session_state.index += 1
            st.rerun()

else:  # Continuous Mode
    if not st.session_state.continuous_mode:
        if st.button("▶Start Continuous Monitoring", type="primary"):
            st.session_state.continuous_mode = True
            st.session_state.started = True
            st.session_state.index = 0  # Use index as segment counter
            st.rerun()

    if st.session_state.continuous_mode:
        st.caption(f"Processing continuous segment {st.session_state.index + 1}")

        # Fetch from API (mock or real)
        current_segment = fetch_radar_segment_from_api()
        if current_segment is None:
            st.error("Failed to fetch segment. Stopping.")
            st.session_state.continuous_mode = False
            st.stop()

        input_tensor = preprocess_segment(current_segment)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred_class].item() * 100

        label = get_label_name(pred_class)

        if pred_class == 1:  # DRONE
            st.error(f"{label} detected! Confidence: {confidence:.2f}%")

            if not st.session_state.alarm_active:
                pause_alarm()
                st.session_state.alarm_active = True
                st.session_state.alarm_start_time = time.time()
                play_alarm_10_sec()

            if time.time() - st.session_state.alarm_start_time > 10:
                pause_alarm()
                st.session_state.index += 1
                st.session_state.alarm_active = False
                st.session_state.alarm_start_time = None
                st.rerun()
            else:
                st.warning("Alarm active. Auto-advancing in 10 sec.")
                if st.button("Continue", type="primary"):
                    pause_alarm()
                    st.session_state.index += 1
                    st.session_state.alarm_active = False
                    st.session_state.alarm_start_time = None
                    st.rerun()

                components.html("""
                <script>
                setTimeout(function() {
                    location.reload();
                }, 1000);
                </script>
                """, height=0)

        else:  # BIRD
            st.success(f"{label} detected. Confidence: {confidence:.2f}%")
            time.sleep(1)  # Simulate processing delay
            st.session_state.index += 1
            st.rerun()

# Stop button
if st.session_state.started:
    if st.button("🛑 Stop Processing"):
        if st.session_state.alarm_active:
            pause_alarm()
        st.session_state.started = False
        st.session_state.continuous_mode = False
        st.session_state.index = 0
        st.session_state.radar_data = None
        st.session_state.alarm_active = False
        st.session_state.alarm_start_time = None
        st.rerun()