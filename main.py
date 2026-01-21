import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import streamlit.components.v1 as components
import time
import os
import base64


MODEL_PATH = os.path.join("models", "radar_model_scripted.pt")
ALARM_FILE = "beep-03.wav"
SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if "index" not in st.session_state:
    st.session_state.index = 0

if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False

if "alarm_start_time" not in st.session_state:
    st.session_state.alarm_start_time = None

if "started" not in st.session_state:
    st.session_state.started = False

if "radar_data" not in st.session_state:
    st.session_state.radar_data = None

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
        segment = segment.reshape(SEQ_LEN, 256)  # Assuming SEQ_LEN=5, 5*256=1280

    frames = []
    for _ in range(SEQ_LEN):
        frame = np.abs(segment)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32)
    return torch.from_numpy(frames).unsqueeze(0).to(DEVICE)

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

uploaded_file = st.file_uploader("Upload Radar Segment (.npy)", type=["npy"])

if uploaded_file and not st.session_state.started:
    if st.button("▶️ Start Prediction", type="primary"):
        try:
            st.session_state.radar_data = np.load(uploaded_file, allow_pickle=True)
            if st.session_state.radar_data.ndim != 2 or st.session_state.radar_data.shape[1] != 1280:
                st.error("Invalid file! Radar data must have shape (N, 1280).")
                st.session_state.radar_data = None
            else:
                st.session_state.started = True
                st.session_state.index = 0
                st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

# MAIN LOGIC
if st.session_state.started and st.session_state.radar_data is not None:
    radar_data = st.session_state.radar_data
    total_segments = radar_data.shape[0]

    if st.session_state.index >= total_segments:
        st.success("All segments processed successfully!")
        if st.button("Reset and Upload New File"):
            st.session_state.started = False
            st.session_state.index = 0
            st.session_state.radar_data = None
            st.session_state.alarm_active = False
            st.session_state.alarm_start_time = None
            st.rerun()
        st.stop()

    current_segment = radar_data[st.session_state.index]

    # Progress bar
    progress = (st.session_state.index + 1) / total_segments
    st.progress(progress)
    st.caption(f"Processing segment {st.session_state.index + 1} of {total_segments}")

    # Preprocess and predict
    input_tensor = preprocess_segment(current_segment)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(output, dim=1).item()
        confidence = probs[0, pred_class].item() * 100

    label = get_label_name(pred_class)

    # Display result
    if pred_class == 1:  # DRONE
        st.error(f"{label} detected! Confidence: {confidence:.2f}%")

        if not st.session_state.alarm_active:
            st.session_state.alarm_active = True
            st.session_state.alarm_start_time = time.time()
            play_alarm_10_sec()

        if time.time() - st.session_state.alarm_start_time > 10:
            pause_alarm()  # Ensure alarm is paused
            st.session_state.index += 1
            st.session_state.alarm_active = False
            st.session_state.alarm_start_time = None
            st.rerun()
        else:
            st.warning("Alarm is active for 10 seconds. Auto-advancing if no action.")
            if st.button("Continue to Next Segment", type="primary"):
                pause_alarm()
                st.session_state.index += 1
                st.session_state.alarm_active = False
                st.session_state.alarm_start_time = None
                st.rerun()

            # Periodic reload to check timer every 1 second
            components.html("""
            <script>
            setTimeout(function() {
                location.reload();
            }, 1000);
            </script>
            """, height=0)

    else:  # BIRD
        st.success(f"{label} detected. Confidence: {confidence:.2f}%")
        time.sleep(1)  # Auto-advance after 1 second
        st.session_state.index += 1
        st.rerun()

# Stop button available during processing
if st.session_state.started:
    if st.button("🛑 Stop Processing"):
        if st.session_state.alarm_active:
            pause_alarm()
        st.session_state.started = False
        st.session_state.index = 0
        st.session_state.radar_data = None
        st.session_state.alarm_active = False
        st.session_state.alarm_start_time = None
        st.rerun()