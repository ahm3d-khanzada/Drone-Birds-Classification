import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_SAVE_PATH = r"D:\radar_drone_bird\models\radar_cnn_lstm.pth"
SEQ_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_label_name(pred):
    return "🚁 DRONE" if pred == 1 else "🐦 BIRD"


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class RadarCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.fc(out[:, -1])


@st.cache_resource
def load_model():
    model = RadarCNNLSTM()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()


def preprocess_segment(segment):
    """
    Input: segment shape (1280,) or (5,256)
    Output: tensor shape (1, SEQ_LEN, 5, 256)
    """
    # Flattened 1D segment → (5,256)
    if segment.ndim == 1:
        segment = segment.reshape(5, 256)
    
    frames = []
    for _ in range(SEQ_LEN):
        # Convert complex → magnitude if needed
        frame = np.abs(segment)
        # Normalize 0-1
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32)
    return torch.from_numpy(frames).unsqueeze(0)  # (1, SEQ_LEN, 5, 256)


# STREAMLIT UI
st.title("🚀 FMCW Radar Drone vs Bird Classifier")
st.write("Upload radar segment (.npy) and get prediction")

uploaded_file = st.file_uploader(
    "Upload Radar Segment (.npy)", type=["npy"]
)

# PREDICTION
if uploaded_file is not None:
    radar_data = np.load(uploaded_file, allow_pickle=True)
    st.write("Radar data shape:", radar_data.shape)

    # Multiple segments: (N,1280)
    if radar_data.ndim == 2 and radar_data.shape[1] == 1280:
        st.write(f"Detected {radar_data.shape[0]} segments, predicting...")
        for i, seg in enumerate(radar_data):
            x = preprocess_segment(seg).to(DEVICE)
            with torch.no_grad():
                out = model(x)
                probs = F.softmax(out, dim=1)
                pred = torch.argmax(out, dim=1).item()
                confidence = probs[0, pred].item() * 100
            label = get_label_name(pred)
            st.success(f"Segment {i+1}: {label} ({confidence:.2f}%)")

    # Single segment: (1280,)
    elif radar_data.ndim == 1 and radar_data.shape[0] == 1280:
        x = preprocess_segment(radar_data).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)
            pred = torch.argmax(out, dim=1).item()
            confidence = probs[0, pred].item() * 100
        label = get_label_name(pred)
        st.success(f"Prediction: {label} ({confidence:.2f}%)")

    else:
        st.error("Radar segment must have shape (1280,) or (N,1280)")
