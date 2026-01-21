# 🚁 FMCW Radar Drone vs Bird Classifier (Streamlit + PyTorch)

This project is a **Streamlit-based web application** that classifies **FMCW radar signals** to detect whether an object is a **Drone 🚁** or a **Bird 🐦** using a **PyTorch TorchScript model**.

The application is fully **Dockerized**, uses **CPU-only PyTorch**, and is ready for **local use or cloud deployment**.

---

## 📌 Features

- ✅ Drone vs Bird classification using FMCW radar data
- ✅ Pretrained TorchScript (`.pt`) model
- ✅ Streamlit interactive UI
- ✅ Confidence score for each prediction
- ✅ Alarm sound when a **Drone** is detected
- ✅ Fully Dockerized (CPU-only, no CUDA required)
- ✅ Ready to deploy on Docker Hub / Cloud

---

## 📂 Project Structure

```bash
Drone-Birds-Classification
└── 📁radar_drone_bird
    └── 📁dataset
        ├── Dataset_download.txt
    └── 📁models
        ├── radar_model_scripted.pt <- Select this
    └── 📁notebook
        ├── Radar_CNN_+_LSTM_.ipynb <- Select this
    ├── .dockerignore
    ├── .gitignore
    ├── api.py
    ├── app.py
    ├── beep-03.wav
    ├── dockerfile
    ├── main.py
    └── requirements.txt
```

---

## 🧠 Model Details

- Framework: **PyTorch**
- Format: **TorchScript**
- Input shape: `(N, 1280)`
- Sequence length: `5`
- Classes:
  - `0 → Bird 🐦`
  - `1 → Drone 🚁`

---

## 🚀 Run Locally

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run main.py
```
http://localhost:8501

## 🐳 Run with Docker (Recommended)
```bash
docker pull ahm3dkhanzada/radar-drone-bird
```
```bash
docker run -p 8501:8501 ahm3dkhanzada/radar-drone-bird
```
http://localhost:8501

## 🐳 Build Docker Image
```bash
docker build -t radar-drone-bird .
docker run -p 8501:8501 radar-drone-bird
```
## 🔔 Alarm Behavior

- When Drone 🚁 is detected:

  - Alarm plays for 10 seconds

  - User can manually continue

- For Bird 🐦:

  - Auto-advances to next segment
    
## ⚙️ Configuration

Key parameters in main.py:
```bash
SEQ_LEN = 5
DEVICE = "cpu"
MODEL_PATH = "models/radar_model_scripted.pt"
```

## 🧩 Requirements

  - Python 3.11

  - Streamlit

  - PyTorch (CPU-only)

  - NumPy, Pandas, Matplotlib

  - Docker (optional but recommended)

## 👨‍💻 Author

Ahmed Khanzada
AI / ML Engineer
Docker • PyTorch • Streamlit

## 📜 License

This project is provided for educational and research purposes.
Feel free to modify and extend.

