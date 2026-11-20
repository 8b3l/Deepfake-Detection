# 🛡️ DeepFake Video Detection System

An end-to-end **AI Video Forensics** desktop application designed to detect manipulated and synthetic human faces in video streams using a hybrid **Spatiotemporal Deep Learning** pipeline (**MTCNN + ResNet50 + GRU**).

---

## 🖥️ Application Preview

![Uploading image.png…]()


---

## 🧠 Model Architecture & Pipeline

The system analyzes videos across both spatial (frame-level visual artifacts) and temporal (inter-frame motion inconsistencies) dimensions:

1. **Face Detection & Alignment (`MTCNN`):** Locates, crops, and aligns facial regions across sampled video frames to eliminate background noise.
2. **Spatial Feature Extraction (`ResNet50`):** Extracts deep visual embeddings from each aligned face frame to capture subtle blending and texture anomalies.
3. **Temporal Sequence Modeling (`GRU`):** Processes the sequence of frame embeddings over time using Gated Recurrent Units to classify the video as **Authentic (`REAL`)** or **Manipulated (`FAKE`)** with a confidence score.
4. **Desktop Forensics UI (`CustomTkinter` & `OpenCV`):** Provides real-time 1.0x embedded video playback, live inference progress tracking, and clear visual verdict banners.

---

## 📁 Project Structure

```text
Deepfake-Detection/
├── app.py               # Main CustomTkinter GUI & real-time embedded video player
├── processor.py         # Face extraction (MTCNN) and ResNet50 + GRU inference pipeline
├── video_player.py      # Auxiliary video stream handling utilities
├── requirements.txt     # Python dependencies
└── .gitignore           # Excludes large weights, datasets, and cache files
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/8b3l/Deepfake-Detection.git](https://github.com/8b3l/Deepfake-Detection.git)
cd Deepfake-Detection
```

### 2. Install Dependencies
Make sure you have **Python 3.10+** installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Download the Pre-trained Model Weights
Because of file size limits, the trained model weights are hosted in the repository's **Releases** section:
1. Go to the [**Releases (v1.0.0)**](https://github.com/8b3l/Deepfake-Detection/releases/tag/v1.0.0) page.
2. Download **`Model.keras`** (`280 MB`).
3. Place `Model.keras` directly inside the root `Deepfake-Detection/` folder alongside `app.py`.

### 4. Run the Application
```bash
python app.py
```

---

## 🛠️ Tech Stack

* **Deep Learning & Vision:** TensorFlow, Keras, ResNet50, GRU, MTCNN, OpenCV, Pillow, NumPy
* **User Interface:** CustomTkinter, Tkinter
* **Language:** Python 3

---

## 👤 Author

**Ammar Bassam Saleh**  
Computer Engineering Graduate | AI & Machine Learning Engineer  
* **GitHub:** [@8b3l](https://github.com/8b3l)  
* **LinkedIn:** [linkedin.com/in/ammarsaleh11](https://www.linkedin.com/in/ammarsaleh11)
