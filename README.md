# 🛡️ DeepFake Video Detection System

An end-to-end **AI Video Forensics** desktop application designed to detect manipulated and synthetic human faces in video streams using a hybrid **Spatiotemporal Deep Learning** pipeline (**MTCNN + ResNet50 + GRU**).

---

## 🖥️ Application Preview

<img width="2880" height="1819" alt="Screenshot (231)" src="https://github.com/user-attachments/assets/a41c92f9-4695-4f5f-bfb5-ee54f042539e" />


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
