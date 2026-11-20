# Deepfake Video Detection

A desktop app for screening videos for signs of facial manipulation. The app previews a local video, runs an existing ResNet50 + GRU model, and displays the model's score alongside a preliminary classification.

## Run the desktop app

Use Python 3.11–3.13. The following commands are for Windows PowerShell. Clone from a short path such as `C:\src` if Windows reports a path-length error when installing TensorFlow.

```powershell
git clone https://github.com/8b3l/Deepfake-Detection.git
cd Deepfake-Detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download [`Model.keras` (280 MiB)](https://github.com/8b3l/Deepfake-Detection/releases/download/v1.0.0/Model.keras) from the [v1.0.0 release](https://github.com/8b3l/Deepfake-Detection/releases/tag/v1.0.0) and place it beside `app.py`.

```powershell
.\.venv\Scripts\python.exe app.py
```

Select a video and click **Start Analysis**. MP4, AVI, MOV, and MKV files can be selected. Playback and analysis depend on the codecs available through OpenCV. Video processing runs locally.

## Model and results

- The released Keras file contains the ResNet50 frame encoder, a GRU for temporal modeling, and a single sigmoid output. The app restores the saved architecture and weights and handles the older Keras format used by v1.0.0.
- Analysis reads the first 76 frames at most, sampling every fifth frame for a sequence of 16 RGB frames resized to 224 × 224. Pixel values are scaled to 0–1. If a short video yields fewer frames, its last sampled frame is repeated. The pipeline uses whole frames.
- The app follows the repository's existing label convention. A score of 0.5 or higher is labeled **Likely Fake**. A lower score is labeled **Likely Real**. The displayed number is the model's raw output. The release does not document its training label mapping, preprocessing, or calibration, and this threshold is provisional.
- Edits outside the sampled interval may be missed. Audio is outside the scope of this model. Treat results as preliminary.

## Evaluate on labeled videos

Prepare a CSV with one row per video. Relative paths are resolved from the CSV's folder.

```csv
path,label
videos/clip-01.mp4,real
videos/clip-02.mp4,fake
```

```powershell
.\.venv\Scripts\python.exe evaluate.py evaluation.csv
```

The JSON report gives the video count, confusion matrix, precision, recall, F1, and false-positive rate. To compare thresholds, pass `--threshold` on a validation set and report the selected threshold on a separate test set. Keep each source video and its derivatives in the same split. When identities are known, group those by split as well. Use clips independent of training when reporting test performance.

Run the regression tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project files

- `app.py` — CustomTkinter interface and embedded OpenCV video playback
- `processor.py` — video sampling, Keras model loading, and inference
- `evaluate.py` — evaluation on labeled videos
- `tests/` — regression tests for inference and evaluation

## Author

**Ammar Bassam Saleh**  
Computer Engineering Graduate | AI & Machine Learning Engineer  
[GitHub](https://github.com/8b3l) · [LinkedIn](https://www.linkedin.com/in/ammarsaleh11)
