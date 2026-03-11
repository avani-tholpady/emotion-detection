#  Real-Time Emotion Detection

A real-time facial emotion detection app built with **Python**, **OpenCV**, and **DeepFace** — featuring a polished live UI with color-coded emotions, a sidebar overlay, confidence bars, and session analytics.

---

## ✨ Features

- 🎨 **Color-coded bounding boxes** — each emotion has its own distinct color
- 🖥️ **Live sidebar overlay** — shows emotion confidence bars for all 7 emotions in real time
- ⏱️ **Emotion hold timer** — tracks how long the current emotion has been held
- 👥 **Multi-face support** — detects and analyzes multiple faces simultaneously
- 📊 **Session summary** — on exit, prints total detections, dominant emotion & full breakdown
- 📝 **CSV logging** — every detection is saved to `logs/detection_log.csv`
- ⚡ **Threaded analysis** — DeepFace runs in a background thread so the camera stays smooth
- 🔍 **Confidence threshold filter** — only shows results above 40% confidence

---

##  Detected Emotions

| Emotion   | Color   |
|-----------|---------|
| 😊 Happy   | Green   |
| 😢 Sad     | Blue    |
| 😠 Angry   | Red     |
| 😲 Surprise | Cyan   |
| 😨 Fear    | Purple  |
| 🤢 Disgust | Dark Green |
| 😐 Neutral | Gray    |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run will auto-download DeepFace model weights (~100MB). This only happens once.

### 3. Run
```bash
python emotion_detection.py
```

Press **`q`** to quit and see your session summary.

---

## 📁 Project Structure

```
emotion-detection/
│
├── emotion_detection.py     # Main script
├── requirements.txt         # Dependencies
├── README.md                # This file
│
└── logs/
    └── detection_log.csv    # Auto-generated detection log
```

---

## ⚙️ Configuration

You can tweak these constants at the top of `emotion_detection.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `40.0` | Minimum confidence % to display a result |
| `ANALYZE_EVERY_N_FRAMES` | `5` | How often DeepFace runs (lower = more frequent but heavier) |
| `SIDEBAR_WIDTH` | `260` | Width of the right sidebar panel |

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/) — Camera capture & UI rendering
- [DeepFace](https://github.com/serengil/deepface) — Facial emotion analysis
- [NumPy](https://numpy.org/) — Frame canvas manipulation

---

## 📊 Session Summary Example

```
========================================
       SESSION SUMMARY
========================================
  Duration        : 02:35
  Total Detections: 312
  Dominant Emotion: happy (189 times)

  Emotion Breakdown:
    happy        189x  (60.6%)
    neutral       94x  (30.1%)
    surprise      18x  ( 5.8%)
    sad            7x  ( 2.2%)
    angry          4x  ( 1.3%)
========================================
  Log saved to: logs/detection_log.csv
========================================
```

---

## 📄 License

MIT License — free to use, modify, and share.
