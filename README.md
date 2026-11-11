# 🛡️ AI-Based Fall Detection System — V2.0

### 🚀 Intelligent Human Fall Detection | Computer Vision | YOLO Pose Estimation | ByteTrack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Computer%20Vision-00D9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLO-Pose%20Estimation-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tracking-ByteTrack-00C853?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Version-2.0-orange?style=for-the-badge" />
</p>

<p align="center">
  <b>An intelligent computer-vision prototype designed to identify possible human falls in real time using AI-powered pose estimation and multi-person tracking.</b>
</p>

---

## 🌌 Project Overview

**AI-Based Fall Detection System V2.0** is an advanced Python-based computer vision prototype that detects possible human falls through webcam feeds and recorded video.

Powered by **Ultralytics YOLO Pose Estimation**, **ByteTrack**, and a custom temporal posture analysis engine, the system monitors multiple people simultaneously and evaluates their body posture over time.

The system uses geometric measurements, temporal confirmation, and tracking IDs to reduce unstable detections and generate structured fall-event logs.

### ✨ Core Highlights

- 🤖 AI-powered human pose estimation.
- 👥 Multi-person detection and tracking.
- 🎯 Unique tracking IDs for individuals.
- 📐 Torso-angle and body-orientation analysis.
- ⏱️ Temporal fall confirmation.
- 📊 Smoothed posture measurements.
- 🚨 Per-person fall-state monitoring.
- 🎥 Annotated MP4 video recording.
- 📝 Automatic CSV event logging.
- 🧪 Unit tests for decision-making logic.

---

## 🧠 System Architecture

```text
       ┌─────────────────────────┐
       │   Webcam / Video Input  │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │     OpenCV Processing   │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │  YOLO Pose Estimation   │
       │  Human & Keypoint Data  │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │   ByteTrack Tracking    │
       │   Person Tracking IDs   │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │   Posture Analysis      │
       │  Angle & Body Geometry  │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Temporal Confirmation   │
       │  Per-Person Fall State  │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │  Alert & Event Logging  │
       │   CSV / Annotated MP4   │
       └─────────────────────────┘
```

---

## ⚡ What's New in V2.0?

| Feature | Description |
|---|---|
| Multi-Person Detection | Detect multiple visible people |
| ByteTrack Integration | Maintain tracking IDs across frames |
| Temporal Analysis | Confirm fall-like postures over time |
| Posture Smoothing | Reduce unstable measurements |
| Individual Fall States | Track each person's status separately |
| Configurable Thresholds | Customize detection parameters |
| Video Recording | Save annotated MP4 output |
| Event Logging | Record confirmed events in CSV |
| Unit Testing | Test fall-decision logic |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video capture and image processing |
| Ultralytics YOLO | Human pose estimation |
| ByteTrack | Multi-person object tracking |
| NumPy | Numerical calculations |
| CSV | Fall-event data storage |
| Unittest | Automated logic testing |

---

## 📂 Project Structure

```text
AI_Fall_Detection_System_V2/
│
├── fall_detection/
│   ├── __init__.py
│   ├── detector.py
│   └── logic.py
│
├── tests/
│   └── test_logic.py
│
├── output/
│   └── .gitkeep
│
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 💻 Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/ATIQULTIU/AI_Fall_Detection_System_V2.git
```

```bash
cd AI_Fall_Detection_System_V2
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Model Weights

The default `yolo11n-pose.pt` model weights are automatically downloaded by Ultralytics during the first run if they are not already cached.

**Note:** Model weights are not included in the repository.

---

## 🚀 How to Run

### Start Webcam Detection

```bash
python main.py
```

### Process a Video File

```bash
python main.py --source "sample_video.mp4"
```

### Save Annotated Video

```bash
python main.py --source "sample_video.mp4" --output output/annotated.mp4
```

### Run Using CPU

```bash
python main.py --device cpu
```

### Customize Detection Parameters

```bash
python main.py --conf 0.40 --angle 60 --horizontal-ratio 1.20 --confirm 1.0
```

Press **Q** or **ESC** to stop the preview.

---

## ⚙️ Configuration Options

| Argument | Description | Example |
|---|---|---|
| `--source` | Webcam index or video path | `0` |
| `--model` | YOLO pose model weights | `yolo11n-pose.pt` |
| `--conf` | Minimum detection confidence | `0.40` |
| `--imgsz` | Inference image size | `640` |
| `--confirm` | Fall confirmation duration | `1.0` |
| `--angle` | Torso angle threshold | `60` |
| `--horizontal-ratio` | Body width/height threshold | `1.20` |
| `--device` | Processing device | `cpu` |
| `--output` | Annotated MP4 destination | `output/annotated.mp4` |
| `--log` | CSV event-log destination | `output/fall_events.csv` |
| `--no-display` | Disable preview window | Flag |

---

## 📊 Output & Event Monitoring

The system generates structured output files to support monitoring and analysis.

**Fall Event Log**

```text
output/fall_events.csv
```

Contains one event row when a tracked person enters a confirmed fall state.

**Annotated Video**

```text
output/annotated.mp4
```

Generated only when an output path is supplied using `--output`.

---

## 🧪 Testing

Run the automated unit tests:

```bash
python -m unittest discover -s tests -v
```

The tests validate the fall-decision and temporal state logic.

**Important:** Passing unit tests does not demonstrate real-world fall-detection accuracy.

---

## 🔬 Detection Workflow

1. OpenCV captures video frames from a webcam or video file.
2. YOLO estimates human positions and body keypoints.
3. ByteTrack associates detections with individual tracking IDs.
4. The system calculates torso angles and bounding-box aspect ratios.
5. Posture measurements are smoothed over time.
6. Temporal confirmation checks whether a fall-like posture persists.
7. Confirmed events are displayed and recorded in CSV format.

---

## ⚠️ Safety & Limitations

This project is an educational and research prototype.

- It is not a certified medical device.
- It does not provide clinically validated fall detection.
- Bending, lying down, camera perspective, occlusion, and unusual movements may trigger false alerts or missed detections.
- Tracking IDs may change when people leave or re-enter the camera view.
- Detection performance depends on camera positioning, lighting, video quality, and model performance.

**Never rely on this system as the sole method of monitoring a person or contacting emergency services.**

---

## 🔮 Future Development Roadmap

- [ ] Modern real-time web dashboard.
- [ ] Live detection analytics.
- [ ] Email and notification integration.
- [ ] Historical fall-event visualization.
- [ ] Database integration.
- [ ] Improved temporal fall classification.
- [ ] Mobile monitoring interface.
- [ ] Edge-device deployment.
- [ ] Performance benchmarking and validation.

---

## 👨‍💻 Developer

<p align="center">
  <b>MD. Atiqul Islam (Atik)</b><br/>
  Software Developer | Computer Science & Engineering
</p>

<p align="center">
  📧 <a href="mailto:atik.cmttiu1001@gmail.com">atik.cmttiu1001@gmail.com</a><br/>
  💻 <a href="https://github.com/ATIQULTIU">GitHub Profile</a><br/>
  🌐 <a href="https://md-atiqul-islam.my.canva.site/">Portfolio Website</a>
</p>

---

## 📜 License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>⚡ AI INNOVATION • COMPUTER VISION • INTELLIGENT MONITORING ⚡</b>
  <br/><br/>
  Made with ❤️ and Python by <b>MD. Atiqul Islam (Atik)</b>
  <br/>
  <sub>AI-Based Fall Detection System — Version 2.0</sub>
</p>
