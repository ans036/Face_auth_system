# Biometric Access Control & Security Terminal

A professional-grade, containerized face authentication system built with **FastAPI**, **React**, and **ArcFace**. This system features real-time recognition, manual verification overrides, and an automated security breach logging dashboard.

## 🚀 Key Features

* **Multi-Template Accuracy**: Stores multiple facial profiles (e.g., with/without spectacles, different angles) to ensure robust matching and solve intra-class variation issues.
* **Facial Alignment HUD**: Uses MediaPipe to mathematically level facial geometry (eyes and nose) before recognition, significantly improving consistency.
* **Manual Verification & Logging**: Toggle between live scanning and manual "Click to Verify" mode. Only manual captures are logged to the permanent security record.
* **Security Breach Dashboard**: A dedicated interface to view timestamped snapshots of unauthorized individuals with their associated confidence scores.
* **Autonomous Audio Alarm**: Integrated hardware-level buzzer that triggers immediately upon detection of an unknown subject.
* **Illumination Normalization**: Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) to flatten harsh shadows and improve recognition in variable lighting.

---

## 🛠️ Tech Stack

* **Core AI**: ArcFace (ONNX) & MediaPipe Tasks API.
* **Backend**: FastAPI (Python 3.10) with SQLAlchemy ORM.
* **Frontend**: Vanilla JS/CSS with HTML5 Canvas for real-time HUD rendering.
* **Database**: SQLite for persistent user metadata and embedding storage.
* **Orchestration**: Docker Compose for seamless deployment across environments.

---

## 📂 Project Structure

```text
face_auth_system/
├── backend/
│   ├── api/            # Identify and Security Dashboard routes
│   ├── core/           # Detector (TFLite) and Recognizer (ArcFace)
│   ├── utils/          # Alignment and CLAHE image normalization
│   └── scripts/        # Multi-template gallery builder
├── frontend/
│   └── public/         # Live Scanner and Security Log UI
├── database/           # User image gallery (organized by username)
├── models/             # detector.tflite and arcface.onnx
└── unauthorized_attempts/ # Auto-generated snapshots of intruders

```

---

## 🚦 Getting Started

### 1. Prerequisites

* **Docker Desktop** (WSL2 backend recommended).
* **Git LFS** (Required to pull the large ArcFace model file).

### 2. Setup Gallery

Place images of authorized users in `database/<username>/`. For maximum accuracy, include 5-10 images showing different facial states (e.g., `Anish_glasses.jpg`, `Anish_sideview.jpg`).

### 3. Deploy

Launch the full stack with one command:

```bash
docker compose up --build

```

Access the Live Scanner at `http://localhost:3000` and the Security Log at `http://localhost:3000/security.html`.

---

## 🛡️ Security Logic & Thresholds

The system uses **Cosine Similarity** with a strict threshold of **0.60** to distinguish between authorized users and strangers.

* **Green Box**: Authorized subject detected (Confidence > 60%).
* **Red Box**: Unknown subject or low-confidence match.
* **Manual Trigger**: Click "Manual Verify" to save the current frame to `/unauthorized_attempts` and update the `security.log` file.

---

## ⚖️ License

Distributed under the MIT License.

---
