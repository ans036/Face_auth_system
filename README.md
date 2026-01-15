# Biometric Access Control & Security Terminal v3.1.0

[![CI](https://github.com/ans036/Face_auth_system/actions/workflows/ci.yml/badge.svg)](https://github.com/ans036/Face_auth_system/actions/workflows/ci.yml)

A professional-grade **Multi-Modal Biometric System** fusing **Face Recognition (InsightFace)**, **Voice Authentication (SpeechBrain)**, and **Liveness Detection** for high-security enterprise access control.

> **Status**: ✅ Development Stable | v3.1.0
> **Latest Feature**: CI/CD Pipeline with Pytest Testing


## 🚀 Key Features

### 🛡️ Multi-Modal Security
*   **Face Recognition**: InsightFace Buffalo-L model (600K identity training).
*   **Voice Authentication**: SpeechBrain ECAPA-TDNN speaker verification.
*   **Liveness Detection**: Real-time blink eye tracking to prevent photo spoofing.
*   **Fusion Logic**: Adaptive scoring combines face + voice probabilities for high-confidence matches.

### 👮 Admin & Monitoring
*   **Admin Panel**: Secured dashboard to view stats, logs, and captured images.
*   **Security Logs**: Automated logging of all authorized/unauthorized attempts.
*   **Unauthorized Capture**: Automatically saves images of failed access attempts.
*   **Private User Messages**: Securely deliver messages to specific users upon authentication.

### ⚙️ advanced Tech
*   **Windows Docker Fix**: Custom port mapping (8001:8000) to bypass known Windows networking bugs.
*   **Expression-Robust**: Top-5 averaging handles smiling/neutral variations.
*   **CLAHE Preprocessing**: Adaptive histogram equalization for variable lighting.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Models** | InsightFace (Face) + SpeechBrain (Voice) |
| **Backend** | FastAPI + SQLAlchemy + TensorFlow Lite |
| **Frontend** | Vanilla JS + WebRTC + Chart.js |
| **Database** | SQLite (Local Secure Storage) |
| **Container** | Docker Compose (Optimized for Windows) |

---

## 📂 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/identify/` | POST | Multi-modal identification (Face + Voice + Liveness) |
| `/admin/logs` | GET | Retrieve security logs (Admin only) |
| `/admin/stats` | GET | System statistics (Admin only) |
| `/enroll/` | POST | Enroll new users/voices |
| `/gallery/rebuild` | POST | Rebuild embeddings gallery |

---

## 🚦 Getting Started

### 1. Prerequisites
*   **Docker Desktop** (WSL2 backend recommended)
*   **Git LFS** (for large model files)

### 2. Setup Database
Place user images in `database/<username>/` and voice samples (optional) in `database/<username>/voice/`.

### 3. Deploy
```bash
docker compose up --build
```

### 4. Access
*   **Live Scanner**: http://localhost:3000
*   **Admin Panel**: http://localhost:3000/admin.html
    *   *Default User*: `admin`
    *   *Default Pass*: `admin123`
*   **API Docs**: http://localhost:8001/docs

---

## 🔧 Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| **Face Threshold** | 0.30 | Lowered for better recall with multi-modal fusion |
| **Voice Weight** | 0.25 | Booster weight for voice match |
| **Liveness** | Blink | Required for "Live" status |
| **Backend Port** | 8001 | Mapped to 8000 internally to fix Windows port hanging |

---

## 🛡️ Security Features
*   **Green Box**: Authenticated (Face + Voice + Life)
*   **Red Box**: Unknown subject
*   **🔊 Audio Alarm**: Triggers on unauthorized manual verification
*   **📸 Evidence**: Unauthorized images saved to `unauthorized_attempts/`

---

## 🧪 Testing

Run the pytest test suite locally:

```bash
cd backend
pip install pytest pytest-asyncio httpx pillow
pytest tests/ -v
```

The CI pipeline runs automatically on push/PR to `main` branch.

---

## ⚖️ License
MIT License
