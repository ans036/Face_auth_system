# Biometric Access Control & Security Terminal v2.0

A professional-grade face authentication system built with **FastAPI**, **InsightFace Buffalo-L**, and real-time WebRTC. Features multi-image enrollment, gallery management API, and automated security breach logging.

## 🚀 Key Features

* **InsightFace Buffalo-L Model**: State-of-the-art face recognition with 600K identity training
* **Multi-Image Enrollment API**: Upload multiple face images via REST API for robust matching
* **Gallery Management**: List, delete, and rebuild face galleries via API
* **Expression-Robust Recognition**: Top-5 averaging handles smiling/neutral variations
* **Manual Verification & Alarm**: Triggers audio alarm when unauthorized subjects attempt verification
* **Security Dashboard**: View timestamped snapshots of unauthorized access attempts
* **CLAHE Preprocessing**: Adaptive histogram equalization for variable lighting

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Model** | InsightFace Buffalo-L (w600k_r50) |
| **Backend** | FastAPI + SQLAlchemy |
| **Frontend** | Vanilla JS with HTML5 Canvas |
| **Database** | SQLite |
| **Container** | Docker Compose |

---

## 📂 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/identify/` | POST | Identify face from image |
| `/enroll/` | POST | Enroll new images (multi-file) |
| `/gallery/` | GET | List all enrolled users |
| `/gallery/{user}` | DELETE | Remove user from gallery |
| `/gallery/rebuild` | POST | Rebuild gallery from database folder |

---

## 🚦 Getting Started

### 1. Prerequisites

* **Docker Desktop** (WSL2 backend recommended)
* **Git LFS** (for large model files)

### 2. Setup Gallery

Place images in `database/<username>/`:
```
database/
├── Anish/
│   ├── pic1.jpg
│   ├── smile.jpg
│   └── glasses.jpg
└── Sayani/
    └── pic1.jpg
```

### 3. Deploy

```bash
docker compose up --build
```

### 4. Access

* **Live Scanner**: http://localhost:3000
* **Security Log**: http://localhost:3000/security.html
* **API Docs**: http://localhost:8000/docs

---

## 🔧 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Threshold | 0.45 | Minimum similarity for authentication |
| Confidence Gap | 0.03 | Required gap between top 2 candidates |
| Top-K Average | 5 | Number of best matches to average |

---

## 📸 Enrollment via API

```bash
# Add smiling photos to improve accuracy
curl -X POST http://localhost:8000/enroll/ \
  -F "username=Anish" \
  -F "files=@smile1.jpg" \
  -F "files=@smile2.jpg"
```

---

## 🛡️ Security Features

* **Green Box**: Authenticated user (score ≥ 45%)
* **Red Box**: Unknown subject
* **🔊 Audio Alarm**: Plays 3x when manual verify triggered on unauthorized subject
* **📸 Snapshot**: Unauthorized attempts saved to `/unauthorized_attempts/`

---

## ⚖️ License

MIT License
