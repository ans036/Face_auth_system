# Biometric Access Control & Security Terminal v3.3.0

[![CI - Run Tests](https://github.com/ans036/Face_auth_system/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ans036/Face_auth_system/actions/workflows/ci.yml)

A professional-grade **Multi-Modal Biometric System** fusing **Face Recognition (InsightFace)**, **Voice Authentication (SpeechBrain)**, and **Liveness Detection** for high-security enterprise access control. Now powered by **PostgreSQL + pgvector** for enterprise-scale performance.

> **Status**: ✅ Production Ready | v3.3.0
> **Latest Feature**: High-Performance Vector Database (pgvector) Integration

## 🚀 Key Features

### 🛡️ Multi-Modal Security
*   **Face Recognition**: InsightFace Buffalo-L model (600K identity training).
*   **Voice Authentication**: SpeechBrain ECAPA-TDNN speaker verification.
*   **Liveness Detection**: Real-time blink eye tracking to prevent photo spoofing.
*   **Fusion Logic**: Adaptive scoring combines face + voice probabilities for high-confidence matches.

### ⚡ Enterprise Performance (New)
*   **PostgreSQL + pgvector**: Native vector similarity search using HNSW indexing.
*   **100x Faster**: Sub-10ms queries for large galleries (vs 500ms+ with SQLite).
*   **Deployment Ready**: Dockerized database with automatic migration and health checks.
*   **Code Optimizations**: 
    - **Vectorized Scoring**: Matrix multiplication replaces linear loop iterations.
    - **Zero-Copy Loading**: Embeddings pre-parsed to NumPy arrays, avoiding I/O overhead.
    - **In-Memory Audio**: Direct BytesIO processing for voice samples reduces disk latency.

### 👮 Admin & Monitoring
*   **Admin Panel**: Secured dashboard to view stats, logs, and captured images.
*   **Security Logs**: Automated logging of all authorized/unauthorized attempts.
*   **Unauthorized Capture**: Automatically saves images of failed access attempts.
*   **Private User Messages**: Securely deliver messages to specific users upon authentication.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Models** | InsightFace (Face) + SpeechBrain (Voice) |
| **Backend** | FastAPI + SQLAlchemy (Async) |
| **Database** | **PostgreSQL + pgvector** (Production) / SQLite (Dev Fallback) |
| **Vector Search** | Hierarchical Navigable Small World (HNSW) Index |
| **Frontend** | Vanilla JS + WebRTC + Chart.js |
| **Container** | Docker Compose (Full Stack) |

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
*Auto-migration scripts will handle the database creation.*

### 4. Access
*   **Live Scanner**: http://localhost:3000
*   **Admin Panel**: http://localhost:3000/admin.html
*   **API Docs**: http://localhost:8001/docs

---

## 🔧 Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| **DB Backend** | Postgres | Automatic fallback to SQLite if unavailable |
| **Face Weight** | 0.85 | Primary biometric factor |
| **Voice Weight** | 0.15 | Secondary booster factor |
| **Liveness** | Blink | Required for "Live" status |

---

## 🛡️ Security Features
*   **Green Box**: Authenticated (Face + Voice + Life)
*   **Red Box**: Unknown subject
*   **📸 Evidence**: Unauthorized images saved to `unauthorized_attempts/`

---

## 🧪 Testing

Run the integration suite inside Docker:

```bash
docker compose run --rm backend pytest tests/ -v
```

---

## ⚖️ License
MIT License
