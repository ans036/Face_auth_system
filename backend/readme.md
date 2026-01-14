1\. The GitHub README (README.md)

Create this file in the root of your project (D:/face\_auth\_system/README.md).



Markdown



\# Face Authentication System



A robust, containerized Face Recognition \& Authentication system built with \*\*FastAPI\*\*, \*\*MediaPipe\*\*, \*\*ONNX Runtime (ArcFace)\*\*, and \*\*React\*\*.



\## 🚀 Features

\* \*\*Real-time Face Detection\*\*: Powered by MediaPipe.

\* \*\*Face Embeddings\*\*: Uses high-performance ArcFace ONNX models.

\* \*\*Automated Gallery\*\*: Builds a searchable face database from image folders on startup.

\* \*\*Secure Logging\*\*: Full audit trail of access attempts stored in `security.log`.

\* \*\*Dockerized Architecture\*\*: Simplified deployment with Docker Compose.



\## 🛠️ Project Structure

```text

face\_auth\_system/

├── backend/            # FastAPI Application

├── frontend/           # Node.js/Express Web UI

├── database/           # Known faces (Organized by folder name)

├── models/             # AI Model storage (.onnx, .tflite)

├── config/             # System thresholds and security settings

└── docker-compose.yaml # Orchestration

📦 Prerequisites

Docker \& Docker Compose



An internet connection (for the initial model download)



🚦 Quick Start

Prepare the Gallery: Place images of authorized users in database/<username>/. For example: database/Anish/pic1.jpg.



Download Models: Run the download script in the models folder:



Bash



bash models/download\_model.sh

Launch System:



Bash



docker compose up --build

Access:



Frontend: http://localhost:3000



Backend API: http://localhost:8000/docs



🔒 Security

All security events, including scores and access reasons, are logged to security.log using standard JSON format for easy auditing.





---



\### 2. The `.gitignore` File

Create this file in the \*\*root\*\* of your project to prevent uploading temporary Docker files or your database to GitHub.



```text

\# Python

\_\_pycache\_\_/

\*.py\[cod]

\*$py.class

venv/

.env



\# Docker \& Database

face\_auth.db

security.log

unlocked.txt



\# Node.js

node\_modules/

npm-debug.log



\# Models (Usually too large for GitHub)

models/\*.onnx

models/\*.tflite



\# OS files

.DS\_Store

Thumbs.db

