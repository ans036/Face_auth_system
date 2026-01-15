const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');
const statusDot = document.getElementById('statusDot');
const alarm = document.getElementById('alarmSound');
const captureBtn = document.getElementById('captureBtn');
const hudTime = document.getElementById('hudTime');

let isProcessing = false;
let alarmInterval = null;
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://127.0.0.1:8000';

// Update clock
setInterval(() => {
    hudTime.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}, 1000);

// Initialize Camera
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
        video.srcObject = stream;
        statusText.innerText = "SCANNER READY";
        setInterval(() => { if (!isProcessing) processFrame(false); }, 400);
    })
    .catch(() => {
        statusText.innerText = "CAMERA ACCESS DENIED";
        statusDot.classList.add('alert');
    });

// Manual Capture
captureBtn.addEventListener('click', () => processFrame(true));

// Alarm system
function playAlarm(repeat) {
    if (alarmInterval) clearInterval(alarmInterval);

    alarm.currentTime = 0;
    alarm.play().catch(() => { });

    if (repeat) {
        let count = 0;
        alarmInterval = setInterval(() => {
            if (++count < 3) {
                alarm.currentTime = 0;
                alarm.play().catch(() => { });
            } else {
                clearInterval(alarmInterval);
            }
        }, 400);
    }
}

async function processFrame(isManual) {
    if (isManual) isProcessing = true;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        formData.append('is_manual', isManual);

        try {
            const res = await fetch(`${apiBaseUrl}/identify/`, { method: 'POST', body: formData });
            const data = await res.json();

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.status === "success" && data.matches.length > 0) {
                const m = data.matches[0];
                const isKnown = m.name !== "Unknown";
                const color = isKnown ? '#00e676' : '#ff5252';

                // Update status
                statusDot.classList.toggle('alert', !isKnown);

                // Alarm for unknown
                if (!isKnown) {
                    playAlarm(isManual);
                    if (isManual) {
                        document.body.style.background = '#1a0000';
                        setTimeout(() => document.body.style.background = '#0a0a0a', 200);
                    }
                }

                // Draw face box
                const [y1, x1, y2, x2] = m.box;
                const w = x2 - x1, h = y2 - y1;

                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, w, h);

                // Corner highlights
                const corner = 15;
                ctx.lineWidth = 3;

                ctx.beginPath();
                ctx.moveTo(x1, y1 + corner); ctx.lineTo(x1, y1); ctx.lineTo(x1 + corner, y1);
                ctx.moveTo(x2 - corner, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + corner);
                ctx.moveTo(x1, y2 - corner); ctx.lineTo(x1, y2); ctx.lineTo(x1 + corner, y2);
                ctx.moveTo(x2 - corner, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - corner);
                ctx.stroke();

                // Label
                const label = `${m.name.toUpperCase()}: ${(m.score * 100).toFixed(1)}%`;
                ctx.font = "600 13px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;

                ctx.fillStyle = 'rgba(0,0,0,0.75)';
                ctx.fillRect(x1, y1 - 24, tw + 12, 20);
                ctx.fillStyle = color;
                ctx.fillText(label, x1 + 6, y1 - 9);

                // Status text
                statusText.innerText = isKnown
                    ? `AUTHENTICATED: ${m.name.toUpperCase()}`
                    : isManual ? "⚠ UNAUTHORIZED - LOGGED" : "UNRECOGNIZED SUBJECT";
                statusText.style.color = color;
            } else {
                statusText.innerText = "SCANNING...";
                statusText.style.color = '#00bcd4';
            }
        } catch {
            statusText.innerText = "CONNECTION ERROR";
            statusText.style.color = '#ff5252';
        } finally {
            if (isManual) setTimeout(() => isProcessing = false, 1500);
        }
    }, 'image/jpeg', 0.9);
}
