const video = document.getElementById('video'), canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const statusText = document.getElementById('status'), alarm = document.getElementById('alarmSound');
const captureBtn = document.getElementById('captureBtn');
const processingOverlay = document.getElementById('processingOverlay');
const statusIndicator = document.getElementById('statusIndicator');
const hudTime = document.getElementById('hudTime');

let isProcessing = false;
let alarmRepeatCount = 0;
let alarmInterval = null;
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://127.0.0.1:8000';

// Update HUD time
setInterval(() => {
    const now = new Date();
    hudTime.textContent = now.toLocaleTimeString('en-US', { hour12: false });
}, 1000);

// Initialize Camera
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }).then(s => {
    video.srcObject = s;
    statusText.innerText = "SCANNER ONLINE - AWAITING SUBJECT";
    setInterval(() => { if (!isProcessing) processFrame(false); }, 400);
}).catch(err => {
    statusText.innerText = "CAMERA ACCESS DENIED";
    statusIndicator.classList.add('alert');
});

// Manual Capture Button
captureBtn.addEventListener('click', () => {
    console.log("Manual verification triggered...");
    processFrame(true);
});

// 🔊 Enhanced Alarm System
function playAlarm(isManual) {
    if (alarmInterval) {
        clearInterval(alarmInterval);
        alarmInterval = null;
    }

    alarm.currentTime = 0;
    alarm.play().catch(e => console.warn("Audio blocked by browser policy"));

    if (isManual) {
        alarmRepeatCount = 0;
        alarmInterval = setInterval(() => {
            alarmRepeatCount++;
            if (alarmRepeatCount < 3) {
                alarm.currentTime = 0;
                alarm.play().catch(e => { });
            } else {
                clearInterval(alarmInterval);
                alarmInterval = null;
            }
        }, 500);
    }
}

// Show processing animation
function showProcessing() {
    processingOverlay.classList.add('active');
    setTimeout(() => processingOverlay.classList.remove('active'), 500);
}

async function processFrame(isManual) {
    if (isManual) {
        isProcessing = true;
        showProcessing();
    }

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

                // Update status indicator
                if (isKnown) {
                    statusIndicator.classList.remove('alert');
                } else {
                    statusIndicator.classList.add('alert');
                }

                // 🔊 ALARM for unauthorized
                if (!isKnown) {
                    playAlarm(isManual);

                    if (isManual) {
                        document.body.style.background = '#330000';
                        setTimeout(() => {
                            document.body.style.background = '#050505';
                        }, 300);
                    }
                }

                // 🎨 ENHANCED DRAWING
                const color = isKnown ? '#00ff9d' : '#ff3333';
                const x1 = m.box[1], y1 = m.box[0], x2 = m.box[3], y2 = m.box[2];
                const w = x2 - x1, h = y2 - y1;

                // Main box
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, w, h);

                // Corner accents
                ctx.lineWidth = 4;
                const cornerLen = 20;

                // Top-left
                ctx.beginPath();
                ctx.moveTo(x1, y1 + cornerLen);
                ctx.lineTo(x1, y1);
                ctx.lineTo(x1 + cornerLen, y1);
                ctx.stroke();

                // Top-right
                ctx.beginPath();
                ctx.moveTo(x2 - cornerLen, y1);
                ctx.lineTo(x2, y1);
                ctx.lineTo(x2, y1 + cornerLen);
                ctx.stroke();

                // Bottom-left
                ctx.beginPath();
                ctx.moveTo(x1, y2 - cornerLen);
                ctx.lineTo(x1, y2);
                ctx.lineTo(x1 + cornerLen, y2);
                ctx.stroke();

                // Bottom-right
                ctx.beginPath();
                ctx.moveTo(x2 - cornerLen, y2);
                ctx.lineTo(x2, y2);
                ctx.lineTo(x2, y2 - cornerLen);
                ctx.stroke();

                // Label background
                const label = `${m.name.toUpperCase()}: ${(m.score * 100).toFixed(1)}%`;
                ctx.font = "bold 16px 'Courier New', monospace";
                const textWidth = ctx.measureText(label).width;

                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(x1, y1 - 28, textWidth + 16, 24);

                ctx.fillStyle = color;
                ctx.fillText(label, x1 + 8, y1 - 10);

                // Scanning effect on face
                if (!isKnown) {
                    ctx.strokeStyle = 'rgba(255, 51, 51, 0.3)';
                    ctx.lineWidth = 1;
                    for (let i = y1; i < y2; i += 8) {
                        ctx.beginPath();
                        ctx.moveTo(x1, i);
                        ctx.lineTo(x2, i);
                        ctx.stroke();
                    }
                }

                // Status message
                if (isKnown) {
                    statusText.innerText = `✓ AUTHENTICATED: ${m.name.toUpperCase()}`;
                } else if (isManual) {
                    statusText.innerText = "🚨 SECURITY BREACH - LOGGED TO DATABASE";
                } else {
                    statusText.innerText = "⚠ UNRECOGNIZED SUBJECT DETECTED";
                }
                statusText.style.color = color;
            } else {
                statusText.innerText = "SCANNING...";
                statusText.style.color = '#00ffff';
            }
        } catch (err) {
            statusText.innerText = "⚠ CONNECTION ERROR";
            statusText.style.color = '#ff3333';
        } finally {
            if (isManual) setTimeout(() => { isProcessing = false; }, 1500);
        }
    }, 'image/jpeg', 0.9);
}
