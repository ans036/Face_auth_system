const video = document.getElementById('video'), canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const statusText = document.getElementById('status'), alarm = document.getElementById('alarmSound');
const captureBtn = document.getElementById('captureBtn');

let isProcessing = false;
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://127.0.0.1:8000';

// Initialize Camera
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }).then(s => {
    video.srcObject = s;
    setInterval(() => { if(!isProcessing) processFrame(false); }, 400); 
});

// Manual Capture Button
captureBtn.addEventListener('click', () => {
    console.log("Manual verification triggered...");
    processFrame(true);
});

async function processFrame(isManual) {
    if (isManual) isProcessing = true; // Block live stream during manual verify

    canvas.width = video.videoWidth; 
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        formData.append('is_manual', isManual); // Sends "true" or "false" string

        try {
            const res = await fetch(`${apiBaseUrl}/identify/`, { method: 'POST', body: formData });
            const data = await res.json();
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (data.status === "success" && data.matches.length > 0) {
                const m = data.matches[0];
                const isKnown = m.name !== "Unknown";
                
                // 🔊 ALARM: Only play if Subject is Unknown
                if (!isKnown) {
                    alarm.currentTime = 0; // Reset sound to start
                    alarm.play().catch(e => console.warn("Audio blocked by browser policy"));
                }

                // 🎨 DRAWING
                const color = isKnown ? '#00ff9d' : '#ff3333';
                ctx.strokeStyle = color;
                ctx.lineWidth = 4;
                // m.box is [y1, x1, y2, x2]
                ctx.strokeRect(m.box[1], m.box[0], m.box[3]-m.box[1], m.box[2]-m.box[0]);
                
                ctx.fillStyle = color;
                ctx.font = "bold 20px monospace";
                ctx.fillText(`${m.name.toUpperCase()}: ${(m.score * 100).toFixed(1)}%`, m.box[1], m.box[0] - 10);
                
                statusText.innerText = isKnown ? `SYSTEM: AUTHENTICATED [${m.name}]` : "SYSTEM ALERT: UNAUTHORIZED SUBJECT";
                statusText.style.color = color;
            }
        } catch (err) {
            statusText.innerText = "SERVER COMMUNICATION ERROR";
        } finally {
            if (isManual) setTimeout(() => { isProcessing = false; }, 1500);
        }
    }, 'image/jpeg', 0.9);
}
