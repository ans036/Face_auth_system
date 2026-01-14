const video = document.getElementById('video'), canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const statusText = document.getElementById('status'), alarm = document.getElementById('alarmSound');
const captureBtn = document.getElementById('captureBtn');

let isProcessing = false;
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://127.0.0.1:8000';

navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
    video.srcObject = s;
    setInterval(() => processFrame(false), 300); // Live loop (isManual = false)
});

captureBtn.addEventListener('click', () => processFrame(true));

async function processFrame(isManual) {
    if (isProcessing && !isManual) return;
    if (isManual) isProcessing = true;

    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob);
        formData.append('is_manual', isManual); // Tell backend to log this

        try {
            const res = await fetch(`${apiBaseUrl}/identify/`, { method: 'POST', body: formData });
            const data = await res.json();
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (data.status === "success" && data.matches.length > 0) {
                const m = data.matches[0];
                const isKnown = m.name !== "Unknown";
                
                // Trigger Alarm if Unknown
                if (!isKnown) alarm.play();

                // Draw Box & Probability
                ctx.strokeStyle = isKnown ? '#00ff9d' : '#ff3333';
                ctx.lineWidth = 4;
                ctx.strokeRect(m.box[1], m.box[0], m.box[3]-m.box[1], m.box[2]-m.box[0]);
                
                ctx.fillStyle = isKnown ? '#00ff9d' : '#ff3333';
                ctx.font = "bold 18px Arial";
                ctx.fillText(`${m.name}: ${(m.score * 100).toFixed(1)}%`, m.box[1], m.box[0] - 10);
                
                statusText.innerText = isKnown ? `MATCH: ${m.name}` : "UNAUTHORIZED SUBJECT DETECTED";
                statusText.style.color = ctx.fillStyle;
            }
        } finally {
            if (isManual) setTimeout(() => isProcessing = false, 1000);
        }
    }, 'image/jpeg');
}