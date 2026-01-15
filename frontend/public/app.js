const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');
const statusDot = document.getElementById('statusDot');
const alarm = document.getElementById('alarmSound');
const captureBtn = document.getElementById('captureBtn');
const hudTime = document.getElementById('hudTime');
const livenessStatus = document.getElementById('livenessStatus');
const voiceStatus = document.getElementById('voiceStatus');

// Feature badges
const livenessBadge = document.getElementById('livenessBadge');
const voiceBadge = document.getElementById('voiceBadge');
const faceBadge = document.getElementById('faceBadge');

let isProcessing = false;
let alarmInterval = null;
const apiBaseUrl = window.location.hostname === 'localhost' ? 'http://localhost:8001' : 'http://127.0.0.1:8001';

// Generate unique session ID for blink tracking
const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// Update clock
setInterval(() => {
    hudTime.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}, 1000);

// Initialize Camera AND Microphone
let audioStream = null;
let mediaRecorder = null;
let lastAudioBlob = null;

// Request both video and audio permissions
Promise.all([
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }),
    navigator.mediaDevices.getUserMedia({ audio: true }).catch(err => {
        console.warn("Audio access denied/failed:", err);
        return null;
    })
]).then(([videoStream, audioStreamResult]) => {
    video.srcObject = videoStream;
    audioStream = audioStreamResult;

    statusText.innerText = "SCANNER READY";

    // Update voice badge based on audio availability
    if (audioStream && voiceStatus) {
        voiceStatus.textContent = '🎤 VOICE: READY';
        voiceStatus.style.color = '#00e676';
        if (voiceBadge) {
            voiceBadge.textContent = '🎤 VOICE: READY';
            voiceBadge.className = 'badge badge-active';
        }

        // Initialize MediaRecorder for continuous audio capture
        try {
            // Use supported MIME type
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
            mediaRecorder = new MediaRecorder(audioStream, { mimeType });

            let audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);

                    // Keep only last 5 seconds for instant recognition
                    if (audioChunks.length > 6) {
                        audioChunks = [audioChunks[0], ...audioChunks.slice(-5)];
                    }

                    lastAudioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                }
            };
            // Capture 1-second chunks
            mediaRecorder.start(1000);
            console.log("Audio recording started for voice auth");
        } catch (e) {
            console.error("Failed to start MediaRecorder:", e);
        }

    } else {
        console.warn("No audio stream available for voice auth");
    }

    setInterval(() => { if (!isProcessing) processFrame(false); }, 400);
}).catch((err) => {
    console.error("Camera access failed:", err);
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

// Update liveness indicators
function updateLivenessUI(liveness) {
    if (!livenessStatus || !livenessBadge) return;

    if (liveness.is_live) {
        livenessStatus.textContent = '👁 LIVENESS: VERIFIED';
        livenessStatus.style.color = '#00e676';
        livenessBadge.textContent = '👁 LIVENESS: VERIFIED';
        livenessBadge.className = 'badge badge-active';
    } else if (liveness.blink_detected) {
        livenessStatus.textContent = '👁 LIVENESS: BLINK!';
        livenessStatus.style.color = '#00bcd4';
        livenessBadge.textContent = '👁 LIVENESS: BLINK!';
        livenessBadge.className = 'badge badge-info';
    } else {
        livenessStatus.textContent = '👁 LIVENESS: BLINK TO VERIFY';
        livenessStatus.style.color = '#ffc107';
        livenessBadge.textContent = '👁 LIVENESS: WAITING';
        livenessBadge.className = 'badge badge-warning';
    }
}

// Update voice status
function updateVoiceUI(voiceAvailable) {
    if (!voiceStatus || !voiceBadge) return;

    if (voiceAvailable) {
        voiceStatus.textContent = '🎤 VOICE: READY';
        voiceStatus.style.color = '#00e676';
        voiceBadge.textContent = '🎤 VOICE: READY';
        voiceBadge.className = 'badge badge-active';
    } else {
        voiceStatus.textContent = '🎤 VOICE: STANDBY';
        voiceStatus.style.color = '#666';
        voiceBadge.textContent = '🎤 VOICE: STANDBY';
        voiceBadge.className = 'badge badge-info';
    }
}

async function processFrame(isManual) {
    console.log("[DEBUG] processFrame called, isManual:", isManual, "isProcessing:", isProcessing);

    if (isManual) isProcessing = true;

    // Check if video is ready
    console.log("[DEBUG] video dimensions:", video.videoWidth, "x", video.videoHeight);
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.log("[DEBUG] Video not ready yet, returning");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    console.log("[DEBUG] Canvas drawn, calling toBlob...");

    canvas.toBlob(async (blob) => {
        console.log("[DEBUG] toBlob callback, blob:", blob ? `${blob.size} bytes` : "null");

        if (!blob) {
            console.error("[DEBUG] Blob is null, cannot proceed");
            return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        formData.append('is_manual', isManual);
        formData.append('session_id', sessionId);

        // Append audio only every 3rd request to reduce load
        if (lastAudioBlob && window.frameCount % 3 === 0) {
            formData.append('voice_sample', lastAudioBlob, 'voice.webm');
            console.log("[DEBUG] Voice sample appended");
        }
        window.frameCount = (window.frameCount || 0) + 1;

        console.log("[DEBUG] Sending fetch to:", `${apiBaseUrl}/identify/`);
        try {
            const res = await fetch(`${apiBaseUrl}/identify/`, { method: 'POST', body: formData });
            console.log("[DEBUG] Fetch response status:", res.status);
            const data = await res.json();
            console.log("[DEBUG] Response data:", JSON.stringify(data).substring(0, 200));

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Update liveness UI
            if (data.liveness) {
                updateLivenessUI(data.liveness);
            }

            if (data.status === "success" && data.matches.length > 0) {
                const m = data.matches[0];
                const isKnown = m.name !== "Unknown";
                const isLive = data.liveness?.is_live || false;

                // Check if voice was used
                const modalities = m.modalities || [];
                const voiceUsed = modalities.includes("voice");
                const voiceScore = m.voice_score ? (m.voice_score * 100).toFixed(0) + '%' : 'N/A';

                // Color based on auth + liveness
                let color = '#ff5252'; // Red = unknown
                if (isKnown && isLive) {
                    color = '#00e676'; // Green = known + live
                } else if (isKnown && !isLive) {
                    color = '#ffc107'; // Yellow = known but no liveness yet
                }

                // Update face badge
                if (faceBadge) {
                    faceBadge.textContent = isKnown ? `👤 FACE: ${m.name.toUpperCase()}` : '👤 FACE: UNKNOWN';
                    faceBadge.className = isKnown ? 'badge badge-active' : 'badge badge-warning';
                }

                // Update voice badge color if used
                if (voiceBadge && voiceUsed) {
                    voiceBadge.textContent = `🎤 VOICE: MATCH (${voiceScore})`;
                    voiceBadge.className = 'badge badge-active';
                }

                // Update status
                statusDot.className = 'status-dot';
                if (!isKnown) statusDot.classList.add('alert');
                else if (!isLive) statusDot.classList.add('warning');

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

                // Label with liveness indicator
                const liveTag = isLive ? '✓' : '○';
                const label = `${liveTag} ${m.name.toUpperCase()}: ${(m.score * 100).toFixed(1)}%`;
                ctx.font = "600 13px 'Inter', sans-serif";
                const tw = ctx.measureText(label).width;

                ctx.fillStyle = 'rgba(0,0,0,0.75)';
                ctx.fillRect(x1, y1 - 24, tw + 12, 20);
                ctx.fillStyle = color;
                ctx.fillText(label, x1 + 6, y1 - 9);

                // Status text with MULTIMODAL info
                let methods = ["Face"];
                if (isLive) methods.push("Liveness");
                if (voiceUsed) methods.push("Voice");
                const methodStr = methods.join(" + ");

                if (isKnown && isLive) {
                    statusText.innerText = `✓ FULLY VERIFIED: ${m.name.toUpperCase()} (${methodStr})`;
                } else if (isKnown && !isLive) {
                    statusText.innerText = `DETECTED: ${m.name.toUpperCase()} - BLINK TO COMPLETE (${methodStr})`;
                } else if (isManual) {
                    statusText.innerText = "⚠ UNAUTHORIZED - LOGGED";
                } else {
                    statusText.innerText = "UNRECOGNIZED SUBJECT";
                }
                statusText.style.color = color;
            } else if (data.status === "no_face") {
                statusText.innerText = "NO FACE DETECTED";
                statusText.style.color = '#666';
                if (faceBadge) {
                    faceBadge.textContent = '👤 FACE: SCANNING';
                    faceBadge.className = 'badge badge-info';
                }
            } else {
                statusText.innerText = "SCANNING...";
                statusText.style.color = '#00bcd4';
            }
        } catch (e) {
            console.error("API Error", e);
            statusText.innerText = "CONNECTION ERROR";
            statusText.style.color = '#ff5252';
        } finally {
            if (isManual) setTimeout(() => isProcessing = false, 1500);
        }
    }, 'image/jpeg', 0.9);
}
