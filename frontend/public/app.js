const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const identifyBtn = document.getElementById("identifyBtn");
const refreshUnlock = document.getElementById("refreshUnlock");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera error: " + err));

function captureBlob() {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  return new Promise(resolve => {
    canvas.toBlob(blob => resolve(blob), "image/jpeg");
  });
}

identifyBtn.onclick = async () => {
  const blob = await captureBlob();
  const form = new FormData();
  form.append("file", blob, "capture.jpg");

  result.textContent = "Identifying...";
  fetch("http://localhost:8000/identify/", { method: "POST", body: form })
    .then(r => r.json())
    .then(j => {
      result.textContent = JSON.stringify(j, null, 2);
    }).catch(e => {
      result.textContent = "Error: " + e;
    });
};

refreshUnlock.onclick = async () => {
  fetch("http://localhost:8000/health")
    .then(r => r.json())
    .then(j => result.textContent = "Health: " + JSON.stringify(j))
    .catch(e => result.textContent = "Error: " + e);
};
