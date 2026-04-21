const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const detectBtn = document.getElementById("detectBtn");
const result = document.getElementById("result");

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    result.textContent = "Camera access failed: " + err.message;
  }
}

detectBtn.addEventListener("click", async () => {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL("image/jpeg");

  result.textContent = "Detecting...";

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: imageData })
  });

  const data = await response.json();

  if (data.error) {
    result.textContent = data.error;
  } else {
    result.textContent = `Emotion: ${data.emotion} (${(data.confidence * 100).toFixed(1)}%)`;
  }
});

startCamera();