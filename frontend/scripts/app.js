let detector;
const BACKEND_URL = "https://your-backend.onrender.com/predict_pose"; // change to your backend URL

async function setupCamera() {
  const video = document.getElementById("webcam");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  });
  video.srcObject = stream;
  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function detectPose() {
  const video = document.getElementById("webcam");
  const pose = await detector.estimatePoses(video);

  if (pose.length > 0 && pose[0].keypoints.length === 17) {
    const keypoints = pose[0].keypoints.map(kp => [kp.y / 480, kp.x / 640, kp.score]);

    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keypoints })
    });

    const data = await response.json();
    if (data.pose) {
      document.getElementById("pose-label").textContent = data.pose;
      document.getElementById("pose-warning").textContent = data.warning || "None";
    }
  }

  requestAnimationFrame(detectPose);
}

async function init() {
  await tf.setBackend('webgl');
  const video = await setupCamera();
  video.play();

  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: 'lightning'
  });

  detectPose();
}

init();
