from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
import os

app = FastAPI()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

async def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        return len(results.detections)
    return 0

@app.post("/count_faces")
async def count_faces(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face_count = await detect_faces(image)

    return JSONResponse({
        "result": face_count == 1
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Get port from environment variable or use 8000
    uvicorn.run(app, host="0.0.0.0", port=port)  # Bind to 0.0.0.0 to allow external access
