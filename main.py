from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import mediapipe as mp
import cv2
import numpy as np
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv


app = FastAPI()
load_dotenv()

# Initialize Cloudinary with environment variables (set your cloud_name, api_key, and api_secret)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


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

    if face_count == 1:
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        try:
            upload_result = cloudinary.uploader.upload(image_bytes, folder="faces")
            image_url = upload_result.get("secure_url")

            return JSONResponse({
                "result": True,
                "image_url": image_url
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    else:
        return JSONResponse({
            "result": False,
            "message": f"Face count is {face_count}, not 1."
        })

@app.get("/ping")
async def ping():
    return {"message": "API is working!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
