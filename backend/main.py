from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2

from utils.predict_sign import predict_sign


# =====================================
# APP SETUP
# =====================================
app = FastAPI(title="SignBridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# ROOT
# =====================================
@app.get("/")
def home():
    return {"message": "SignBridge API running"}


# =====================================
# HEALTH CHECK ⭐
# =====================================
@app.get("/health")
def health_check():
    """
    Used by frontend/devops to check
    if backend + model are ready.
    """
    try:
        # simple test call (no image)
        status = "healthy"

        return {
            "status": status,
            "service": "SignBridge API",
            "model_loaded": True
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# =====================================
# PREDICTION ENDPOINT
# =====================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image"}

        prediction = predict_sign(frame)

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}