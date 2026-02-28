import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from collections import deque

prediction_buffer = deque(maxlen=15)

# =========================================
# LOAD TRAINED MODEL
# =========================================
MODEL_PATH = "../ai_model/sign_pipeline.pkl"
ENCODER_PATH = "../ai_model/label_encoder.pkl"
TASK_MODEL = "models/hand_landmarker.task"

pipeline = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# =========================================
# MEDIAPIPE HAND LANDMARKER SETUP
# =========================================
base_options = python.BaseOptions(model_asset_path=TASK_MODEL)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================================
# NORMALIZE LANDMARKS (same as training)
# =========================================
def normalize_landmarks(landmarks):

    for hand in range(2):
        base = hand * 63

        wrist_x = landmarks[base]
        wrist_y = landmarks[base + 1]
        wrist_z = landmarks[base + 2]

        if wrist_x == 0 and wrist_y == 0:
            continue

        for i in range(21):
            idx = base + i * 3
            landmarks[idx] -= wrist_x
            landmarks[idx + 1] -= wrist_y
            landmarks[idx + 2] -= wrist_z

    return landmarks


# =========================================
# EXTRACT LANDMARK FEATURES
# =========================================
def extract_landmarks(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    landmarks = np.zeros(128)

    if result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks[:2]):

            handedness = result.handedness[hand_index][0].category_name
            landmarks[126 + hand_index] = 1 if handedness == "Right" else 0

            for i, lm in enumerate(hand_landmarks):
                base = hand_index * 63 + i * 3
                landmarks[base] = lm.x
                landmarks[base + 1] = lm.y
                landmarks[base + 2] = lm.z

    landmarks = normalize_landmarks(landmarks)

    return landmarks


# =========================================
# PREDICT SIGN
# =========================================
def predict_sign(frame):

    features = extract_landmarks(frame)

    # no hand detected
    if np.all(features[:126] == 0):
        return "No hand detected"

    prediction = pipeline.predict([features])[0]
    label = encoder.inverse_transform([prediction])[0]

    # prediction_buffer.append(label)

    # # majority voting
    # final_prediction = max(set(prediction_buffer),
    #                     key=prediction_buffer.count)

    return label


# =========================================
# REAL-TIME CAMERA TEST
# =========================================
def run_camera():

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        sign = predict_sign(frame)

        # display prediction
        cv2.putText(
            frame,
            f"Prediction: {sign}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("SignBridge Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================
# RUN DIRECTLY
# =========================================
if __name__ == "__main__":
    run_camera()