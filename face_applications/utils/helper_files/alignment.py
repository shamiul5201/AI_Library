import numpy as np
from . import faceBlendCommon as fbc

def align_face(image_path):
    import dlib
    import cv2
    import numpy as np
    import faceBlendCommon as fbc
    import os

    PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'shape_predictor_5_face_landmarks.dat')

    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Shape predictor model not found at {PREDICTOR_PATH}")

    # Load face detector and shape predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)

    # Read and process the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    points = fbc.getLandmarks(face_detector, landmark_detector, image_rgb)
    if not points:
        raise ValueError("No face landmarks detected in the image.")

    # Normalize image for alignment
    h, w = 600, 600
    aligned_image, _ = fbc.normalizeImagesAndLandmarks((h, w), image, np.array(points))

    return np.uint8(aligned_image * 255)
