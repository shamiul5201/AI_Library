import os
import cv2
import dlib
import numpy as np
import tempfile
from django.shortcuts import render
from django.conf import settings
from .utils.helper_files.faceBlendCommon import getLandmarks, normalizeImagesAndLandmarks


# Path to shape predictor model
PREDICTOR_PATH = os.path.join(settings.BASE_DIR, "face_applications/utils/models/shape_predictor_5_face_landmarks.dat")

# Load Dlib models once
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)

def face_alignment_view(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        uploaded_image = request.FILES["image"]

        # Temporary storage in MEDIA_ROOT/temp
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        input_image_path = os.path.join(temp_dir, "input_image.jpg")
        output_image_path = os.path.join(temp_dir, "aligned_image.jpg")

        try:
            # Save the uploaded image
            with open(input_image_path, "wb") as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)

            # Read the image
            im = cv2.imread(input_image_path)
            im_dlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Perform face alignment
            points = getLandmarks(face_detector, landmark_detector, im_dlib)
            points = np.array(points)
            im = np.float32(im) / 255.0
            h, w = 600, 600  # Output dimensions
            im_norm, _ = normalizeImagesAndLandmarks((h, w), im, points)
            im_norm = np.uint8(im_norm * 255)

            # Save the aligned image
            cv2.imwrite(output_image_path, im_norm)

            # Generate URLs for temporary images
            context["input_image_url"] = f"{settings.MEDIA_URL}temp/input_image.jpg"
            context["output_image_url"] = f"{settings.MEDIA_URL}temp/aligned_image.jpg"

        except Exception as e:
            context["error"] = f"Face alignment failed: {str(e)}"

    return render(request, "face_applications/face_alignment.html", context)