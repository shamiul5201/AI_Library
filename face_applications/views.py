import os
import cv2
import dlib
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .utils.helper_files.faceBlendCommon import getLandmarks, normalizeImagesAndLandmarks

PREDICTOR_PATH = os.path.join(settings.BASE_DIR, "face_applications/utils/models/shape_predictor_5_face_landmarks.dat")

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)

def face_alignment_view(request):
    if request.method == "POST":
        try:
            # Decode the base64 image data
            data = request.body.decode("utf-8")
            image_data = base64.b64decode(data.split(",")[1])
            image = Image.open(BytesIO(image_data))

            # Create temporary directories for input and output images
            temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            input_image_path = os.path.join(temp_dir, "input_image.jpg")
            output_image_path = os.path.join(temp_dir, "aligned_image.jpg")

            # Save the uploaded image
            image.save(input_image_path)

            # Align the image and save the result
            align_image(input_image_path, output_image_path)

            # Construct URLs for the saved images
            input_image_url = f"{settings.MEDIA_URL}temp/input_image.jpg"
            output_image_url = f"{settings.MEDIA_URL}temp/aligned_image.jpg"

            return JsonResponse({
                "input_image_url": input_image_url,
                "output_image_url": output_image_url,
            })
        except Exception as e:
            return JsonResponse({"error": f"Face alignment failed: {str(e)}"})
    return render(request, "face_applications/face_alignment.html")

def align_image(input_image_path, output_image_path):
    # Read the input image
    im = cv2.imread(input_image_path)
    im_dlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Get landmarks and normalize the face
    points = getLandmarks(face_detector, landmark_detector, im_dlib)
    points = np.array(points)
    im = np.float32(im) / 255.0
    h, w = 600, 600
    im_norm, _ = normalizeImagesAndLandmarks((h, w), im, points)
    im_norm = np.uint8(im_norm * 255)

    # Save the aligned image
    cv2.imwrite(output_image_path, im_norm)



