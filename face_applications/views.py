import base64
import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def face_alignment_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_data = data.get("image")

            if not image_data:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Decode the base64 image
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_image = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Perform face alignment
            aligned_face, success = align_face(image)
            if not success:
                return JsonResponse({"error": "Face alignment failed."}, status=400)

            # Convert images to base64 strings for frontend display
            _, original_encoded = cv2.imencode('.jpg', image)
            _, aligned_encoded = cv2.imencode('.jpg', aligned_face)

            original_base64 = base64.b64encode(original_encoded).decode('utf-8')
            aligned_base64 = base64.b64encode(aligned_encoded).decode('utf-8')

            return JsonResponse({
                "original_image": f"data:image/jpeg;base64,{original_base64}",
                "aligned_image": f"data:image/jpeg;base64,{aligned_base64}",
            }, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "face_applications/face_alignment.html")


def align_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, False

    # Extract and align the first detected face
    x, y, w, h = faces[0]
    aligned_face = image[y:y+h, x:x+w]
    aligned_face = cv2.resize(aligned_face, (200, 200))

    return aligned_face, True


# import os
# import cv2
# import dlib
# import numpy as np
# import tempfile
# from django.shortcuts import render
# from django.conf import settings
# from .utils.helper_files.faceBlendCommon import getLandmarks, normalizeImagesAndLandmarks


# # Path to shape predictor model
# PREDICTOR_PATH = os.path.join(settings.BASE_DIR, "face_applications/utils/models/shape_predictor_5_face_landmarks.dat")

# # Load Dlib models once
# face_detector = dlib.get_frontal_face_detector()
# landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)

# def face_alignment_view(request):
#     context = {}

#     if request.method == "POST" and request.FILES.get("image"):
#         uploaded_image = request.FILES["image"]

#         # Temporary storage in MEDIA_ROOT/temp
#         temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
#         os.makedirs(temp_dir, exist_ok=True)
#         input_image_path = os.path.join(temp_dir, "input_image.jpg")
#         output_image_path = os.path.join(temp_dir, "aligned_image.jpg")

#         try:
#             # Save the uploaded image
#             with open(input_image_path, "wb") as f:
#                 for chunk in uploaded_image.chunks():
#                     f.write(chunk)

#             # Read the image
#             im = cv2.imread(input_image_path)
#             im_dlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#             # Perform face alignment
#             points = getLandmarks(face_detector, landmark_detector, im_dlib)
#             points = np.array(points)
#             im = np.float32(im) / 255.0
#             h, w = 600, 600  # Output dimensions
#             im_norm, _ = normalizeImagesAndLandmarks((h, w), im, points)
#             im_norm = np.uint8(im_norm * 255)

#             # Save the aligned image
#             cv2.imwrite(output_image_path, im_norm)

#             # Generate URLs for temporary images
#             context["input_image_url"] = f"{settings.MEDIA_URL}temp/input_image.jpg"
#             context["output_image_url"] = f"{settings.MEDIA_URL}temp/aligned_image.jpg"

#         except Exception as e:
#             context["error"] = f"Face alignment failed: {str(e)}"

#     return render(request, "face_applications/face_alignment.html", context)