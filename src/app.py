import base64
import face_recognition
import dlib
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to the shape predictor model
shape_predictor_path = './shape_predictor_68_face_landmarks.dat'

def load_image_and_encode(image_data):
    """Loads an image from base64, converts to RGB, and returns face encoding."""
    image = face_recognition.load_image_file(image_data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(image)
    return encoding[0] if encoding else None

def crop_and_resize_face(image, face, size=(200, 200)):
    """Crops the face from the image and resizes it to a fixed size."""
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    face_image = image[y:y+h, x:x+w]
    return cv2.resize(face_image, size)  # Resize to fixed size

def detect_landmarks(image):
    """Detects facial landmarks using dlib shape predictor."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    
    faces = detector(gray)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
    
    face = faces[0]  # Assuming one face per image
    image = crop_and_resize_face(image, face)  # Use new crop and resize function
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, dlib.rectangle(0, 0, image.shape[1], image.shape[0]))
    return landmarks, image

def calculate_circularity(landmarks):
    """Calculates the circularity of the face based on landmarks using convex hull."""
    points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(17, 68)])
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Calculate the perimeter
    perimeter = np.sum([np.linalg.norm(points[hull.vertices[i]] - points[hull.vertices[(i+1) % len(hull.vertices)]]) for i in range(len(hull.vertices))])
    
    # Calculate the area
    area = hull.volume
    
    # Calculate circularity: 4π * (area / perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

def validate_circularity(known_circularity, unknown_circularity):
    """Validates that the unknown face circularity does not differ by more than 0.01 from the known face circularity."""
    max_difference = 0.01
    difference = abs(known_circularity - unknown_circularity)
    
    return difference <= max_difference, difference

@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    registered_photo_data = data['registered_photo']  # Foto terdaftar
    check_in_photo_data = data['photo_check_in']  # Foto absensi/check-in

    # Decode base64 menjadi gambar
    registered_photo = base64.b64decode(registered_photo_data)
    check_in_photo = base64.b64decode(check_in_photo_data)

    # Simpan gambar yang terdecode ke file sementara dalam format BMP
    with open('registered.jpg', 'wb') as f:
        f.write(registered_photo)
    with open('check_in.jpg', 'wb') as f:
        f.write(check_in_photo)

    try:
        # Proses foto dan lakukan pencocokan
        registered_image_loaded = cv2.imread('registered.bmp')  # Memuat gambar BMP
        check_in_image_loaded = cv2.imread('check_in.bmp')  # Memuat gambar BMP

        # Lanjutkan dengan deteksi landmark dan circularity seperti semula
        registered_landmarks, _ = detect_landmarks(registered_image_loaded)
        check_in_landmarks, _ = detect_landmarks(check_in_image_loaded)

        registered_circularity = calculate_circularity(registered_landmarks)
        check_in_circularity = calculate_circularity(check_in_landmarks)

        # Validasi circularity
        is_valid, circularity_difference = validate_circularity(registered_circularity, check_in_circularity)
        if not is_valid:
            return jsonify({
                'valid': False,
                'circularity_difference': circularity_difference
            }), 200

        # Bandingkan encoding wajah
        registered_encoding = load_image_and_encode('registered.bmp')  # Ganti menjadi BMP
        check_in_encoding = load_image_and_encode('check_in.bmp')  # Ganti menjadi BMP

        results = face_recognition.compare_faces([registered_encoding], check_in_encoding)

        return jsonify({
            'valid': results[0],
            'message': 'Face match: true' if results[0] else 'Face match: false'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

