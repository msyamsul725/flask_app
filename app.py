import base64
import face_recognition
import dlib
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

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
    known_image_data = data['photo_attendance']
    unknown_image_data = data['photo_testing']

    # Decode base64 menjadi gambar
    known_image = base64.b64decode(known_image_data)
    unknown_image = base64.b64decode(unknown_image_data)

    # Simpan gambar yang terdecode ke file sementara
    with open('known.jpg', 'wb') as f:
        f.write(known_image)
    with open('unknown.jpg', 'wb') as f:
        f.write(unknown_image)

    try:
        # Load dan proses gambar
        known_image_loaded = cv2.imread('known.jpg')
        unknown_image_loaded = cv2.imread('unknown.jpg')

        # Deteksi landmark dan circularity
        known_landmarks, _ = detect_landmarks(known_image_loaded)
        unknown_landmarks, _ = detect_landmarks(unknown_image_loaded)

        known_circularity = calculate_circularity(known_landmarks)
        unknown_circularity = calculate_circularity(unknown_landmarks)

        # Validasi circularity
        is_valid, circularity_difference = validate_circularity(known_circularity, unknown_circularity)

        if not is_valid:
            return jsonify({
                'message': 'Face match: false',
                'circularity_difference': circularity_difference
            }), 200

        # Encode wajah untuk perbandingan
        known_encoding = load_image_and_encode('known.jpg')
        unknown_encoding = load_image_and_encode('unknown.jpg')

        # Bandingkan wajah
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)

        return jsonify({'message': 'Face match: true' if results[0] else 'Face match: false'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
