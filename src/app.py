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

# Path to Haar Cascade for glasses detection
glasses_cascade_path = cv2.data.haarcascades + './haarcascade_eye_tree_eyeglasses.xml'

# Load the glasses detector
glasses_cascade = cv2.CascadeClassifier(glasses_cascade_path)

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

def detect_glasses(image):
    """Detects if glasses are present in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glasses = glasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Return True if glasses are detected
    return len(glasses) > 0

@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    try:
        # Decode base64 menjadi gambar
        registered_photo_data = data['registered_photo']  # Foto terdaftar
        check_in_photo_data = data['photo_check_in']  # Foto absensi/check-in

        registered_photo = base64.b64decode(registered_photo_data)
        check_in_photo = base64.b64decode(check_in_photo_data)

        # Simpan gambar yang terdecode ke file sementara dalam format JPG
        with open('registered.jpg', 'wb') as f:
            f.write(registered_photo)
        with open('check_in.jpg', 'wb') as f:
            f.write(check_in_photo)

        # Proses foto dan lakukan pencocokan
        registered_image_loaded = cv2.imread('registered.jpg')  # Memuat gambar JPG
        check_in_image_loaded = cv2.imread('check_in.jpg')  # Memuat gambar JPG

        # Deteksi kacamata pada gambar check-in
        if detect_glasses(check_in_image_loaded):
            return jsonify({
                'valid': False,
                'message': 'Anda memakai kacamata'
            }), 200

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
        registered_encoding = load_image_and_encode('registered.jpg')
        check_in_encoding = load_image_and_encode('check_in.jpg')

        results = face_recognition.compare_faces([registered_encoding], check_in_encoding)

        return jsonify({
            'valid': results[0],
            'message': 'Face match: true' if results[0] else 'Face match: false'
        }), 200

    except KeyError:
        return jsonify({
            'valid': False,
            'message': 'Foto terdaftar atau check-in tidak ditemukan.'
        }), 400
    except Exception as e:
        return jsonify({
            'valid': False,
            'message': 'Terjadi kesalahan saat memproses gambar: ' + str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/check-eyeglass', methods=['POST'])
def check_eyeglass():
    data = request.json
    photo_data = data.get('check_eyeglass')  # Foto yang akan diperiksa

    if not photo_data:
        return jsonify({
            'valid': False,
            'message': 'Foto tidak ditemukan.'
        }), 400

    try:
        # Decode base64 menjadi gambar
        check_photo = base64.b64decode(photo_data)

        # Simpan gambar yang terdecode ke file sementara dalam format BMP
        with open('check_in.jpg', 'wb') as f:
            f.write(check_photo)

        # Memuat gambar BMP
        check_image_loaded = cv2.imread('check_in.bmp')  # Memuat gambar BMP

        # Deteksi kacamata pada gambar
        if detect_glasses(check_image_loaded):
            return jsonify({
                'valid': False,
                'message': 'Anda memakai kacamata'
            }), 200
        else:
            return jsonify({
                'valid': True,
                'message': 'Tidak memakai kacamata'
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
