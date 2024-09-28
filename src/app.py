import cv2
import numpy as np
import face_recognition
import dlib
from scipy.spatial import ConvexHull
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"

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
    
    # Print all detected landmarks
    landmark_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    print("Detected landmarks:", landmark_coords)

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
    """Validates that the unknown face circularity does not differ by more than 0.005 from the known face circularity."""
    max_difference = 0.03  # Updated max difference
    difference = abs(known_circularity - unknown_circularity)
    
    return difference <= max_difference, difference

def check_face_orientation(landmarks):
    """Check if the face is oriented towards the camera."""
    nose_tip = landmarks.part(30)  # Tip of the nose
    left_eye = landmarks.part(36)  # Left eye corner
    right_eye = landmarks.part(45)  # Right eye corner
    
    # Calculate distance ratios for orientation
    eye_line_length = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
    nose_to_eye_distance = np.linalg.norm([nose_tip.x - left_eye.x, nose_tip.y - left_eye.y])

    # Adjust threshold based on empirical data
    if nose_to_eye_distance / eye_line_length < 0.65:  # Adjusted threshold
        return False  # Face is tilted
    return True  # Face is looking forward

@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    registered_photo_data = data.get('registered_photo')  # Registered photo
    check_in_photo_data = data.get('photo_check_in')  # Check-in photo

    # Ensure both photos are present in the received data
    if not registered_photo_data or not check_in_photo_data:
        return jsonify({
            'valid': False,
            'message': 'Registered photo or check-in photo not found.'
        }), 400

    try:
        # Decode base64 to images
        registered_photo = base64.b64decode(registered_photo_data)
        check_in_photo = base64.b64decode(check_in_photo_data)

        # Save decoded images to temporary files
        with open('registered.jpg', 'wb') as f:
            f.write(registered_photo)
        with open('check_in.jpg', 'wb') as f:
            f.write(check_in_photo)

        # Load images
        registered_image_loaded = cv2.imread('registered.jpg')
        check_in_image_loaded = cv2.imread('check_in.jpg')

        # Detect landmarks
        registered_landmarks, _ = detect_landmarks(registered_image_loaded)
        check_in_landmarks, _ = detect_landmarks(check_in_image_loaded)

        # Calculate circularities
        registered_circularity = calculate_circularity(registered_landmarks)
        check_in_circularity = calculate_circularity(check_in_landmarks)

        # Validate circularity
        is_valid, circularity_difference = validate_circularity(registered_circularity, check_in_circularity)
        if not is_valid:
            return jsonify({
                'valid': False,
                'message': f'Circularity difference too large: {circularity_difference}',
            }), 200

        # Compare face encodings
        registered_encoding = load_image_and_encode('registered.jpg')
        check_in_encoding = load_image_and_encode('check_in.jpg')

        # Hitung persentase kecocokan
        match_distance = face_recognition.face_distance([registered_encoding], check_in_encoding)[0]
        match_percentage = (1 - match_distance) * 100  # Convert to percentage

        # Tentukan apakah kecocokan valid berdasarkan threshold 70%
        valid_match = match_percentage > 70

        return jsonify({
            'valid': bool(valid_match),  # Pastikan valid_match adalah boolean
            'match_percentage': float(match_percentage),  # Konversi ke float untuk serialisasi
            'message': 'Face match: true' if valid_match else 'Face match: false'
        }), 200

    except Exception as e:
        return jsonify({
            'valid': False,
            'message': f'Error: {str(e)}'
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
