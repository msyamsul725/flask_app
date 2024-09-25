import face_recognition
import dlib
import cv2
import sys
import numpy as np
from scipy.spatial import ConvexHull
 
# Paths from command line arguments
known_image_path = sys.argv[1]
unknown_image_path = sys.argv[2]
 
# Path to the shape predictor model
shape_predictor_path = 'D:/flutter referensi 2024/absensi_online_trackmate/absensi_online_server/python_app/shape_predictor_68_face_landmarks.dat'
 
def load_image_and_encode(path):
    """Loads an image, converts to RGB, and returns face encoding."""
    print(f"Loading image from: {path}")
    image = face_recognition.load_image_file(path)
 
    # Convert image to RGB for face_recognition
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    encoding = face_recognition.face_encodings(image)
    return encoding[0] if encoding else None
 
def crop_face(image, face):
    """Crops the face from the image based on the detected face bounding box."""
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    return image[y:y+h, x:x+w]
 
def detect_landmarks(image):
    """Detects facial landmarks using dlib shape predictor."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
 
    faces = detector(gray)
 
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
 
    face = faces[0]  # Assuming one face per image
    image = crop_face(image, face)
 
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
 
    print(f"Face circularity (convex hull): {circularity}")
    return circularity
 
def validate_circularity(known_circularity, unknown_circularity):
    """Validates that the unknown face circularity does not differ by more than 0.01 from the known face circularity."""
    max_difference = 0.01
    difference = abs(known_circularity - unknown_circularity)
    print(f"Known face circularity: {known_circularity}, Unknown face circularity: {unknown_circularity}, Difference: {difference}")
 
    if difference <= max_difference:
        return True
    else:
        return False
 
try:
    # Load and process the images
    known_image = cv2.imread(known_image_path)
    unknown_image = cv2.imread(unknown_image_path)
 
    # Check if image loading was successful
    if known_image is None or unknown_image is None:
        print("Error: One of the images could not be loaded.")
        sys.exit(1)
 
    # Detect landmarks and circularity
    known_landmarks, _ = detect_landmarks(known_image)
    unknown_landmarks, _ = detect_landmarks(unknown_image)
 
    known_circularity = calculate_circularity(known_landmarks)
    unknown_circularity = calculate_circularity(unknown_landmarks)
 
    # Validate circularity
    if not validate_circularity(known_circularity, unknown_circularity):
        print("Face circularity validation failed: false")
        sys.exit(0)
 
    # Encode faces for comparison
    known_encoding = load_image_and_encode(known_image_path)
    unknown_encoding = load_image_and_encode(unknown_image_path)
 
    # Check if encoding was successful
    if known_encoding is None or unknown_encoding is None:
        print("Error: One of the images could not be processed.")
        sys.exit(1)
 
    # Compare the faces
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
 
    print("Results OK")
    print("true" if results[0] else "false")
 
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)