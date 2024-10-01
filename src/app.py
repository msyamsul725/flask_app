import cv2
import numpy as np
import face_recognition
import dlib
from scipy.spatial import ConvexHull
from flask import Flask, request, jsonify
import base64
from PIL import Image

app = Flask(__name__)

shape_predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
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

        # Load and encode images
        registered_encoding = load_image_and_encode('registered.jpg')
        check_in_encoding = load_image_and_encode('check_in.jpg')

        if registered_encoding is None:
            return jsonify({
                'valid': False,
                'message': 'Registered photo does not contain a recognizable face.'
            }), 200
        
        if check_in_encoding is None:
            return jsonify({
                'valid': False,
                'message': 'Check-in photo does not contain a recognizable face.'
            }), 200

        # Hitung persentase kecocokan
        match_distance = face_recognition.face_distance([registered_encoding], check_in_encoding)[0]
        match_percentage = (1 - match_distance) * 100  # Convert to percentage

        # Tentukan apakah kecocokan valid berdasarkan threshold 45%
        valid_match = match_percentage > 60

        return jsonify({
            'valid': bool(valid_match),  # Pastikan valid_match adalah boolean
            'match_percentage': float(match_percentage),  # Konversi ke float untuk serialisasi
            'message': 'Face match: true' if valid_match else 'Face match: false'
        }), 200

    except ValueError as ve:
        return jsonify({
            'valid': False,
            'message': str(ve)
        }), 400

    except Exception as e:
        return jsonify({
            'valid': False,
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resize image function
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image

    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        r = height / float(h)
        dim = (int(w * r), height)

    return cv2.resize(image, dim, interpolation=inter)




def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

#==============================================================================   
#   2.绘制回归线 & 找瞳孔函数
#       输入：图片 & numpy格式的landmarks
#       输出：左瞳孔坐标 & 右瞳孔坐标
#==============================================================================   
def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) #画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

#==============================================================================   
#   3.人脸对齐函数
#       输入：图片 & 左瞳孔坐标 & 右瞳孔坐标
#       输出：对齐后的人脸图片
#============================================================================== 
def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)
    scale = desired_dist / dist 
    angle = np.degrees(np.arctan2(dy,dx)) 
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)

    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))
    
    return aligned_face

#==============================================================================   
#   4.是否戴眼镜判别函数
#       输入：对齐后的人脸图片
#       输出：判别值(True/False)
#============================================================================== 
def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    edgeness = sobel_y 
    
    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] 
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1*0.3 + measure_2*0.7

    # 判别值
    judge = measure > 0.15  
    return judge

#==============================================================================   
#   5. 处理图像的主要函数
#       输入：图片路径
#       输出：是否戴眼镜的结果
#============================================================================== 
def process_image(image_path):
    predictor_path = "./shape_predictor_5_face_landmarks.dat"  # Jalur data pelatihan landmark wajah
    detector = dlib.get_frontal_face_detector()  # Detektor wajah
    predictor = dlib.shape_predictor(predictor_path)  # Detektor landmark wajah

    # Membaca gambar
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error loading image.")
        return False  # Kembalikan False jika gambar tidak dapat dimuat
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    rects = detector(gray, 1)

    if len(rects) == 0:
        print("No faces detected.")
        return False  # Kembalikan False jika tidak ada wajah yang terdeteksi

    for i, rect in enumerate(rects):
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)

        # Linear regression
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

        # Penyelarasan wajah
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

        # Menentukan apakah memakai kacamata
        judge = judge_eyeglass(aligned_face)

        return judge  # Kembalikan nilai judge

    # Jika tidak ada wajah terdeteksi, kembalikan False
    return False

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

        # Simpan gambar yang terdecode ke file sementara dalam format JPG
        with open('check_in.jpg', 'wb') as f:
            f.write(check_photo)

        # Panggil fungsi glasses_detector dengan path ke file gambar
        if process_image('check_in.jpg'):
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
        return jsonify({
            'valid': False,
            'message': str(e)
        }), 500



if __name__ == '__main__':
    app.run(debug=True)
