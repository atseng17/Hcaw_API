import dlib
import cv2
import numpy as np

def init_lmark_model():
    # Load the face detection model from dlib
    detector = dlib.get_frontal_face_detector()

    # Load the facial landmark predictor model from dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def get_landmarks(predictor, face, gray, image):
    # Use the predictor to find facial landmarks for the current face
    landmarks = predictor(gray, face)

    # Convert the landmarks to a NumPy array
    landmarks = np.array([(landmark.x, landmark.y) for landmark in landmarks.parts()])

    return landmarks


def get_stats(image_path, landmarks, image, draw=False):
    # Initialize an array to store left eye landmarks
    left_eye_landmarks = []

    # Assuming you've already detected and obtained facial landmarks
    for (x, y) in landmarks[36:42]:  # Indices 36 to 41 correspond to the left eye landmarks
        left_eye_landmarks.append((x, y))

    # Extract x-coordinates of left eye landmarks
    x_coordinates = [landmark[0] for landmark in left_eye_landmarks]

    # Calculate the width of the left eye as the Euclidean distance between the leftmost and rightmost landmarks
    left_eye_width = np.sqrt((max(x_coordinates) - min(x_coordinates))**2)
    
    eye_gap = abs(landmarks[39][0]-landmarks[42][0])
    
    nose_width = abs(landmarks[31][0]-landmarks[35][0])
    nose_height = abs(landmarks[30][0]-landmarks[33][0])
    
    mouth_width = abs(landmarks[48][0]-landmarks[54][0])
    face_max_width = abs(landmarks[0][0]-landmarks[16][0])

    if draw:
        # Loop through the landmarks and draw them on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # eye gap
        cv2.line(image, (landmarks[39][0],landmarks[39][1]), (landmarks[42][0],landmarks[42][1]), (255, 0, 0), 2)
        # eye width
        cv2.line(image, (landmarks[37][0],landmarks[37][1]), (landmarks[41][0],landmarks[41][1]), (255, 0, 0), 2)
        # nose width
        cv2.line(image, (landmarks[31][0],landmarks[31][1]), (landmarks[35][0],landmarks[35][1]), (0, 0, 255), 2)
        # nose height
        cv2.line(image, (landmarks[30][0],landmarks[30][1]), (landmarks[33][0],landmarks[33][1]), (0, 0, 255), 2)
        # mouth width
        cv2.line(image, (landmarks[48][0],landmarks[48][1]), (landmarks[54][0],landmarks[54][1]), (255, 0, 255), 2)
        # face max width
        cv2.line(image, (landmarks[0][0],landmarks[0][1]), (landmarks[16][0],landmarks[16][1]), (255, 0, 255), 2)
        cv2.imwrite(image_path.replace("samples","annotation"), image)

    e_w_g = left_eye_width/eye_gap
    nose_w_h = nose_width/nose_height
    m_w_f_w = mouth_width/face_max_width
    print(f"eye width to gap: {e_w_g} pixels")
    print(f"nose width to height ratio: {nose_w_h}")
    print(f"mouth width to face max width: {m_w_f_w}")
    return e_w_g, nose_w_h, m_w_f_w



def get_single_image_stats(image_path, detector, predictor):
    image, gray = load_and_preprocess(image_path)
    # Use the face detector to find faces in the grayscale image
    faces = detector(gray)
    if len(faces)>1:
        raise ValueError("Currently the api does not acceps image with more than one face")
    face = faces[0]
    # get land marks
    landmarks = get_landmarks(predictor, face, gray, image)
    # get facial features
    e_w_g, nose_w_h, m_w_f_w = get_stats(image_path, landmarks, image, draw=True)
    return e_w_g, nose_w_h, m_w_f_w