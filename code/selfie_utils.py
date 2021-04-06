from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
import imutils
import cv2
import time
import os

def edit_img(frame):
    '''
    This functions gets an image and does the following:
        1. resizes the image
        2. changes the image to gray-scale

    Input:
        frame   numpy array
    '''
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, frame

def take_selfie(frame, total, output_folder):
    '''

    '''
    img_name = os.path.join(output_folder, "selfie_frame_{}.png".format(total))

    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))

    return 


def mouth_aspect_ratio(mouth):
    '''

    '''
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A + B + C) / 3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = avg/D
    return mar #return the mouth aspect ratio

def eye_aspect_ratio(eye):
    '''

    '''
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def init_model():
    print("Initializing model")
    shape_predictor= "dat_files/shape_predictor_68_face_landmarks.dat" #dace_landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    return detector, predictor

def detect_face(np_img, detector, predictor):
    '''

    '''
    faces = []
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    l_eye_start, l_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    r_eye_start, r_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    rects = detector(np_img, 0)
    for rect in rects:
        shape = predictor(np_img, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mouth_start:mouth_end]
        l_eye = shape[l_eye_start:l_eye_end]
        r_eye = shape[r_eye_start:r_eye_end]
        mar = mouth_aspect_ratio(mouth)
        l_ear = eye_aspect_ratio(l_eye)
        r_ear = eye_aspect_ratio(r_eye)

        faces.append({'mar': mar,
                      'mouth_hull' : cv2.convexHull(mouth),
                      'l_ear' : l_ear, 
                      'l_eye_hull' : cv2.convexHull(l_eye),
                      'r_ear' : r_ear,
                      'r_eye_hull' : cv2.convexHull(r_eye)})
    return faces

def show_frame(np_img):
    from PIL import Image
    img = Image.fromarray(np_img)
    img.show()



