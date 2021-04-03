from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
import imutils
import cv2

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
    return gray

def smile(mouth):
    '''

    '''
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A + B + C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar = avg/D
    return mar

def detect_face(np_img):
    '''

    '''
    shape_predictor= "code/dat_files/shape_predictor_68_face_landmarks.dat" #dace_landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    l_eye_start, l_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    r_eye_start, r_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    rects = detector(np_img, 0)
    for rect in rects:
        print("running on:  ", rect)
        shape = predictor(np_img, rect)
        shape = face_utils.shape_to_np(shape)
        mouth= shape[mouth_start:mouth_end]
        l_eye = shape[l_eye_start:l_eye_end]
        r_eye = shape[r_eye_start:r_eye_end]
        mar = smile(mouth)
        mouth_hull = cv2.convexHull(mouth)
        print(mar)
    return mar, mouth_hull #TODO check if eyes open




