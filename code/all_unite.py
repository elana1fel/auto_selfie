import cv2
from skimage.metrics import structural_similarity as compare_ssim
import os
import time

import selfie_utils
import cartoonifier
import pencilifier


last_taken_selfie = None
total = 0
counter = 0

def takeSelfies(frame, gray, output_folder, cartoon, pencil, useCascade, useDlib, useBoth, eyes_cascade = None, smile_cascade= None, faces = None):
    '''
    This functions takes the selfie if conditions of found smile and two eyes is met. the function call the desired filter function if needed

    Input:

        frame               numpy array             the current observed frame
        gray                numpy array             the current observed frame as a gray scale image
        output_folder       string                  path to save the selfies in
        cartoon             boolean                 if true - use this filter when taking selfie
        pencil              boolean                 if true - use this filter when taking selfie
        useCascade          boolean                 if true - look for features only using cascades
        useDlib             boolean                 if true - look for features only using Dlib
        useBoth             boolean                 if true - look for features using cascades and Dlib
        eyes_cascade        numpy array             array of arrays of coordinates of found eyes on image by using haar cascade.
        smile_cascade       numpy array             array of coordinates of found smile on image by using haar cascade.
        faces               dict                    dict contains features found using Dlib

    '''
    global total, counter, last_taken_selfie

    featuresDetected = False
    detectCascade = ((eyes_cascade is not None) and (smile_cascade is not None ) and (len(eyes_cascade)%2 ==0) and (len(smile_cascade)>=1))
    detectDlib = False
    if faces:
        for face in faces:
            currentFaceDetectDlib = (face['mar'] <= .26 or face['mar'] > .32) and (face['l_ear'] > .25) and (face['r_ear'] > .25)
            detectDlib = currentFaceDetectDlib or detectDlib

    if useBoth and detectCascade and detectDlib:
        featuresDetected = True

    elif useCascade and detectCascade:
        featuresDetected = True

    elif useDlib and detectDlib:
        featuresDetected = True

    if featuresDetected:
        counter += 1

        # print(f"counter is: {counter}")
        if counter >= 5:  # we need to check it
            if last_taken_selfie is not None:
                (score, diff) = compare_ssim(last_taken_selfie, gray, full=True)
                # print("SSIM: {}".format(score))
                # check if there is a difference between current img to last taken selfie.
                # if it is the same go back to while loop

                if score < 0.8:
                    total += 1
                    if cartoon:
                        cartoonifier.cartoonify(frame, total, output_folder)
                    elif pencil:
                        pencilifier.pencilMe(frame, total, output_folder)
                    else:
                        selfie_utils.take_selfie(frame, total, output_folder)
                    last_taken_selfie = gray
                    counter = 0
                pass

            else:
                # first selfie

                if cartoon:
                    cartoonifier.cartoonify(frame, total, output_folder)
                elif pencil:
                    pencilifier.pencilMe(frame, total, output_folder)
                else:
                    selfie_utils.take_selfie(frame, total, output_folder)

                total += 1
                last_taken_selfie = gray
                counter = 0

                pass

def haarDetection(face_cascade, eye_cascade, smile_cascade, gray, img):
    '''
    This functions detect the face, eyes and smile in a given gray image and draw the detection on the colored image - all using haar cascades

    Input:
        face_cascade        Cascade Classifier      classifier that is used to detect faces in a given image
        eye_cascade         Cascade Classifier      classifier that is used to detect eyes in a given image
        smile_cascade       Cascade Classifier      classifier that is used to detect smile in a given image
        gray                numpy array             the current observed frame as a gray scale image
        img                 numpy array             the current observed frame


    Output:
        img                 numpy array             the current observed frame with the rectangles of the detect face, eyes and smile
        eyes                numpy array             array of arrays of coordinates of found eyes on image.
        smile               numpy array             array of coordinates of found smile on image.

    '''


    eyes = None
    smile = None
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in faces:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 0), 2)
        roi_gray = gray[y_face:y_face + h_face, x_face:x_face + w_face]
        roi_color = img[y_face:y_face + h_face, x_face:x_face + w_face]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.03, minNeighbors=40, minSize=(30,30))
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 255, 0), 2)


        # add from eyes pattern
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=40, minSize=(30,30))
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            cv2.rectangle(roi_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (0, 255, 0), 2)

    return img, eyes, smile


def dlibDetection(frame, detector, predictor, show_stats, draw_contours):
    '''
    This functions detect the face, eyes and smile in a given image and draw the detection of the features - using Dlib

    Input:
        frame               numpy array      classifier that is used to detect faces in a given image
        detector
        predictor
        show_stats          boolean             indicates if show stats of features on image
        draw_contours       boolean             indicates if show contours of features on image


    Output:
        resized_frame       numpy array             the frame resized
        faces               dict                    dict contains features found using Dlib

    '''
    gray_img, resized_frame = selfie_utils.edit_img(frame)
    faces = selfie_utils.detect_face(gray_img, detector, predictor)

    i = 0  # for putText
    for face in faces:
        if draw_contours:
            for face_part in ['mouth_hull', 'l_eye_hull', 'r_eye_hull']:
                cv2.drawContours(resized_frame, [face[face_part]], -1, (0, 255, 0), 1)

        if show_stats:
            y0, dy = 30, 30
            for ar in ['mar', 'l_ear', 'r_ear']:
                y = y0 + i * dy
                cv2.putText(resized_frame, f"{ar}: {face[ar]:.5f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                i += 1
            cv2.putText(resized_frame, f"{'mar'}: {face['mar']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
    return resized_frame, faces


def run(useCascade, useDlib, useBoth, face_cascade=None, eye_cascade=None, smile_cascade=None, cartoon=False, pencil=False):
    '''
    This functions is the main function of the selfies.
    it reads the frames from the camera and take the selfie with or without filter.
    the function also lets the user can change the filter selection by clicking on the relevant letters.

    Input:
        useCascade          boolean                 if true - look for features only using cascades
        useDlib             boolean                 if true - look for features only using Dlib
        useBoth             boolean                 if true - look for features using cascades and Dlib
        face_cascade        numpy array             array of coordinates of found face on image by using haar cascade.
        eyes_cascade        numpy array             array of arrays of coordinates of found eyes on image by using haar cascade.
        smile_cascade       numpy array             array of coordinates of found smile on image by using haar cascade.
        cartoon             boolean                 if true - use this filter when taking selfie
        pencil              boolean                 if true - use this filter when taking selfie

    '''



    print("for regular selfies press 'n'\n"
          "for cartoon selfies press 'c'\n"
          "for pencil selfies press 'p'\n")

    output_folder = 'selfies_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(output_folder)


    print("starting looking for perfect selfie mode")

    cap = cv2.VideoCapture(0)

    final, eye, smile = None, None, None
    faces = None

    detector, predictor = selfie_utils.init_model()


    while True:
        ret, img = cap.read()
        frame = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if useBoth or useDlib:
            show_stats = True
            draw_contours = True
            if useBoth:
                show_stats = False
                draw_contours = False

            final, faces = dlibDetection(frame, detector, predictor, show_stats, draw_contours)

        if useBoth or useCascade:
            final, eye, smile = haarDetection(face_cascade, eye_cascade, smile_cascade, gray, img)



        takeSelfies(frame, gray, output_folder, cartoon, pencil, useCascade, useDlib, useBoth, eye, smile, faces)

        cv2.imshow('Video', final)

        key2 = cv2.waitKey(1) & 0xFF
        if key2 == ord('q'):
            break
        if key2 == ord('p'):
            pencil = True
            cartoon = False
            print("taking *PENCIL* selfies..")
        if key2 == ord('c'):
            pencil = False
            cartoon = True
            print("taking *CARTOON* selfies..")

        if key2 == ord('n'):
            print("taking *REGULAR* selfies ")
            pencil = False
            cartoon = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    useCascade = True
    useDlib = True
    useBoth = False

    if useDlib and useCascade:
        useBoth = True
        useCascade = False
        useDlib = False

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    run(useCascade, useDlib, useBoth, face_cascade, eye_cascade, smile_cascade, cartoon=False, pencil=False)