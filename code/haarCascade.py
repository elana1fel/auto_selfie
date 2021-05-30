import cv2
import similarity.compare_sim as compare_sim
import os
import time

import selfie_utils
import cartoonifier
import pencilifier

last_taken_selfie = None
total = 0
counter = 0

def takeSelfies(eyes, smile, frame, gray, output_folder, cartoon, pencil):
    '''
    This functions takes the selfie if conditions of found smile and two eyes is met. the function call the desired filter function if needed

    Input:
        eyes                numpy array             array of arrays of coordinates of found eyes on image.
        smile               numpy array             array of coordinates of found smile on image.
        frame               numpy array             the current observed frame
        gray                numpy array             the current observed frame as a gray scale image
        output_folder       string                  path to save the selfies in
        cartoon             boolean                 if true - use this filter when taking selfie
        pencil              boolean                 if true - use this filter when taking selfie

    '''
    global total, counter, last_taken_selfie

    if ((eyes is not None) and (smile is not None ) and (len(eyes)%2 ==0) and (len(smile)>=1)):
        counter += 1

        # print(f"counter is: {counter}")
        if counter >= 5:  # we need to check it
            if last_taken_selfie is not None:
                (score, diff) = compare_sim(last_taken_selfie, gray)
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

def detection(face_cascade, eye_cascade, smile_cascade, gray, img):
    '''
    This functions detect the face, eyes and smile in a given gray image and draw the detection on the colored image

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


def runHaarCascade(face_cascade, eye_cascade, smile_cascade, cartoon=False, pencil=False):
    '''
    This functions is the main function of the selfies using Haar Cascade.
    it reads the frames from the camera and take the selfie with or without filter.
    the function also lets the user can change the filter selection by clicking on the relevant letters.

    Input:
        face_cascade        Cascade Classifier      classifier that is used to detect faces in a given image
        eye_cascade         Cascade Classifier      classifier that is used to detect eyes in a given image
        smile_cascade       Cascade Classifier      classifier that is used to detect smile in a given image
        cartoon             boolean                 if true - use this filter when taking selfie
        pencil              boolean                 if true - use this filter when taking selfie
    '''


    print("for regular selfies press 'n'\n"
          "for cartoon selfies press 'c'\n"
          "for pencil selfies press 'p'\n")

    output_folder =  'selfies_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(output_folder)

    print("starting looking for perfect selfie mode")

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        origImg = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # grayscaleFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final, eye, smile = detection(face_cascade, eye_cascade, smile_cascade, gray, img)
        cv2.imshow('Video', final)

        takeSelfies(eye, smile, origImg, gray, output_folder, cartoon, pencil)

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

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    runHaarCascade(face_cascade, eye_cascade, smile_cascade)
