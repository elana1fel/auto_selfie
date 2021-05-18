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

def takeSelfies(eyes, smile,frame, gray, output_folder, cartoon, pencil, useCanny=False):

    global total, counter, last_taken_selfie

    if ((eyes is not None) and (smile is not None ) and (len(eyes)>0) and (len(smile)>0)):
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
                        cartoonifier.cartoonify(frame, total, output_folder, useCanny)
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
                    cartoonifier.cartoonify(frame, total, output_folder, useCanny)
                elif pencil:
                    pencilifier.pencilMe(frame, total, output_folder)
                else:
                    selfie_utils.take_selfie(frame, total, output_folder)

                total += 1
                last_taken_selfie = gray
                counter = 0

                pass

def detection(face_cascade, eye_cascade, smile_cascade, gray, img):
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


def runHaarCascade(face_cascade, eye_cascade, smile_cascade, cartoon=False, pencil=False, useCanny=False):

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

        takeSelfies(eye, smile, origImg, gray, output_folder, cartoon, pencil, useCanny)

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

    useCanny = False

    runHaarCascade(face_cascade, eye_cascade, smile_cascade, useCanny=useCanny)
