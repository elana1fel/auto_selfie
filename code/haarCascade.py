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

def takeSelfies(eye, smile,frame, grayscaleFrame, output_folder, cartoon, pencil, useCanny=False):

    global total, counter, last_taken_selfie

    if ((eye is not None) and (smile is not None ) and (len(eye)>0) and (len(smile)>0)):
        counter += 1

        # print(f"counter is: {counter}")
        if counter >= 5:  # we need to check it
            if last_taken_selfie is not None:
                (score, diff) = compare_ssim(last_taken_selfie, grayscaleFrame, full=True)
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
                    last_taken_selfie = grayscaleFrame
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
                last_taken_selfie = grayscaleFrame
                counter = 0

                pass

def detection(cascade_face, cascade_eye, cascade_smile, grayscale, img):
    eye = None
    smile = None
    face = cascade_face.detectMultiScale(grayscale, 1.3, 4)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face + h_face, x_face:x_face + w_face]
        ri_color = img[y_face:y_face + h_face, x_face:x_face + w_face]

        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.1, 14)
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)

        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.2, 15)
        for (x_smile, y_smile, w_smile, h_smile) in smile:
            cv2.rectangle(ri_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)

    return img, eye, smile


def runHaarCascade(cascade_face, cascade_eye, cascade_smile, cartoon=False, pencil=False, useCanny=False):

    print("for regular selfies press 'n'\n"
          "for cartoon selfies press 'c'\n"
          "for pencil selfies press 'p'\n")

    output_folder =  'selfies_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(output_folder)

    print("starting looking for perfect selfie mode")

    vc = cv2.VideoCapture(0)

    while True:
        _, img = vc.read()
        origImg = img.copy()
        grayscaleFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final, eye, smile = detection(cascade_face, cascade_eye, cascade_smile, grayscaleFrame, img)
        cv2.imshow('Video', final)

        takeSelfies(eye, smile, origImg, grayscaleFrame, output_folder, cartoon, pencil, useCanny)

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


    vc.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    useCanny = False

    runHaarCascade(cascade_face, cascade_eye, cascade_smile, useCanny=useCanny)
