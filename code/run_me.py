from imutils.video import VideoStream, FPS
#from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_ssim
import similarity.compare_sim as compare_sim

import numpy as np
import time
import cv2
import os

import selfie_utils
import cartoonifier
import pencilifier

def train(training_folder):
    for file in os.listdir:
        file_path = os.path.join(training_folder, file)

def run(input_path=None, cartoon = False, pencil=False):

    print("for regular selfies press 'n'\n"
          "for cartoon selfies press 'c'\n"
          "for pencil selfies press 'p'\n")


    if input_path is None: #we are running on camera mode
        running_mode = 'camera'
    else:
        running_mode = 'video'

    output_folder =  'selfies_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(output_folder)

    total = 0
    counter = 0
    last_taken_selfie = None
    show_stats = True
    draw_contours = True
    detector, predictor = selfie_utils.init_model()
    print("starting looking for perfect selfie mode")
    if input_path is None: #we are running on camera mode
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        fps= FPS().start()

    elif running_mode == 'video':
        cap = cv2.VideoCapture(input_path)

    cv2.namedWindow(running_mode)
    while True:
        if running_mode == 'camera':
            frame = vs.read()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        gray_img, resized_frame = selfie_utils.edit_img(frame)
        faces = selfie_utils.detect_face(gray_img, detector, predictor) 

        i = 0 #for putText
        for face in faces:
            if draw_contours:
                for face_part in ['mouth_hull', 'l_eye_hull', 'r_eye_hull']:
                    cv2.drawContours(resized_frame, [face[face_part]], -1, (0, 255, 0), 1)
        
            if show_stats:
                y0, dy = 30, 30
                for ar in ['mar', 'l_ear', 'r_ear']:
                    y = y0 + i * dy
                    cv2.putText(resized_frame, f"{ar}: {face[ar]:.5f}", (10, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    i += 1
                # for ar in ['mar', 'l_ear', 'r_ear']:
                #     cv2.putText(frame, f"{ar}: {face[ar]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(resized_frame, f"{'mar'}: {face['mar']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if (face['mar'] <= .26 or face['mar'] > .32) and (face['l_ear'] > .25) and (face['r_ear'] > .25):
                counter += 1

                # print(f"counter is: {counter}")
                if counter >= 5: #we need to check it
                    if last_taken_selfie is not None:
                        (score, diff) = compare_sim(last_taken_selfie, gray_img)
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
                            last_taken_selfie = gray_img
                            counter = 0
                        pass

                    else:
                        # first selfie
                        # TODO:take a selfiec

                        if cartoon:
                            cartoonifier.cartoonify(frame, total, output_folder)
                        elif pencil:
                            pencilifier.pencilMe(frame, total, output_folder)
                        else:
                            selfie_utils.take_selfie(frame, total, output_folder)

                        total += 1
                        last_taken_selfie = gray_img
                        counter = 0

                        pass

        cv2.imshow("Frame", resized_frame)
        if running_mode == 'camera':
            fps.update()

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

    if running_mode == 'camera':
        fps.stop()
        vs.stop()


    cv2.destroyAllWindows()



if __name__ == "__main__":
    #input_path = r'test_video.mp4'
    # run(input_path)

    run()
