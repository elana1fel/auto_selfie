from imutils.video import VideoStream, FPS
import numpy as np
import time
import cv2
import selfie_utils

RUNNING_MODE = CAMERA

def run():
    counter = 0
    last_taken_selfie = None
    show_stats = False
    draw_contours = False

    print("starting looking for perfect selfie mode")
    if RUNNING_MODE == 'CAMERA':
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        fps= FPS().start()
    cv2.namedWindow(RUNNING_MODE)

    while True:
        frame = vs.read()
        #gray_img = selfie_utils.edit_img(frame)
        mar = 5 #TODO TEMP        


        if draw_contours:
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
    
        if show_stats:
            cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if mar <= .3 or mar > .38 :
            COUNTER += 1

            if COUNTER >= 5: #we need to check it
                if last_taken_selfie is not None:
                    # check if there is a difference between current img to last taken selfie.
                    # if it is the same go back to while loop
                    pass

                else:
                    # first selfie
                    # TODO:take a selfie
                    pass

        cv2.imshow("Frame", frame)
        fps.update()

        key2 = cv2.waitKey(1) & 0xFF
        if key2 == ord('q'):
            break

    if RUNNING_MODE = 'CAMERA':
        fps.stop()


    cv2.destroyAllWindows()

    if RUNNING_MODE = 'CAMERA':
        vs.stop()


if __name__ == "__main__":
    run()