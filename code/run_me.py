from imutils.video import VideoStream, FPS
import numpy as np
import time
import cv2
import selfie_utils


def run(video_path=None):
    if video_path is None: #we are running on camera mode
        running_mode = 'camera'
    else:
        running_mode = 'video'

    counter = 0
    last_taken_selfie = None
    show_stats = True
    draw_contours = True
    detector, predictor = selfie_utils.init_model()
    print("starting looking for perfect selfie mode")
    if video_path is None: #we are running on camera mode
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
        fps= FPS().start()

    elif running_mode == 'video':
        cap = cv2.VideoCapture(video_path)

    cv2.namedWindow(running_mode)
    while True:
        if running_mode == 'camera':
            frame = vs.read()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        gray_img, frame = selfie_utils.edit_img(frame)
        faces = selfie_utils.detect_face(gray_img, detector, predictor) 

        for face in faces:
            if draw_contours:
                for face_part in ['mouth_hull', 'l_eye_hull', 'r_eye_hull']:
                    cv2.drawContours(frame, [face[face_part]], -1, (0, 255, 0), 1)
        
            if show_stats:
                for ar in ['mar', 'l_ear', 'r_ear']:
                    cv2.putText(frame, f"{ar}: {face[ar]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if face['mar'] <= .3 or face['mar'] > .38 :
                counter += 1

                if counter >= 5: #we need to check it
                    if last_taken_selfie is not None:
                        # check if there is a difference between current img to last taken selfie.
                        # if it is the same go back to while loop
                        pass

                    else:
                        # first selfie
                        # TODO:take a selfie
                        counter = 0
                        pass

        cv2.imshow("Frame", frame)
        if running_mode == 'camera':
            fps.update()

        key2 = cv2.waitKey(1) & 0xFF
        #if key2 == ord('q'):
        #    break

    if running_mode == 'camera':
        fps.stop()
        vs.stop()


    cv2.destroyAllWindows()



if __name__ == "__main__":
    #video_path = r'test_video.mp4'
    run()
    run(video_path)