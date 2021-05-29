import cv2
import os

def pencilMe(img, total, output_folder):
    '''
    This functions make a pancil sketch out of a given image and saves the sketch in a given path.
    this is done by:
        - Converting an image into gray_scale image
        - Inverting the image
        - Smoothing the image
        - Obtaining the final sketch

    Input:
        img               numpy array       image to create is cartoon
        total             int               counter of number of selfies - used as id
        output_folder     string            path to save the selfies in

    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = cv2.divide(img_gray, 255 - img_smoothing, scale=256)

    img_name = os.path.join(output_folder, "selfie_pancil_{}.png".format(total))

    cv2.imwrite(img_name, final_img)
    print("{} written!".format(img_name))


