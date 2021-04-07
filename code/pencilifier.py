import cv2
import os

def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

def pencilMe(img, total, output_folder):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)

    img_name = os.path.join(output_folder, "selfie_pancil_{}.png".format(total))

    cv2.imwrite(img_name, final_img)
    print("{} written!".format(img_name))


