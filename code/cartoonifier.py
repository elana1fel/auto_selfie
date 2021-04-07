import cv2
import numpy as np
import os

def edge_mask(img, line_size=7, blur_value=7):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

def color_quantization(img, k=9):
# Transform the image
  data = np.float32(img).reshape((-1, 3))

# Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result


def cartoonify(img, total, output_folder):
  edges = edge_mask(img)
  img = color_quantization(img)
  blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
  cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

  img_name = os.path.join(output_folder, "selfie_cartton_{}.png".format(total))

  cv2.imwrite(img_name, cartoon)
  print("{} written!".format(img_name))
