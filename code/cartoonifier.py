import cv2
import numpy as np
import os

import canny

def edge_mask(img, line_size, blur_value):
  '''
  This functions creates an edge mask for given image using canny and dilation

  Input:
      img       numpy array     given image to create its edge mask

  Output:
      edges     numpy array     the edge mask of the image, with dilation
  '''


  ##################### start part of try using canny #####################
  '''
  # sigma, L_th, H_th = 1, 0.05, 0.27
  
  # edges = canny.cannyEdges(gray, sigma, L_th, H_th)
  
  # edges = edges.astype(np.uint8)
  # edges = np.logical_not(edges)
  # edges = edges*255
  
  # edges = edges.astype(np.uint8)
  '''
  ##################### end part of try usin canny #####################

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

  return edges

def color_quantization(img, k=9):

  '''
  This functions reduce the number of colors to a given image

  Input:
      img       numpy array     image to reduce its colors
      k         int             number of new different colors


  Output:
      result     numpy array     image recolored to k different colors
  '''

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
  '''
  This functions make a cartoon out of a given image and saves the cartoon in a given path

  Input:
      img               numpy array       image to create is cartoon
      total             int               counter of number of selfies - used as id
      output_folder     string            path to save the selfies in

  '''

  line_size = 7
  blur_value = 7

  edges = edge_mask(img, line_size, blur_value)
  img = color_quantization(img)

  ##################### start part of try using canny #####################
  '''
  # edges = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
  # cartoon = np.bitwise_and(img, edges)
  '''
  ##################### end part of try using canny #####################

  blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
  cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

  img_name = os.path.join(output_folder, "selfie_cartoon_{}.png".format(total))

  cv2.imwrite(img_name, cartoon)

  print("{} written!".format(img_name))
