import cv2
import numpy as np
import os

import canny

def edge_mask(img):
  '''
  This functions creates an edge mask for given image using canny and dilation

  Input:
      img       numpy array     given image to create its edge mask

  Output:
      edges     numpy array     the edge mask of the image, with dilation
  '''
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  sigma, L_th, H_th = 0.5, 0.1, 0.3
  edges = canny.cannyEdges(gray, sigma, L_th, H_th)

  edges = edges.astype(np.uint8)
  edges = edges*255

  kernel = np.ones((33, 33), np.uint8)
  edges = cv2.dilate(edges, kernel, iterations=15)
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



  edges = edge_mask(img)
  img = color_quantization(img)

  blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
  cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

  img_name = os.path.join(output_folder, "selfie_cartoon_{}.png".format(total))

  cv2.imwrite(img_name, cartoon)

  print("{} written!".format(img_name))
