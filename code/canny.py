import cv2
import numpy as np
from scipy.signal import convolve2d


def Deriv_Gauss(sigma, mask_size, param_type):
    '''
    This func calculetes gauss Derivatives

    Input:
        sigma               float       sigma size
        mask_size           int         mask size
        param_type          str         'x' or 'y'
    Returns:
        gauss_Derivative    np.array    gauss_Derivative for x or y axis
    '''
    # Create a vector with values
    ax = np.linspace(-(mask_size - 1) / 2., (mask_size - 1) / 2., mask_size)

    # Assigne two matrices with the appropriate x and y values on which the gaussian function is computed"
    xx, yy = np.meshgrid(ax, ax)

    axis = yy
    if param_type == "x":
        axis = xx

    gaus_exp = -((xx ** 2) + (yy ** 2)) / (2 * sigma ** 2)
    return -((axis * np.exp(gaus_exp)) / (2 * np.pi * (sigma ** 4)))


def Grad_o(Ix, Iy):
    '''
    This func calculetes grad orientation

    Input:
        Ix                  np.array
        Iy                  np.array

    Returns:
        grad_orientation    np.array    grad orientation in degrees
    '''
    return (np.rad2deg(np.arctan2(Iy, Ix)))


def Grad_m(Ix, Iy):
    '''
    This func calculetes grad magnitute

    Input:
        Ix                  np.array
        Iy                  np.array

    Returns:
        grad_magnitute      np.array    grad magnitute
    '''

    mag = np.hypot(Ix, Iy)
    return mag


def thinning(G_magnitude, G_orientation):
    '''
    This func does thinning stage in canny.
    It checks for each pixel if his G_magnitude is higher then his neighbors
    This implementation uses matrix operations for better performence

    Input:
        G_magnitude         np.array    G_magnitude matrix
        G_orientation       np.array    G_orientation matrix

    Returns:
        suppressed_arr      np.array    img after thinning stage
    '''
    G_orientation[G_orientation < 0] += 180
    G_magnitude = (G_magnitude - np.min(G_magnitude)) / (np.max(G_magnitude) - np.min(G_magnitude))
    suppressed_arr = np.zeros((G_orientation.shape))

    shifted_mag = create_shifted_matrix(G_magnitude)
    G_magnitude_i_jp1, G_magnitude_i_jm1, G_magnitude_ip1_j, G_magnitude_im1_j, \
    G_magnitude_im1_jm1, G_magnitude_ip1_jp1, G_magnitude_im1_jp1, G_magnitude_ip1_jm1 = shifted_mag

    # angle 0
    matching_pixels = (G_orientation >= 0) & \
                      (G_orientation < 22.5) & \
                      (G_magnitude >= G_magnitude_i_jm1) & \
                      (G_magnitude >= G_magnitude_i_jp1)

    suppressed_arr[matching_pixels] = G_magnitude[matching_pixels]

    matching_pixels = (G_orientation >= 157.5) & \
                      (G_orientation <= 180) & \
                      (G_magnitude >= G_magnitude_i_jm1) & \
                      (G_magnitude >= G_magnitude_i_jp1)

    suppressed_arr[matching_pixels] = G_magnitude[matching_pixels]

    # angle 45
    matching_pixels = (G_orientation >= 22.5) & \
                      (G_orientation < 67.5) & \
                      (G_magnitude >= G_magnitude_im1_jm1) & \
                      (G_magnitude >= G_magnitude_ip1_jp1)

    suppressed_arr[matching_pixels] = G_magnitude[matching_pixels]

    # angle 90
    matching_pixels = (G_orientation >= 67.5) & \
                      (G_orientation < 112.5) & \
                      (G_magnitude >= G_magnitude_im1_j) & \
                      (G_magnitude >= G_magnitude_ip1_j)

    suppressed_arr[matching_pixels] = G_magnitude[matching_pixels]

    # angle 135
    matching_pixels = (G_orientation >= 112.5) & \
                      (G_orientation <= 157.5) & \
                      (G_magnitude >= G_magnitude_im1_jp1) & \
                      (G_magnitude >= G_magnitude_ip1_jm1)

    suppressed_arr[matching_pixels] = G_magnitude[matching_pixels]

    # we dont count pixels in i=o, i=G_orientation.shape[0], j=o and j=G_orientation.shape[1] as edges
    iii = G_orientation.shape[0] - 1
    jjj = G_orientation.shape[1] - 1

    suppressed_arr[:, 0] = 0  # first col
    suppressed_arr[:, jjj] = 0  # last col
    suppressed_arr[0] = 0  # first row
    suppressed_arr[iii] = 0  # last row

    return suppressed_arr


def hysteresis(Et, L_th, H_th):
    '''
    This func make sure all pixels will have value above some threshold
    It checks for each pixel if its value is larger than H_th,
    or if the pixel's value is between H_th and L_th and the pixel or pixel it connected to
    (by pixels with values between H_th and L_th) are adjacent to pixel with value larger than H_th

    Input:
        Et         np.array     Matrix after thinning
        L_th       float        low threshold value
        H_th       float        High threshold value

    Returns:
        connectedComponents_img      bool np.array    img with all pixels that met the requirements (set as True)
    '''
    # keep all pixels that have value larger than H_th and give them the value 'True', all other get value 'False'
    img_high = np.where(Et > H_th, 1, 0).astype(bool)

    # keep all pixels that have value larger than L_th and give them the value '1', all other get value '0'
    img_low = np.where(Et > L_th, 1, 0).astype("uint8")

    # low_labels_img - image where all group of connected pixels get the same value (different value per group)
    # num_low_labels - not important for the rest of code

    _, low_labels_img = cv2.connectedComponents(img_low)

    # all labels of groups from low_labels_img that have at least one pixel with value larger than h_th
    good_low_labels = np.ndarray.flatten(low_labels_img)[np.ndarray.flatten(img_high)]

    # keep all pixels from low_labels_img that their group label is in good_low_labels
    connectedComponents_img = np.isin(low_labels_img, good_low_labels)

    return connectedComponents_img


def canny(img, sigma, L_th, H_th):
    '''
    This func runs the classic canny algorithm:
    1. Apply Deriv_Gauss karnel to smooth the image in order to remove the noise
    2. Find the intensity gradients of the image
    3. Apply non-maximum suppression to get rid of spurious response to edge detection
    4. Apply double threshold to determine potential edges and track edge by hysteresis:
       Finalize the detection of edges by suppressing all the other edges that are weak
       and not connected to strong edges.

    Input:
        img         np.array    input image
        sigma       float
        L_th        float       low threshold value
        H_th        float       High threshold value

    Returns:
        canny_image np.array    canny result - edges img
    '''
    mask_size = sigma * 4  # for 95 percentage coverage
    if (mask_size % 2) == 0:
        mask_size = mask_size + 1

    mask_size = int(mask_size)
    if mask_size < 3:
        mask_size = 3
        # the true formula is much complicated but we are not supposed to support
        # sigma smaller then 1 so it is only estimation

    # params = "sigma=" + str(sigma) + ", L_th=" + str(L_th) + ", H_th=" + str(H_th)
    G_dx = Deriv_Gauss(sigma, mask_size, 'x')
    G_dy = Deriv_Gauss(sigma, mask_size, 'y')

    Ix = convolve2d(img, G_dx, 'same')
    Iy = convolve2d(img, G_dy, 'same')

    G_orientation = Grad_o(img, Ix, Iy)
    G_magnitude = Grad_m(img, Ix, Iy)

    Et = thinning(G_magnitude, G_orientation)

    canny_image = hysteresis(Et, L_th, H_th)
    return canny_image


def create_shifted_matrix(np_array, shift_direction=None):
    '''
    This func gets array and creats its shifted arrays versions.

    Input:
        np_array        np_array        the array we want to shift
        shift_direction str             wanted shift direction
    Returns:
        arrays          list of arrays  8 shifted array by the following order:
                                        1. orig_array[i][j+1] - orig_array_i_jp1
                                        2. orig_array[i][j-1] - orig_array_i_jm1
                                        3. orig_array[i+1][j] - orig_array_ip1_j
                                        4. orig_array[i-1][j] - orig_array_im1_j
                                        5. orig_array[i-1][j-1] - orig_array_im1_jm1
                                        6. orig_array[i+1][j+1] - orig_array_ip1_jp1
                                        7. orig_array[i-1][j+1] - orig_array_im1_jp1
                                        8. orig_array[i+1][j-1] - orig_array_ip1_jm1
    '''
    if shift_direction is not None:
        if shift_direction == 'i_jp1':
            return np.c_[np.zeros(np_array.shape[0]), np_array[:, :-1]]
        if shift_direction == 'i_jm1':
            return np.c_[np_array[:, 1:], np.zeros(np_array.shape[0])]
        if shift_direction == 'ip1_j':
            return np.r_[[np.zeros(np_array.shape[1])], np_array[:-1, :]]
        if shift_direction == 'im1_j':
            return np.r_[np_array[1:, :], [np.zeros(np_array.shape[1])]]
        if shift_direction == 'ip1_jm1':
            orig_array_ip1_j = np.r_[[np.zeros(np_array.shape[1])], np_array[:-1, :]]
            return np.c_[orig_array_ip1_j[:, 1:], np.zeros(np_array.shape[0])]
        if shift_direction == 'im1_jm1':
            orig_array_im1_j = np.r_[np_array[1:, :], [np.zeros(np_array.shape[1])]]
            return np.c_[orig_array_im1_j[:, 1:], np.zeros(np_array.shape[0])]
        if shift_direction == 'ip1_jp1':
            orig_array_i_jp1 = np.c_[np.zeros(np_array.shape[0]), np_array[:, :-1]]
            return np.r_[[np.zeros(np_array.shape[1])], orig_array_i_jp1[:-1, :]]
        if shift_direction == 'im1_jp1':
            orig_array_i_jp1 = np.c_[np.zeros(np_array.shape[0]), np_array[:, :-1]]
            return np.r_[orig_array_i_jp1[1:, :], [np.zeros(np_array.shape[1])]]

    orig_array_i_jp1 = np.c_[np.zeros(np_array.shape[0]), np_array[:, :-1]]
    orig_array_i_jm1 = np.c_[np_array[:, 1:], np.zeros(np_array.shape[0])]
    orig_array_ip1_j = np.r_[[np.zeros(np_array.shape[1])], np_array[:-1, :]]
    orig_array_im1_j = np.r_[np_array[1:, :], [np.zeros(np_array.shape[1])]]

    orig_array_ip1_jm1 = np.c_[orig_array_ip1_j[:, 1:], np.zeros(np_array.shape[0])]
    orig_array_im1_jm1 = np.c_[orig_array_im1_j[:, 1:], np.zeros(np_array.shape[0])]
    orig_array_ip1_jp1 = np.r_[[np.zeros(np_array.shape[1])], orig_array_i_jp1[:-1, :]]
    orig_array_im1_jp1 = np.r_[orig_array_i_jp1[1:, :], [np.zeros(np_array.shape[1])]]

    return [orig_array_i_jp1,
            orig_array_i_jm1,
            orig_array_ip1_j,
            orig_array_im1_j,
            orig_array_im1_jm1,
            orig_array_ip1_jp1,
            orig_array_im1_jp1,
            orig_array_ip1_jm1]



def cannyEdges(img, sigma, L_th, H_th):

    if img is None:
        raise Exception("there is a problem with the img. check its path:")

    cannyResults = canny(img, sigma, L_th, H_th)

    return cannyResults
