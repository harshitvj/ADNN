import cv2
import numpy as np


def roi(img, vertices=None):
    """ This method returns cropped image according to passed vertices """

    if vertices is None:
        vertices = [np.array([[0, 0], [0, 255], [270, 255], [400, 75], [540, 75], [655, 255], [960, 255], [960, 0]])]

    masked = np.zeros_like(img)
    cv2.fillPoly(masked, vertices, 255)
    masked = cv2.bitwise_and(img, masked)
    return masked
