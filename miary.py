import cv2
import numpy as np
from skimage.measure import compare_ssim
import argparse
import imutils

def ssim(A,B):
    # load the two input images
    imageA = cv2.imread(A)
    imageB = cv2.imread(B)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    #   images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))
