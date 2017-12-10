""" a program that applies a very basic sharpening filter on a grayscale image via a kernel convolution """

# library imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# input image file path
INPUT_IMG_PATH = "original.jpg"

# out image plot titles
INPUT_IMG_NAME = "original image"
OUTPUT_IMG_NAME = "filtered image"

# gray scale images have a color depth of 0
GRAY_SCALE_DEPTH = 0

# load the input image
img = cv2.imread(INPUT_IMG_PATH, cv2.IMREAD_GRAYSCALE)

# define the filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 10, -1],
                   [-1, -1, -1]])


# normalize the kernel to avoid white fading
kernel_sum = np.sum(kernel)
kernel = kernel/kernel_sum


# convolve the image against the kernel
# NOTE: opencv takes care of constraining the value to max 255, min 0 (does not wrap around)
# NOTE: borders are ignored and not handled at all (BORDER_ISOLATED) - should be decided on a per image basis
output_img = cv2.filter2D(src=img, kernel=kernel, ddepth=GRAY_SCALE_DEPTH, borderType=cv2.BORDER_ISOLATED)


# plot and display the image side by side
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title(INPUT_IMG_NAME)
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(output_img, cmap="gray")
plt.title(OUTPUT_IMG_NAME)
plt.xticks([]), plt.yticks([])

plt.show()
