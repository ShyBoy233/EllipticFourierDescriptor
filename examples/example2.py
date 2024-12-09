import os
import numpy as np
import cv2

from EllipticFourier import EllipticFourier

# load an image
img_src = "SpongeBob.jpg"
img = cv2.imread(img_src)

# extract image contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to gray image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # threshold to get binary image. Note that in the example image, the forground is back, thus we use cv2.THRESH_BINARY_INV.
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find all possible contours
contour = np.squeeze(contours[0]) 

# forward elliptic Fourier
efDescriptor = EllipticFourier()

efDescriptor.forward(contour=contour, N=100) # only forward elliptic Fourier

# reconstruct contour using different numbers of coefficients and save results
dst = "example2"
if not os.path.exists(dst):
    os.mkdir(dst)
for num in range(1, 100+1):
    reconstructed_contour = efDescriptor.backward(M=4096, modeStart=1, modeNum=num)
    raw_img = img = cv2.imread(img_src)
    cv2.drawContours(raw_img, np.around(reconstructed_contour).astype(np.int32).reshape((-1,1,2)), -1, (0, 0, 255), 5, cv2.LINE_AA)
    image_with_text = cv2.putText(raw_img, f"Mode numbers: {num}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(dst, f"img_{num:0>4}.jpg"), raw_img)