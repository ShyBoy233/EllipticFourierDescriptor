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

A0, C0, coeffs = efDescriptor.forward(contour=contour, N=100)
print(f"A0: {A0}")
print(f"C0: {C0}")
print("Coefficients:")
print(coeffs)

# backward elliptic Fourier and plot reconstructed contour
reconstructed_contour = efDescriptor.backward(M=4096) # M determines how many points of the reconstructed contour
cv2.drawContours(img, np.around(reconstructed_contour).astype(np.int32).reshape((-1,1,2)), -1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite("example1.jpg", img)