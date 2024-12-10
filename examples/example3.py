import numpy as np
import cv2

from EllipticFourier import EllipticFourier

def extract_contour(img_src):
    # load an image
    img = cv2.imread(img_src)

    # extract image contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to gray image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # threshold to get binary image. Note that in the example image, the forground is back, thus we use cv2.THRESH_BINARY_INV.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find all possible contours
    contour = np.squeeze(contours[0])
    return contour

contour = extract_contour("SpongeBob.jpg")
contour1 = extract_contour("SpongeBob1.jpg")
contour2 = extract_contour("SpongeBob2.jpg")

efDescriptor = EllipticFourier()

efDescriptor.forward(contour=contour, N=100)
coeffs = efDescriptor.normalize(rotation=True, scale=True)

efDescriptor.forward(contour=contour1, N=100)
coeffs1 = efDescriptor.normalize(rotation=True, scale=True)

efDescriptor.forward(contour=contour2, N=100)
coeffs2 = efDescriptor.normalize(rotation=True, scale=True)

np.savetxt("coeffs.txt", coeffs)
np.savetxt("coeffs1.txt", coeffs1)
np.savetxt("coeffs2.txt", coeffs2)