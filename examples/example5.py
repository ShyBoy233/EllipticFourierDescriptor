import numpy as np
import cv2
import matplotlib.pyplot as plt

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

error = efDescriptor.error(contour=contour)
error1 = efDescriptor.error(contour=contour1)
error2 = efDescriptor.error(contour=contour2)

# plot results
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['figure.autolayout'] = True

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(error)
ax[1].plot(error1)
ax[2].plot(error2)

plt.show()