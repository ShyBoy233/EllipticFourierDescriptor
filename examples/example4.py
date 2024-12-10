import numpy as np
import cv2
import matplotlib.pyplot as plt

from EllipticFourier import EllipticFourier

# load an image
img_src = "SpongeBobs.jpg"
img = cv2.imread(img_src)

# extract image contour
contours = []
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conver to gray image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # threshold to get binary image. Note that in the example image, the forground is back, thus we use cv2.THRESH_BINARY_INV.
contour, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find all possible contours
for c in contour:
    contours.append(np.squeeze(c))

# get coefficients of all contours
efDescriptor = EllipticFourier()

coefficients = []
for c in contours:
    efDescriptor.forward(c, N=100)
    coeffs = efDescriptor.normalize(rotation=True, scale=True)
    coefficients.append(coeffs.flatten())
coefficients = np.array(coefficients) # [10, 400]

# perform PCA analysis of shapes
coefficients_std = (coefficients - coefficients.mean(axis=0)) / coefficients.std(axis=0)

cov_matrix = np.cov(coefficients, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

order_of_importance = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[order_of_importance]
eigenvectors_sorted = eigenvectors[:, order_of_importance]

projected_coeffs = np.matmul(coefficients_std, eigenvectors_sorted[:,:2])
explained_variance = np.cumsum(eigenvalues_sorted)/np.sum(eigenvalues_sorted)
explained_variance = np.concatenate([[0], explained_variance])

# plot results
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['figure.autolayout'] = True

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(projected_coeffs[:,0], projected_coeffs[:,1])
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[0].set_xlim([-6, 6])
ax[0].set_ylim([-6, 6])

ax[1].plot(np.arange(10), explained_variance[:10], marker="o")
ax[1].set_xlabel("Number of principla components")
ax[1].set_ylabel("Total explained vairance")

plt.show()