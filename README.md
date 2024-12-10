# EllipticFourierDescriptor
Python implementation and interpretation of "Elliptic Fourier Features of a Closed Contour" with enhanced start point and rotation normalization methods.

# 1 Introduction
Python implementation and interpretation of "Elliptic Fourier Features of a Closed Contour" with enhanced start point and rotation normalization methods.
In the code, the point having the maimum distance from the centroid is selected as the start point and the rotation is normalized by aligning the start point with horizontal axis of the coordiante system

# 2 Dependency and installation
The EllipticFourier module only depends on numpy, thus any python environment having numpy installed is feasible to run the code. Download the file "EllipticFourier.py" and copy the file to your working direction. OpenCV is required to run all examples.

# 3 Usage
Given a closed contour of a shape, generated by e.g. OpenCV, this package can fit a Fourier series approximating the shape of the contour.

## 3.1 Basic usage
Using elliptic Fourier descriptor to describe the shape of SpongeBob. Here we used 10 elliptic Fourier coefficients to describe the shape.

![Image 1](imgs/img1.png)

``` python
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

A0, C0, coeffs = efDescriptor.forward(contour=contour, N=10)
print(f"A0: {A0}")
print(f"C0: {C0}")
print("Coefficients:")
print(coeffs)

# backward elliptic Fourier and plot reconstructed contour
reconstructed_contour = efDescriptor.backward(M=4096) # M determines how many points of the reconstructed contour
cv2.drawContours(img, np.around(reconstructed_contour).astype(np.int32).reshape((-1,1,2)), -1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite("example1.jpg", img)
```
[Link to codes file.](examples/example1.py)

Running the following scripts, you will get $A_0$, $C_0$ and all 10 group of coefficients.
```python
A0: 500.32974835623213
C0: 512.9862975766608
Coefficients:
[[-124.16782725 -331.41636558 -329.99128026  232.88912794]
 [ -78.04208755  -60.82582495  -66.03245759  -26.06059374]
 [   0.9755801   -24.00522499 -104.851157    -39.06491599]
 [  -3.58647986  -51.69773256    1.07226335   54.5695385 ]
 [ -30.71359422   -5.70036968   -1.9184519     3.19809928]
 [  -2.10359673   -9.16108099   55.73271607  -33.88256265]
 [   4.60579685    5.11983397  -19.0496939     9.76848141]
 [   1.94547198   -1.32610621  -22.03017032  -14.17918943]
 [   8.34267328    6.48530055    2.94044438   -2.37734919]
 [   9.6104728     6.48107343    7.25323288   -6.96029392]]
```

## 3.2 Reconstruct contour with different numbers of coefficients
Reconstruct contour with different numbers of coefficients. The more numbers of coefficients, the more accurate the reconstructed contour. All the reconstructed contour overlay on images were saved in directory named example2.

![Image 2](imgs/img2.gif)

```python
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
    cv2.drawContours(raw_img, np.around(reconstructed_contour).astype(np.int32).reshape((-1,1,2)), -1, (0, 0, 255), 2, cv2.LINE_AA)
    image_with_text = cv2.putText(raw_img, f"Mode numbers: {num}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(dst, f"img_{num:0>4}.jpg"), raw_img)
```

[Link to codes file.](examples/example2.py)

# 4 References
1. Frank P Kuhl, Charles R Giardina, Elliptic Fourier features of a closed contour, Computer Graphics and Image Processing, Volume 18, Issue 3, 1982, Pages 236-258. https://doi.org/10.1016/0146-664X(82)90034-X
2. Burger, W., Burge, M.J. (2013). Fourier Shape Descriptors. In: Principles of Digital Image Processing. Undergraduate Topics in Computer Science. Springer, London. https://doi.org/10.1007/978-1-84882-919-0_6
3. pyefd: Python implementation of "Elliptic Fourier Features of a Closed Contour". https://github.com/hbldh/pyefd

# 5 Implementation details
Implementation detials were depicted in the file ["EllipticFourierDescriptor.pdf"](EllipticFourierDescriptor.pdf).