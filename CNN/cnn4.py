import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(r"F:\shapes.png").convert("L")
img = np.array(img)

edges = cv2.Canny(img, 100, 200)

corners = cv2.goodFeaturesToTrack(edges, maxCorners=100, qualityLevel=0.01, minDistance=10)

if corners is not None:
    num_corners = len(corners)

    if num_corners <= 2:
        shape_name = "Circle"
    elif num_corners == 3:
        shape_name = "Triangle"
    elif num_corners == 4:
        shape_name = "Square"
    elif num_corners == 5:
        shape_name = "Pentagon"
    elif num_corners == 6:
        shape_name = "Hexagon"
    else:
        shape_name = "Other"
else:
    shape_name = "No shape detected"

plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()
print(shape_name)