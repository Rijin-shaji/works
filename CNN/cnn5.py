import cv2
import numpy as np

# -----------------------------
# 1. LOAD IMAGE
# -----------------------------
img = cv2.imread(r"F:\shapes2.png")
if img is None:
    raise FileNotFoundError("Image not found")

# -----------------------------
# 2. PREPROCESSING
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

# -----------------------------
# 3. FIND CONTOURS
# -----------------------------
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# 4. SHAPE DETECTION
# -----------------------------
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100:
        continue

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(approx)

    # Solidity (important for star detection)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # -----------------------------
    # SHAPE CLASSIFICATION
    # -----------------------------
    if vertices == 3:
        shape = "Triangle"

    elif vertices == 4:
        aspect_ratio = w / float(h)

        # Diamond detection (rotated square)
        rect = cv2.minAreaRect(cnt)
        angle = abs(rect[2])

        if 0.85 <= aspect_ratio <= 1.15 and angle > 10:
            shape = "Diamond"
        elif 0.95 <= aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif vertices == 5:
        shape = "Pentagon"

    elif vertices == 6:
        shape = "Hexagon"

    else:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Star detection (many vertices + low solidity)
        if solidity < 0.75:
            shape = "Star"
        elif circularity > 0.80:
            shape = "Circle"
        else:
            shape = "Other"

    # -----------------------------
    # DRAW OUTPUT
    # -----------------------------
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        shape,
        (x + 5, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

# -----------------------------
# 5. SHOW RESULT
# -----------------------------
cv2.imshow("Shape Detection (Star & Diamond)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
