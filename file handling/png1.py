#PNG = Portable Network Graphics
#Type: Lossless image format (no quality loss when compressed)
#Using Pillow
from PIL import Image

# Open a PNG image
img = Image.open("image.png")

# Display the image
img.show()

# Image info
print("Format:", img.format)  # PNG
print("Size:", img.size)      # (width, height)
print("Mode:", img.mode)      # RGB, RGBA, L (grayscale)

#Using OpenCV
import cv2

# Read PNG with alpha channel preserved
img = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)

print("Shape:", img.shape)  # (height, width, channels)
cv2.imshow("PNG Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Using imageio
import imageio

# Read PNG
img = imageio.imread("image.png")
print("Shape:", img.shape)
print("Data type:", img.dtype)

#Using PyPNG
import png

# Read PNG
reader = png.Reader(filename="image.png")
width, height, pixels, metadata = reader.read()
pixels = list(pixels)

print("Width:", width)
print("Height:", height)
print("Metadata:", metadata)

#--write
#Using Pillow
from PIL import Image
import num as np

# Example: create a simple RGB image using NumPy
img_array = np.zeros((200, 200, 3), dtype=np.uint8)
img_array[:, :] = [255, 0, 0]  # Red

# Convert to PIL Image
img = Image.fromarray(img_array)

# Save as PNG
img.save("red_image.png")

# Save with optimization and compression
img.save("red_image_optimized.png", optimize=True, compress_level=9)

#Using OpenCV
import cv2
import num as np

# Create an image using NumPy
img = np.zeros((200, 200, 4), dtype=np.uint8)  # 4 channels (RGBA)
img[:, :] = [0, 255, 0, 128]  # Semi-transparent green

# Save as PNG
cv2.imwrite("green_image.png", img)

#Using imageio
import imageio
import num as np

# Create an RGBA image
img = np.zeros((200, 200, 4), dtype=np.uint8)
img[:, :] = [0, 0, 255, 255]  # Blue

# Save PNG
imageio.imwrite("blue_image.png", img)

#Using PyPNG
import png

# Create a simple grayscale image 5x5
pixels = [
    [0, 64, 128, 192, 255],
    [255, 192, 128, 64, 0],
    [0, 64, 128, 192, 255],
    [255, 192, 128, 64, 0],
    [0, 64, 128, 192, 255]
]

# Write PNG
with open("grayscale.png", "wb") as f:
    writer = png.Writer(width=5, height=5, greyscale=True)
    writer.write(f, pixels)

#--visualizing
#Using Pillow
from PIL import Image

# Open the PNG
img = Image.open("image.png")

# Display image in default image viewer
img.show()

#Using Matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Open PNG
img = Image.open("image.png")

# Display using matplotlib
plt.imshow(img)
plt.axis('off')  # Remove axes
plt.show()

#Using OpenCV
import cv2

# Read PNG (preserve alpha)
img = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)

# Display
cv2.imshow("PNG Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Using imageio
import imageio
import matplotlib.pyplot as plt

img = imageio.imread("image.png")
plt.imshow(img)
plt.axis('off')
plt.show()
