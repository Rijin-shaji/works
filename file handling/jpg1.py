#JPG/JPEG stands for Joint Photographic Experts Group.
#It is a lossy compression format for images, mainly photographs.
#Pros: Small file size, widely supported.
#Cons: Loses quality on repeated saves, not ideal for images with sharp edges or text.
#---Popular Python Libraries for JPG
#Pillow (PIL fork) – most common for image I/O and manipulation.
#OpenCV (cv2) – great for image processing and computer vision.
#matplotlib – for visualization.
#imageio – simple read/write.
#scikit-image – scientific image processing.
#PyTorch / TensorFlow – for machine learning tasks.

#Using Pillow
from PIL import Image
img = Image.open("example.jpg")
img.show()
#Pros: Handles EXIF metadata, simple API, good for basic image manipulation.

#Using OpenCV
import cv2
img = cv2.imread("example.jpg")  # returns NumPy array
cv2.imshow("Image", img)
#Pros: Best for computer vision and image processing.

#Using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the path to your image file
image_path = 'your_image_file.jpg'

# Read the image using mpimg.imread()
img = mpimg.imread(image_path)
    
    # Display the image
plt.imshow(img)
    
    # Optional: Turn off the axes for a cleaner view
plt.axis('off')
    
    # Show the plot window
plt.show()
#matplotlib.pyplot --displaying and ploting the data
#matplotlib.image  --read and manipulate image data

#Using imageio
import imageio #it is used to read and write  a wide range of image data
import num as np

# Specify the path to your JPG file
file_path = 'my_image.jpg' 

# Read the image data. This will raise an error if the file is missing or invalid.
image_data = imageio.imread(file_path)

# The image data is stored as a NumPy array
print(f"Successfully read image: {file_path}")
print(f"Image shape (Height, Width, Color Channels): {image_data.shape}")
print(f"Data type: {image_data.dtype}") #type of data an array can hold

#Using scikit-image
from skimage import io #FOr image filtering,analysis,and transformation
import num as np

# Specify the path to your JPG file
file_path = 'my_image.jpg'

# Read the image data
# WARNING: This line will raise an error (e.g., FileNotFoundError) 
# if the file doesn't exist or is corrupted, causing the program to crash.
image_data = io.imread(file_path)

# The image data is stored as a NumPy array
print(f"Successfully read image: {file_path}")
print(f"Image shape (Height, Width, Color Channels): {image_data.shape}")
print(f"Data type: {image_data.dtype}")

#why we use numpy in these
#An image, whether it's a JPEG, PNG, or TIFF, is fundamentally a grid of numbers representing pixel intensity and color.
#Grayscale Image: A 2D array
#Color (RGB) Image: A 3D array
#NumPy's core data structure is the multi-dimensional array (ndarray), which is the perfect, highly efficient structure to store these image grids.

#Using BytesIO (Pillow)
from PIL import Image #image manipulation and processing
from io import BytesIO # download image from the web and pass it ti pillor for the processing

with open("example.jpg", "rb") as f:
    img = Image.open(BytesIO(f.read()))

#--write
# Using Pillow
from PIL import Image

img = Image.open("example.jpg")  # or any PIL Image
img.save("output.jpg", quality=85, optimize=True, progressive=True)  # quality 0-100    
#quality=90 → controls lossy compression
#optimize=True → optimize file size
#progressive=True → It loads progressively for better perceived performance online.

#Using OpenCV
import cv2

# img as a NumPy array
cv2.imwrite("output.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])#file name,in image format,image quality

#Using imageio
import imageio
imageio.imwrite("output.jpg", img, quality=90)

#Using scikit-image
from skimage import io
io.imsave("output.jpg", img, quality=95)

#Using BytesIO + Pillow (in-memory saving)
from PIL import Image
from io import BytesIO

output = BytesIO()
img.save(output, format="JPEG", quality=85)
jpg_bytes = output.getvalue()

#Key Points About Writing JPG
#Compression: JPG is lossy → repeated saves reduce quality.
#Quality Control: Most libraries allow quality 0-100.
#Format Conversion: You can also save JPG as PNG, BMP, etc.
#In-memory saving: Pillow + BytesIO is useful for web apps or APIs.

#--visualizing
#Pillow
from PIL import Image

img = Image.open("example.jpg")
img.show()

#Using OpenCV
import cv2

img = cv2.imread("example.jpg")  # BGR format
cv2.imshow("Image", img)
cv2.waitKey(0)  # duration of the image  until a key is pressed
cv2.destroyAllWindows()#to close all windows that created by openCV

#Using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("example.jpg")
plt.imshow(img)
plt.axis('off')  # remove axes
plt.show()

#. Using OpenCV + Matplotlib Combination
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("example.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert colour

plt.imshow(img_rgb)
plt.axis('off')
plt.show()

#Using scikit-image
from skimage import io

img = io.imread("example.jpg")
io.imshow(img)
io.show()