import numpy as np
from PIL import Image
#
# # Load image (grayscale)
# img = Image.open("F:/bmw-m5-m-3840x2160-21070.jpg").convert("L")  # L = grayscale
# img = img.resize((28,28))  # resize to 28x28
# img_array = np.array(img)
#
# # Flatten
# img_flat = img_array.flatten()
# print(img_flat.shape)




# # Load image and convert to grayscale
# shrinked_img = Image.open("F:/bmw-m5-m-3840x2160-21070.jpg").convert("L")
#
# # Resize/shrink to 28x28
# shrinked_img = shrinked_img.resize((28,28), resample=Image.Resampling.LANCZOS)
#
# # Convert to numpy array and normalize
# img_array = np.array(shrinked_img) / 255.0
#
# # Reshape for CNN input (1 sample, 28x28, 1 channel)
# img_array = img_array.reshape(1,28,28,1)
#   # for grayscale CNN
# shrinked_img.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # Read image using PIL
# img = Image.open("F:/bmw-m5-m-3840x2160-21070.jpg").convert("L")
#
# # Convert PIL image to NumPy array
# img = np.array(img)
#
# # Canny edge detection
# edges = cv2.Canny(img, 100, 200)
#
# # Show result
# plt.imshow(edges, cmap="gray")
# plt.title("Canny Edge Detection")
# plt.axis("off")
# plt.show()

