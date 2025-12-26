import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load image and convert to grayscale
img = Image.open("F:/bmw-m5-m-3840x2160-21070.jpg").convert("L")
img_matrix = np.array(img)
print(img_matrix)
img.show()

# Step 2: Apply SVD
U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)

# Step 3: Choose k (compression level)
k = 5

Uk = U[:, :k]
Sk = np.diag(S[:k])
Vk = VT[:k, :]

# Step 4: Reconstruct image
img_reconstructed = Uk @ Sk @ Vk

# Step 5: Display
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Reconstructed Image (k={k})")
plt.imshow(img_reconstructed, cmap="gray")
plt.axis("off")

plt.show()
fro_error = np.linalg.norm(img_matrix - img_reconstructed, 'fro')
print("Frobenius norm error:", fro_error)

fro_error_from_sigma = np.sqrt(np.sum(S[k:]**2)) 
print(fro_error_from_sigma)

#20
k = 20

Uk = U[:, :k]
Sk = np.diag(S[:k])
Vk = VT[:k, :]

img_reconstructed = Uk @ Sk @ Vk

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Reconstructed Image (k={k})")
plt.imshow(img_reconstructed, cmap="gray")
plt.axis("off")

plt.show()
fro_error = np.linalg.norm(img_matrix - img_reconstructed, 'fro')
print("Frobenius norm error:", fro_error)

fro_error_from_sigma = np.sqrt(np.sum(S[k:]**2)) 
print(fro_error_from_sigma)

#50
k = 50

Uk = U[:, :k]
Sk = np.diag(S[:k])
Vk = VT[:k, :]

img_reconstructed = Uk @ Sk @ Vk

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Reconstructed Image (k={k})")
plt.imshow(img_reconstructed, cmap="gray")
plt.axis("off")

plt.show()
fro_error = np.linalg.norm(img_matrix - img_reconstructed, 'fro')
print("Frobenius norm error:", fro_error)

fro_error_from_sigma = np.sqrt(np.sum(S[k:]**2)) 
print(fro_error_from_sigma)
#100
k = 100

Uk = U[:, :k]
Sk = np.diag(S[:k])
Vk = VT[:k, :]

img_reconstructed = Uk @ Sk @ Vk

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Reconstructed Image (k={k})")
plt.imshow(img_reconstructed, cmap="gray")
plt.axis("off")

plt.show()
fro_error = np.linalg.norm(img_matrix - img_reconstructed, 'fro')
print("Frobenius norm error:", fro_error)

fro_error_from_sigma = np.sqrt(np.sum(S[k:]**2)) 
print(fro_error_from_sigma)