# ==========================================
# Leaf Disease Prediction using CNN
# ==========================================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# 1. PARAMETERS
# -------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# CORRECTED: Use parent directory containing both class folders
# Expected structure:
# dataset/
#   ├── Pepper__bell___healthy/
#   └── Pepper__bell___Bacterial_spot/
TRAIN_DIR = "F:\leaf"  # Parent directory with both classes
VAL_DIR = "F:\leaf"  # Can use same for simplicity or split manually

# -------------------------------
# 2. DATA PREPROCESSING (NO SCIPY NEEDED)
# -------------------------------
# Simple rescaling only - no augmentation requiring scipy
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # Use 20% for validation
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Training data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset='training'  # Set as training data
)

# Validation data
val_data = train_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset='validation'  # Set as validation data
)

print(f"\nFound {train_data.samples} training images")
print(f"Found {val_data.samples} validation images")
print(f"Classes: {train_data.class_indices}")

# -------------------------------
# 3. CNN MODEL
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# -------------------------------
# 4. COMPILE MODEL
# -------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 5. TRAIN MODEL
# -------------------------------
# Calculate steps per epoch
steps_per_epoch = train_data.samples // BATCH_SIZE
validation_steps = val_data.samples // BATCH_SIZE

history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    validation_data=val_data,
    validation_steps=validation_steps
)

# -------------------------------
# 6. SAVE MODEL
# -------------------------------
model.save("leaf_disease_cnn.h5")
print("\n✅ Model saved as leaf_disease_cnn.h5")

# -------------------------------
# 7. ACCURACY & LOSS GRAPHS
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history.history["accuracy"], label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.set_title("Model Accuracy")
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Validation Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Model Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("\n📊 Training graphs saved as training_history.png")


# -------------------------------
# 8. TEST ON A SINGLE IMAGE
# -------------------------------
def predict_leaf(image_path):
    """Predict if a leaf is healthy or diseased"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    # Get class names
    class_names = {v: k for k, v in train_data.class_indices.items()}

    if prediction > 0.5:
        print(f"🔴 Prediction: {class_names[1]} (Confidence: {prediction * 100:.2f}%)")
    else:
        print(f"🟢 Prediction: {class_names[0]} (Confidence: {(1 - prediction) * 100:.2f}%)")

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_names[1] if prediction > 0.5 else class_names[0]}")
    plt.show()


TEST_IMAGE = 'F:/a.JPG'  # Change this to your image path

if os.path.exists(TEST_IMAGE):
    print("\n" + "="*50)
    print("🔍 TESTING PREDICTION ON IMAGE")
    print("="*50)
    predict_leaf(TEST_IMAGE)
else:
    print(f"\n⚠️ Test image '{TEST_IMAGE}' not found.")
    print("To test prediction, change TEST_IMAGE path to an actual image file.")
    print("Example: TEST_IMAGE = 'F:/my_leaf_image.jpg'")

print("\n✅ Training complete! Use predict_leaf('path/to/image.jpg') to test new images.")