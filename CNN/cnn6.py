# ==========================================
# Iris Flower Image Classification using CNN
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
EPOCHS = 5

# CHANGE THIS: Point to your iris dataset folder
# Expected structure:
# iris_dataset/
#   ├── Iris-setosa/
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   │   └── ...
#   ├── Iris-versicolor/
#   │   ├── image1.jpg
#   │   └── ...
#   └── Iris-virginica/
#       ├── image1.jpg
#       └── ...

DATASET_DIR = "F:/archive (2)"  # ← CHANGE THIS to your iris dataset path

# -------------------------------
# 2. DATA PREPROCESSING (NO SCIPY)
# -------------------------------
print("Loading Iris Image Dataset...")

# Simple rescaling only - no augmentation requiring scipy
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2  # Use 20% for validation
)

# Training data
train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # 3 classes (not binary)
    subset='training'
)

# Validation data
val_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)

print(f"\n✅ Found {train_data.samples} training images")
print(f"✅ Found {val_data.samples} validation images")
print(f"\n🌸 Classes detected: {train_data.class_indices}")
print(f"Number of classes: {len(train_data.class_indices)}")

# -------------------------------
# 3. CNN MODEL FOR 3 CLASSES
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")  # 3 classes: Setosa, Versicolor, Virginica
])

# -------------------------------
# 4. COMPILE MODEL
# -------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # For multi-class
    metrics=["accuracy"]
)

print("\n📋 Model Architecture:")
model.summary()

# -------------------------------
# 5. TRAIN MODEL
# -------------------------------
print("\n🚀 Training CNN model on Iris images...")

# Calculate steps per epoch
steps_per_epoch = train_data.samples // BATCH_SIZE
validation_steps = val_data.samples // BATCH_SIZE

history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_data,
    validation_steps=validation_steps
)

# -------------------------------
# 6. SAVE MODEL
# -------------------------------
model.save("iris_flower_cnn.h5")
print("\n✅ Model saved as iris_flower_cnn.h5")

# -------------------------------
# 7. ACCURACY & LOSS GRAPHS
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title("Model Accuracy", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Model Loss", fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("iris_training_history.png", dpi=300)
plt.show()

print("\n📊 Training graphs saved as iris_training_history.png")


# -------------------------------
# 8. PREDICTION FUNCTION
# -------------------------------
def predict_iris_flower(image_path):
    """Identify which type of Iris flower (Setosa, Versicolor, or Virginica)"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Load and preprocess image
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)

    # Get class names
    class_names = {v: k for k, v in train_data.class_indices.items()}
    predicted_species = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100

    # Emoji for each species
    species_emoji = {
        "Iris-setosa": "🌼",
        "Iris-versicolor": "🌺",
        "Iris-virginica": "🌸"
    }
    emoji = species_emoji.get(predicted_species, "🌷")

    # Display results
    print("\n" + "=" * 60)
    print("🌸 IRIS FLOWER TYPE IDENTIFICATION")
    print("=" * 60)
    print(f"{emoji} Identified Type: {predicted_species.upper()}")
    print(f"   Confidence: {confidence:.2f}%")
    print("\n📊 Classification Scores:")
    for idx, species in class_names.items():
        emoji_sp = species_emoji.get(species, "🌷")
        print(f"   {emoji_sp} {species}: {predictions[idx] * 100:.2f}%")
    print("=" * 60)

    # Display image with prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{emoji} Iris Type: {predicted_species}\nConfidence: {confidence:.1f}%",
              fontsize=16, fontweight='bold', color='darkgreen')
    plt.tight_layout()
    plt.show()

    return predicted_species, confidence


# -------------------------------
# 9. TEST - IDENTIFY IRIS TYPE
# -------------------------------
print("\n" + "=" * 60)
print("🧪 IDENTIFYING IRIS FLOWER TYPE FROM IMAGE")
print("=" * 60)
print("\nThis model identifies which type of Iris flower:")
print("  🌼 Iris-setosa")
print("  🌺 Iris-versicolor")
print("  🌸 Iris-virginica")
print("=" * 60)

# CHANGE THIS PATH to your test image
TEST_IMAGE = 'F:/b.jpg'  # ← CHANGE THIS!

if os.path.exists(TEST_IMAGE):
    predict_iris_flower(TEST_IMAGE)
else:
    print(f"\n⚠️ Test image not found: {TEST_IMAGE}")
    print("Please update TEST_IMAGE path to an actual iris flower image.")
    print("\nExample paths:")
    print('  TEST_IMAGE = "F:/iris_dataset/Iris-setosa/flower1.jpg"')
    print('  TEST_IMAGE = "F:/iris_dataset/Iris-versicolor/flower2.jpg"')
    print('  TEST_IMAGE = "F:/iris_dataset/Iris-virginica/flower3.jpg"')
    print("\nYou can also manually test by running:")
    print("predict_iris_flower('path/to/your/iris_image.jpg')")

print("\n✅ Training complete! Model can identify all 3 iris types.")