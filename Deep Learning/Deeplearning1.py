# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical # Corrected import

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ===============================
# 2. LOAD DATASET
# ===============================
X, y = load_iris(return_X_y=True)

# One-hot encode labels (for multiclass)
y = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 3. BUILD MLP MODEL
# ===============================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')   # 3 output classes
])


# ===============================
# 4. COMPILE MODEL
# ===============================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ===============================
# 5. MODEL SUMMARY
# ===============================
model.summary()


# ===============================
# 6. TRAIN MODEL
# ===============================
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    verbose=0
)


# ===============================
# 7. EVALUATE MODEL
# ===============================
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)


# ===============================
# 8. PREDICTIONS
# ===============================
predictions = model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1)
true_classes = tf.argmax(y_test, axis=1)

print("\nPredicted Classes:", predicted_classes.numpy())
print("True Classes     :", true_classes.numpy())