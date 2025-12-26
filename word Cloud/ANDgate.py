# Simple MLP using TensorFlow

import tensorflow as tf
import numpy as np

# 1. Create sample dataset (AND gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0,1,1,0], dtype=float)

# 2. Build MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
model.fit(X, y, epochs=1000, verbose=0)

# 5. Predict
predictions = model.predict(X)

# 6. Display results
print("MLP AND Gate Results:")
for inp, pred in zip(X, predictions):
    print(f"{int(inp[0])} AND {int(inp[1])} = {round(pred[0])}")
