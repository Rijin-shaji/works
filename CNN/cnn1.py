import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# Load your trained MNIST CNN
model = load_model("model.h5")

# Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Spot Detection")

# Canvas
CANVAS_SIZE = 280
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.grid(row=0, column=0, columnspan=2)

# PIL image to draw on
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
draw = ImageDraw.Draw(image)

last_x, last_y = None, None


# Draw function
def draw_digit(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x:
        canvas.create_line(last_x, last_y, x, y,
                           width=15, fill="white", capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill="white", width=15)
    last_x, last_y = x, y


def reset(event):
    global last_x, last_y
    last_x, last_y = None, None


canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", reset)


# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="black")
    result_label.config(text="Prediction: ")


# Preprocess image - FIXED VERSION
def preprocess(img):
    # Convert to numpy array (already white on black)
    img = np.array(img)

    # Find bounding box of the drawn digit
    coords = np.column_stack(np.where(img > 50))
    if coords.shape[0] == 0:
        return np.zeros((28, 28))

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Add padding
    pad = 10
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(CANVAS_SIZE, y1 + pad)
    x1 = min(CANVAS_SIZE, x1 + pad)

    img_cropped = img[y0:y1 + 1, x0:x1 + 1]

    # Resize to 20x20 while maintaining aspect ratio
    h, w = img_cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    img_resized = Image.fromarray(img_cropped).resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Center digit on 28x28 canvas
    img_array = np.array(img_resized)
    h, w = img_array.shape
    new_img = np.zeros((28, 28))
    pad_h = (28 - h) // 2
    pad_w = (28 - w) // 2
    new_img[pad_h:pad_h + h, pad_w:pad_w + w] = img_array

    # Normalize
    new_img = new_img / 255.0

    return new_img


# Predict function
def predict_digit():
    img = preprocess(image)
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    result_label.config(text=f"Prediction: {digit} ({confidence:.1f}%)")


# Buttons
tk.Button(root, text="Predict", command=predict_digit, width=15).grid(row=1, column=0, pady=10)
tk.Button(root, text="Clear", command=clear_canvas, width=15).grid(row=1, column=1, pady=10)

# Result label
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16))
result_label.grid(row=2, column=0, columnspan=2)

root.mainloop()