import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model
from scipy import ndimage

model = load_model("model.h5")

root = tk.Tk()
root.title("Handwritten Digit Spot Detection")

CANVAS_SIZE = 280
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.grid(row=0, column=0, columnspan=2)

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
draw = ImageDraw.Draw(image)

last_x, last_y = None, None


def draw_digit(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x:
        canvas.create_line(last_x, last_y, x, y,
                           width=20, fill="white", capstyle=tk.ROUND)
        draw.line([last_x, last_y, x, y], fill="white", width=20)
    last_x, last_y = x, y


def reset(event):
    global last_x, last_y
    last_x, last_y = None, None


def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="black")
    result_label.config(text="Prediction: ")


def predict_digit():
    # Get the drawn image
    img = image.copy()

    # Convert to numpy array
    img_array = np.array(img)

    # Find bounding box of the digit
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if not rows.any() or not cols.any():
        result_label.config(text="Prediction: Draw something first!")
        return

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box with padding
    padding = 20
    rmin = max(0, rmin - padding)
    rmax = min(CANVAS_SIZE, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(CANVAS_SIZE, cmax + padding)

    img_cropped = img_array[rmin:rmax, cmin:cmax]

    # Resize while maintaining aspect ratio
    img_pil = Image.fromarray(img_cropped)

    # Calculate new size maintaining aspect ratio
    width, height = img_pil.size
    if width > height:
        new_width = 20
        new_height = int(20 * height / width)
    else:
        new_height = 20
        new_width = int(20 * width / height)

    img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create 28x28 black image and paste centered
    img_final = Image.new('L', (28, 28), 0)
    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2
    img_final.paste(img_resized, (offset_x, offset_y))

    # Convert to numpy and normalize
    img_final_array = np.array(img_final)
    img_final_array = img_final_array / 255.0
    img_final_array = img_final_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_final_array, verbose=0)
    digit = np.argmax(prediction)
    confidence = prediction[0][digit] * 100

    result_label.config(text=f"Prediction: {digit} ({confidence:.1f}%)")


canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", reset)

tk.Button(root, text="Predict", command=predict_digit, width=15).grid(row=1, column=0)
tk.Button(root, text="Clear", command=clear_canvas, width=15).grid(row=1, column=1)

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16))
result_label.grid(row=2, column=0, columnspan=2)

root.mainloop()