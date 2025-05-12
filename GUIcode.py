import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from skimage.color import rgb2gray
from skimage import io
from skimage.measure import shannon_entropy

def extract_sobel_features(image_path):
    """
    Load an image from disk, convert to grayscale,
    apply Sobel edge detection, then compute
    variance, skewness, kurtosis, and entropy.
    Returns a 1x4 feature list.
    """
    # 1. Read and grayscale
    img = io.imread(image_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    # 2. Resize to 400x400 (or your training size)
    img = cv2.resize(img, (400, 400))
    # 3. Sobel in X and Y
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sobel_x, sobel_y)
    # 4. Compute statistical features
    var = np.var(grad)
    skew = np.mean(((grad - grad.mean()) / grad.std()) ** 3)
    kurt = np.mean(((grad - grad.mean()) / grad.std()) ** 4) - 3
    ent = shannon_entropy(grad)
    return np.array([[var, skew, kurt, ent]])


def predict_banknote(image_path):
    """
    Given a file path, extract Sobel features,
    apply the loaded model, and return a string.
    """
    features = extract_sobel_features(image_path)
    label = model.predict(features)[0]
    return "FAKE banknote" if label == 1 else "REAL banknote"


def select_image():
    global img_path, displayed_image
    img_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")],
    )
    if img_path:
        result = predict_banknote(img_path)
        prediction_label.config(text=f"Prediction: {result}")

        # Display the selected image
        try:
            img = Image.open(img_path)
            img = img.resize((250, 250))  # Resize the image to fit
            img_tk = ImageTk.PhotoImage(img)
            if displayed_image is None:
                displayed_image = Label(root, image=img_tk, bg="#f0f0f0")
                displayed_image.image = img_tk
                displayed_image.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
            else:
                displayed_image.config(image=img_tk)
                displayed_image.image = img_tk
        except Exception as e:
            print(f"Error displaying image: {e}")


root = tk.Tk()
root.geometry("800x700")
root.title("Fake Banknote Detection")
root.config(bg="#f0f0f0")  # Light gray background

# Load the trained Random Forest model
try:
    model = joblib.load("model.joblib")  # Corrected model filename
    print("Loaded model.joblib")
except FileNotFoundError:
    print("Error: model.joblib not found. Make sure it's in the correct directory.")
    exit()

# Title Label
title_font = ("Helvetica", 30, "bold")
title_label = tk.Label(root, text="Fake Banknote Detection", font=title_font, bg="#f0f0f0", fg="#333")
title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Select Image Button
button_font = ("Arial", 14)
select_button = tk.Button(root, text="Select Image", command=select_image, font=button_font, bg="#4CAF50", fg="white", padx=20, pady=10, relief=tk.RAISED, borderwidth=2)
select_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Prediction Label
prediction_font = ("Arial", 20, "italic")
prediction_label = tk.Label(root, text="Prediction: ", font=prediction_font, bg="#f0f0f0", fg="#555")
prediction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Placeholder for the displayed image
displayed_image = None

root.mainloop()