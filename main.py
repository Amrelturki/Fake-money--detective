import cv2
import numpy as np
import joblib
from skimage.color import rgb2gray
from skimage import io
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from tkinter import filedialog  # Import filedialog for image selection


# Load the trained Random Forest model
model = joblib.load('model.joblib')  # Corrected model filename
print("Loaded model.joblib")



# Visualize Original vs Sobel‐Processed Image

# Define Feature Extraction:


def extract_sobel_features(image_path):
    """
    Load an image from disk, convert to grayscale,
    apply Sobel edge detection, then compute
    variance, skewness, kurtosis, and entropy.
    Returns a 1×4 feature list.
    """
    # 1. Read and grayscale
    img = io.imread(image_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    # 2. Resize to 400×400 (or your training size)
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


# Predict Function


def predict_banknote(image_path):
    """
    Given a file path, extract Sobel features,
    apply the loaded model, and return a string.
    """
    features = extract_sobel_features(image_path)
    label = model.predict(features)[0]
    return "FAKE banknote" if label == 1 else "REAL banknote"

# Run Prediction on Your Image

#  prompt for path and predict
def get_image_path():
    filepath = filedialog.askopenfilename(
        title="Select Banknote Image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")]
    )
    return filepath

img_path = get_image_path()  # Get image path from user
if img_path:
    result = predict_banknote(img_path)
    print("Prediction:", result)
else:
    print("No image selected.")