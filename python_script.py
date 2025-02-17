import tensorflow as tf
import numpy as np
import shap
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

# Load the pre-trained ImageNet model (e.g., VGG16)
model = VGG16(weights='imagenet')

# Preprocess image function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}. Check the file path or format.")
    
    img = cv2.resize(img, (224, 224))  # Resize the image to (224, 224)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32')  # Convert image type to float32
    img = preprocess_input(img)  # Preprocess for VGG16 or other models
    return img

# Generate SHAP explanations
def generate_shap_explanations(img_path, model):
    # Preprocess the image
    img = preprocess_image(img_path)

    # Create a background of zero images matching the input shape
    background = np.zeros((1, 224, 224, 3))  # Zero image as background

    # Use the GradientExplainer for deep learning models
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values for the input image
    shap_values = explainer.shap_values(img)

    # Ensure that SHAP values are not zero-dimensional
    if isinstance(shap_values, list) and len(shap_values) > 0:
        shap_values = shap_values[0]  # Access the first class's SHAP values

    # Convert SHAP values into a heatmap by summing across color channels
    if len(shap_values.shape) == 4:  # Expecting a 4D array (batch, height, width, channels)
        heatmap = np.sum(shap_values, axis=-1)  # Sum over color channels
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU to keep positive attributions
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) != 0 else heatmap  # Normalize
    else:
        raise ValueError("SHAP values have unexpected dimensions: " + str(shap_values.shape))

    # Get the predicted class label
    preds = model.predict(img)
    pred_class = np.argmax(preds[0])

    # ImageNet labels (download this or use a package)
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    import json
    import requests
    response = requests.get(LABELS_URL)
    imagenet_labels = response.json()

    def get_imagenet_label(index):
        return imagenet_labels[str(index)][1]

    predicted_label = get_imagenet_label(pred_class)

    # Save the SHAP heatmap
    output_path = "output_shap.png"
    cv2.imwrite(output_path, np.uint8(255 * heatmap))  # Save the heatmap

    return heatmap, predicted_label
