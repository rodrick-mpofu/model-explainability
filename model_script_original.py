import setuptools.dist
import shap
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7, preprocess_input as preprocess_input_efficientnet
import os
import json
import requests
import matplotlib.pyplot as plt
import sys
#import distutils

os.system("source /srv/shiny-server/SYE/.venv/bin/activate")

# Load ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = response.json()

def get_imagenet_label(index):
    """Retrieve the human-readable label for a given ImageNet index."""
    return imagenet_labels[str(index)][1]

def load_model(model_name):
    """Load a specified pre-trained model and determine the last convolutional layer name and input size."""
    if model_name == "vgg16":
        model = VGG16(weights='imagenet')
        last_conv_layer_name = "block5_conv3"
        preprocess = preprocess_input
        input_size = (224, 224)
    elif model_name == "resnet50":
        model = ResNet50(weights='imagenet')
        last_conv_layer_name = "conv5_block3_out"
        preprocess = preprocess_input_resnet
        input_size = (224, 224)
    elif model_name == "mobilenet_v2":
        model = MobileNetV2(weights='imagenet')
        last_conv_layer_name = "Conv_1"
        preprocess = preprocess_input_mobilenet
        input_size = (224, 224)
    elif model_name == "efficientnetb0":
        model = EfficientNetB0(weights='imagenet')
        last_conv_layer_name = "top_conv"
        preprocess = preprocess_input_efficientnet
        input_size = (224, 224)
    elif model_name == "efficientnetb7":
        model = EfficientNetB7(weights='imagenet')
        last_conv_layer_name = "top_conv"
        preprocess = preprocess_input_efficientnet
        input_size = (600, 600)
    else:
        raise ValueError("Unsupported model type")

    return model, last_conv_layer_name, preprocess, input_size

def generate_gradcam(img_path, model, preprocess, top_n=3, confidence_threshold=0.5, last_conv_layer_name="block5_conv3",  input_size=(224, 224)):
    """Generate Grad-CAM for top predictions above a confidence threshold."""

    top_n = int(top_n)

    # Ensure we're saving images to the "www" folder (used by Shiny)
    output_dir = "www"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Read and preprocess the image
    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, input_size)  # Resized to 224x224 for consistency
    img = np.expand_dims(original_img_resized, axis=0)
    img = preprocess(img)

    # Predict class scores
    preds = model.predict(img)
    preds = preds[0].flatten()  # Flatten the predictions array to 1D
    top_n_preds = np.argsort(preds)[-top_n:][::-1]  # Get top N sorted predictions
    top_predictions = []

    for pred_idx in top_n_preds:
        confidence = preds[pred_idx]

        # Skip predictions below the confidence threshold
        if confidence < confidence_threshold:
            continue

        # Compute Grad-CAM for the current prediction (pred_idx)
        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer(last_conv_layer_name)
            grads_model = Model([model.inputs], [last_conv_layer.output, model.output])
            conv_outputs, predictions = grads_model(img)
            top_class_channel = predictions[:, pred_idx]
            grads = tape.gradient(top_class_channel, conv_outputs)
            pooled_grads = np.mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            conv_outputs = np.dot(conv_outputs, pooled_grads[..., np.newaxis])
            heatmap = np.maximum(conv_outputs, 0)
            heatmap /= np.max(heatmap)

        # Generate heatmap and overlay on original image
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        gradcam_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Save Grad-CAM output in "www" folder for each prediction
        gradcam_path = os.path.join(output_dir, f"gradcam_output_{pred_idx}.png")
        cv2.imwrite(gradcam_path, gradcam_img)

        # Save the original resized image as well
        original_img_path = os.path.join(output_dir, "original_resized.png")
        cv2.imwrite(original_img_path, original_img_resized)

        # Append results
        top_predictions.append({
            "label": get_imagenet_label(pred_idx),  # Get human-readable label
            "confidence": confidence,
            "gradcam_img_path": gradcam_path  # Path to the saved Grad-CAM image
        })

    return top_predictions

def generate_shap(img_path, model, preprocess, top_n=3, confidence_threshold=0.5, input_size=(224, 224)):
    """Generate SHAP-based explanations for top predictions."""

    log_file = "/srv/shiny-server/SYE/shap_debug.log"

    # ✅ Clear previous logs
    with open(log_file, "w") as f:
        f.write("")

    # Redirect output to log file
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout

    print("\n--- SHAP Debugging Start ---")

    try:
        top_n = int(top_n)
        output_dir = "www"
        os.makedirs(output_dir, exist_ok=True)

        print("Reading and preprocessing image...")
        original_img = cv2.imread(img_path)
        original_img_resized = cv2.resize(original_img, input_size)
        img = np.expand_dims(original_img_resized, axis=0)
        img = preprocess(img)

        print("Predicting class scores...")
        preds = model.predict(img)
        preds = preds[0].flatten()

        # ✅ Print predictions before thresholding
        print(f"Raw Predictions: {preds}")

        top_n_preds = np.argsort(preds)[-top_n:][::-1]
        print(f"Top predictions (before filtering): {top_n_preds}")

        top_predictions = []

        print("Creating SHAP explainer...")
        explainer = shap.Explainer(model, img)
        print("SHAP explainer initialized.")

        print("Generating SHAP values...")
        shap_values = explainer(img)  # Compute SHAP values
        print("SHAP values computed.")

        for pred_idx in top_n_preds:
            confidence = preds[pred_idx]

            print(f"Checking prediction {pred_idx}: Confidence = {confidence}")

            if confidence < confidence_threshold:
                print(f"Skipping {pred_idx} due to low confidence.")
                continue

            print(f"Generating SHAP heatmap for prediction index {pred_idx}...")

            fig, ax = plt.subplots()
            shap.image_plot(shap_values.values, img, show=False)
            shap_output_path = os.path.join(output_dir, f"shap_output_{pred_idx}.png")

            print(f"Saving SHAP output to: {shap_output_path}")
            plt.savefig(shap_output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print(f"SHAP output saved for {pred_idx}")

            top_predictions.append({
                "label": get_imagenet_label(pred_idx),
                "confidence": confidence,
                "shap_img_path": shap_output_path
            })

        if len(top_predictions) == 0:
            print("🚨 No valid predictions found above the threshold!")

        return top_predictions

    except Exception as e:
        print(f"🚨 ERROR: {e}")
        return []
