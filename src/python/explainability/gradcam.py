"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This module provides functionality to generate Grad-CAM visualizations
for understanding which parts of an image contributed to a model's prediction.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from ..models.model_loader import get_imagenet_label


def generate_gradcam(img_path, model, preprocess, top_n=3, confidence_threshold=0.5, 
                    last_conv_layer_name="block5_conv3", input_size=(224, 224), 
                    output_dir="outputs"):
    """
    Generate Grad-CAM for top predictions above a confidence threshold.
    
    Args:
        img_path (str): Path to the input image
        model: Loaded Keras model
        preprocess: Preprocessing function for the model
        top_n (int): Number of top predictions to analyze
        confidence_threshold (float): Minimum confidence for predictions
        last_conv_layer_name (str): Name of the last convolutional layer
        input_size (tuple): Input size for the model
        output_dir (str): Directory to save output images
        
    Returns:
        list: List of prediction results with Grad-CAM visualizations
    """
    top_n = int(top_n)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and preprocess the image
    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, input_size)
    img = np.expand_dims(original_img_resized, axis=0)
    img = preprocess(img)
    
    # Predict class scores
    preds = model.predict(img)
    preds = preds[0].flatten()
    top_n_preds = np.argsort(preds)[-top_n:][::-1]
    
    top_predictions = []
    
    for pred_idx in top_n_preds:
        confidence = preds[pred_idx]
        
        # Skip predictions below the confidence threshold
        if confidence < confidence_threshold:
            continue
        
        # Compute Grad-CAM for the current prediction
        heatmap = _compute_gradcam_heatmap(
            img, model, pred_idx, last_conv_layer_name
        )
        
        # Generate heatmap overlay
        gradcam_img = _create_gradcam_overlay(original_img, heatmap)
        
        # Save Grad-CAM output
        gradcam_path = os.path.join(output_dir, f"gradcam_output_{pred_idx}.png")
        cv2.imwrite(gradcam_path, gradcam_img)
        
        # Save the original resized image
        original_img_path = os.path.join(output_dir, "original_resized.png")
        cv2.imwrite(original_img_path, original_img_resized)
        
        # Append results
        top_predictions.append({
            "label": get_imagenet_label(pred_idx),
            "confidence": confidence,
            "gradcam_img_path": gradcam_path
        })
    
    return top_predictions


def _compute_gradcam_heatmap(img, model, class_idx, last_conv_layer_name):
    """
    Compute the Grad-CAM heatmap for a specific class.
    
    Args:
        img: Preprocessed input image
        model: Keras model
        class_idx: Index of the target class
        last_conv_layer_name: Name of the last convolutional layer
        
    Returns:
        numpy.ndarray: Normalized heatmap
    """
    with tf.GradientTape() as tape:
        # Get the last convolutional layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create a model that outputs both conv layer activations and predictions
        grads_model = Model([model.inputs], [last_conv_layer.output, model.output])
        
        # Get conv outputs and predictions
        conv_outputs, predictions = grads_model(img)
        
        # Get the score for the target class
        class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Pool gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()


def _create_gradcam_overlay(original_img, heatmap):
    """
    Create a Grad-CAM overlay on the original image.
    
    Args:
        original_img: Original image array
        heatmap: Normalized heatmap
        
    Returns:
        numpy.ndarray: Image with Grad-CAM overlay
    """
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert to uint8 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Create weighted overlay
    gradcam_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return gradcam_img