"""
SHAP (SHapley Additive exPlanations) implementation.

This module provides functionality to generate SHAP-based explanations
for model predictions using game-theoretic approaches.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ..models.model_loader import get_imagenet_label


def generate_shap(img_path, model, preprocess, top_n=3, confidence_threshold=0.5, 
                 input_size=(224, 224), output_dir="outputs", log_file=None):
    """
    Generate SHAP-based explanations for top predictions.
    
    Args:
        img_path (str): Path to the input image
        model: Loaded Keras model
        preprocess: Preprocessing function for the model
        top_n (int): Number of top predictions to analyze
        confidence_threshold (float): Minimum confidence for predictions
        input_size (tuple): Input size for the model
        output_dir (str): Directory to save output images
        log_file (str): Optional path to log file for debugging
        
    Returns:
        list: List of prediction results with SHAP visualizations
    """
    # Set up logging if specified
    if log_file:
        _setup_logging(log_file)
    
    try:
        # Import SHAP here to avoid dependency issues if not installed
        import shap
        
        top_n = int(top_n)
        os.makedirs(output_dir, exist_ok=True)
        
        print("Reading and preprocessing image...")
        original_img = cv2.imread(img_path)
        original_img_resized = cv2.resize(original_img, input_size)
        img = np.expand_dims(original_img_resized, axis=0)
        img = preprocess(img)
        
        print("Predicting class scores...")
        preds = model.predict(img)
        preds = preds[0].flatten()
        
        print(f"Raw Predictions: {preds}")
        
        top_n_preds = np.argsort(preds)[-top_n:][::-1]
        print(f"Top predictions (before filtering): {top_n_preds}")
        
        top_predictions = []
        
        print("Creating SHAP explainer...")
        explainer = shap.Explainer(model, img)
        print("SHAP explainer initialized.")
        
        print("Generating SHAP values...")
        shap_values = explainer(img)
        print("SHAP values computed.")
        
        for pred_idx in top_n_preds:
            confidence = preds[pred_idx]
            
            print(f"Checking prediction {pred_idx}: Confidence = {confidence}")
            
            if confidence < confidence_threshold:
                print(f"Skipping {pred_idx} due to low confidence.")
                continue
            
            print(f"Generating SHAP heatmap for prediction index {pred_idx}...")
            
            # Create SHAP visualization
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
        
    except ImportError:
        print("🚨 ERROR: SHAP library not installed. Please install with: pip install shap")
        return []
    except Exception as e:
        print(f"🚨 ERROR: {e}")
        return []


def _setup_logging(log_file):
    """Set up logging to redirect stdout and stderr to log file."""
    # Clear previous logs
    with open(log_file, "w") as f:
        f.write("")
    
    # Redirect output to log file
    sys.stdout = open(log_file, "a")
    sys.stderr = sys.stdout
    
    print("\n--- SHAP Debugging Start ---")