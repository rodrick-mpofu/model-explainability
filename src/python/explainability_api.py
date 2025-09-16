"""
Main API for model explainability.

This module provides a unified interface for generating explanations
using different techniques (Grad-CAM, SHAP) with various models.
"""

import os
from .models.model_loader import load_model
from .explainability.gradcam import generate_gradcam
from .explainability.shap_explain import generate_shap


def explain_image(img_path, model_name="vgg16", technique="gradcam", top_n=3, 
                 confidence_threshold=0.5, output_dir="outputs"):
    """
    Generate explanations for an image using the specified technique and model.
    
    Args:
        img_path (str): Path to the input image
        model_name (str): Name of the model to use
        technique (str): Explanation technique ('gradcam' or 'shap')
        top_n (int): Number of top predictions to analyze
        confidence_threshold (float): Minimum confidence for predictions
        output_dir (str): Directory to save output files
        
    Returns:
        list: List of prediction results with explanations
        
    Raises:
        ValueError: If unsupported technique or model is specified
        FileNotFoundError: If image file doesn't exist
    """
    # Validate inputs
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    if technique not in ["gradcam", "shap"]:
        raise ValueError(f"Unsupported technique: {technique}")
    
    # Load the model
    model, last_conv_layer_name, preprocess, input_size = load_model(model_name)
    
    # Generate explanations based on technique
    if technique == "gradcam":
        return generate_gradcam(
            img_path=img_path,
            model=model,
            preprocess=preprocess,
            top_n=top_n,
            confidence_threshold=confidence_threshold,
            last_conv_layer_name=last_conv_layer_name,
            input_size=input_size,
            output_dir=output_dir
        )
    elif technique == "shap":
        log_file = os.path.join(output_dir, "shap_debug.log")
        return generate_shap(
            img_path=img_path,
            model=model,
            preprocess=preprocess,
            top_n=top_n,
            confidence_threshold=confidence_threshold,
            input_size=input_size,
            output_dir=output_dir,
            log_file=log_file
        )


# Convenience functions for backward compatibility
def generate_gradcam_explanation(img_path, model, preprocess, top_n=3, 
                                confidence_threshold=0.5, last_conv_layer_name="block5_conv3", 
                                input_size=(224, 224)):
    """Backward compatibility function for Grad-CAM generation."""
    return generate_gradcam(
        img_path=img_path,
        model=model,
        preprocess=preprocess,
        top_n=top_n,
        confidence_threshold=confidence_threshold,
        last_conv_layer_name=last_conv_layer_name,
        input_size=input_size,
        output_dir="www"  # Default to www for Shiny compatibility
    )


def generate_shap_explanation(img_path, model, preprocess, top_n=3, 
                             confidence_threshold=0.5, input_size=(224, 224)):
    """Backward compatibility function for SHAP generation."""
    return generate_shap(
        img_path=img_path,
        model=model,
        preprocess=preprocess,
        top_n=top_n,
        confidence_threshold=confidence_threshold,
        input_size=input_size,
        output_dir="www",  # Default to www for Shiny compatibility
        log_file="shap_debug.log"
    )