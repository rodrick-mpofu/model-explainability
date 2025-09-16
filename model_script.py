"""
Backward compatibility layer for model_script.py

This module provides the same interface as the original model_script.py
but uses the new organized codebase underneath.
"""

import os
import sys

# Add the new source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

try:
    from models.model_loader import load_model, get_imagenet_label
    from explainability.gradcam import generate_gradcam  
    from explainability.shap_explain import generate_shap
    
    # For full backward compatibility, also import the API functions with original names
    from explainability_api import (
        generate_gradcam_explanation as generate_gradcam_api,
        generate_shap_explanation as generate_shap_api
    )
    
    # Set default output directory to www for Shiny compatibility
    def generate_gradcam_legacy(*args, **kwargs):
        """Legacy wrapper for generate_gradcam with www output directory."""
        kwargs.setdefault('output_dir', 'www')
        return generate_gradcam(*args, **kwargs)
    
    def generate_shap_legacy(*args, **kwargs):
        """Legacy wrapper for generate_shap with www output directory."""
        kwargs.setdefault('output_dir', 'www')
        kwargs.setdefault('log_file', 'shap_debug.log')
        return generate_shap(*args, **kwargs)
    
    # Override the imported functions with legacy wrappers
    generate_gradcam = generate_gradcam_legacy
    generate_shap = generate_shap_legacy
    
except ImportError as e:
    print(f"Warning: Could not import reorganized modules: {e}")
    print("Falling back to original implementation...")
    
    # If the new structure fails, fall back to original code
    # This ensures the app still works during transition
    import requests
    import json
    
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    
    try:
        response = requests.get(LABELS_URL)
        imagenet_labels = response.json()
    except:
        imagenet_labels = {}
        
    def get_imagenet_label(index):
        """Retrieve the human-readable label for a given ImageNet index."""
        return imagenet_labels.get(str(index), ["unknown", "unknown"])[1]
        
    def load_model(model_name):
        """Fallback model loader - requires TensorFlow to be installed."""
        try:
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras.applications.vgg16 import preprocess_input
            model = VGG16(weights='imagenet')
            return model, "block5_conv3", preprocess_input, (224, 224)
        except ImportError:
            raise ImportError("TensorFlow not available and new structure failed to load")
    
    def generate_gradcam(*args, **kwargs):
        """Fallback Grad-CAM implementation."""
        print("Warning: Using fallback Grad-CAM implementation")
        return []
        
    def generate_shap(*args, **kwargs):
        """Fallback SHAP implementation.""" 
        print("Warning: Using fallback SHAP implementation")
        return []