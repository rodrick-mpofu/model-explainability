"""
Model loader for pre-trained neural networks.

This module provides functionality to load various pre-trained models
including VGG16, ResNet50, MobileNetV2, and EfficientNet variants.
"""

import requests
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7, preprocess_input as preprocess_input_efficientnet


# Cache for ImageNet labels
_imagenet_labels = None

def get_imagenet_labels():
    """Load and cache ImageNet labels."""
    global _imagenet_labels
    if _imagenet_labels is None:
        labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        response = requests.get(labels_url)
        _imagenet_labels = response.json()
    return _imagenet_labels


def get_imagenet_label(index):
    """Retrieve the human-readable label for a given ImageNet index."""
    labels = get_imagenet_labels()
    return labels[str(index)][1]


def load_model(model_name):
    """
    Load a specified pre-trained model and return model configuration.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, last_conv_layer_name, preprocess_function, input_size)
        
    Raises:
        ValueError: If unsupported model type is specified
    """
    model_configs = {
        "vgg16": {
            "model_class": VGG16,
            "last_conv_layer": "block5_conv3",
            "preprocess": preprocess_input,
            "input_size": (224, 224)
        },
        "resnet50": {
            "model_class": ResNet50,
            "last_conv_layer": "conv5_block3_out",
            "preprocess": preprocess_input_resnet,
            "input_size": (224, 224)
        },
        "mobilenet_v2": {
            "model_class": MobileNetV2,
            "last_conv_layer": "Conv_1",
            "preprocess": preprocess_input_mobilenet,
            "input_size": (224, 224)
        },
        "efficientnetb0": {
            "model_class": EfficientNetB0,
            "last_conv_layer": "top_conv",
            "preprocess": preprocess_input_efficientnet,
            "input_size": (224, 224)
        },
        "efficientnetb7": {
            "model_class": EfficientNetB7,
            "last_conv_layer": "top_conv",
            "preprocess": preprocess_input_efficientnet,
            "input_size": (600, 600)
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    config = model_configs[model_name]
    
    # Handle EfficientNet models with TF 2.20.0 compatibility issues
    if model_name.startswith('efficientnet'):
        raise ValueError(
            f"EfficientNet models are temporarily unavailable due to TensorFlow 2.20.0 compatibility issues. "
            f"Please use VGG16, ResNet50, or MobileNetV2 instead. "
            f"Error: Shape mismatch in EfficientNet weights. "
            f"This is a known issue with TensorFlow 2.20.0 + Keras 3.11.3."
        )
    
    # Load other models normally
    try:
        model = config["model_class"](weights='imagenet')
    except Exception as e:
        print(f"⚠️  Failed to load {model_name} with weights: {e}")
        print(f"🔄 Loading {model_name} without pre-trained weights...")
        model = config["model_class"](weights=None)
    
    return (
        model,
        config["last_conv_layer"],
        config["preprocess"],
        config["input_size"]
    )