"""
Image processing utilities.

This module provides common image processing functions used across
the explainability modules.
"""

import cv2
import numpy as np


def load_and_preprocess_image(img_path, input_size=(224, 224), preprocess_func=None):
    """
    Load and preprocess an image for model inference.
    
    Args:
        img_path (str): Path to the image file
        input_size (tuple): Target size for resizing
        preprocess_func: Optional preprocessing function to apply
        
    Returns:
        tuple: (original_image, preprocessed_image_batch)
    """
    # Read the original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Resize the image
    resized_img = cv2.resize(original_img, input_size)
    
    # Add batch dimension
    img_batch = np.expand_dims(resized_img, axis=0)
    
    # Apply preprocessing if provided
    if preprocess_func:
        img_batch = preprocess_func(img_batch)
    
    return original_img, img_batch


def save_image(image, filepath):
    """
    Save an image to the specified filepath.
    
    Args:
        image (numpy.ndarray): Image array to save
        filepath (str): Destination filepath
    """
    cv2.imwrite(filepath, image)


def resize_image(image, target_size):
    """
    Resize an image to the target size.
    
    Args:
        image (numpy.ndarray): Image to resize
        target_size (tuple): Target (width, height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    return cv2.resize(image, target_size)


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to a grayscale heatmap.
    
    Args:
        heatmap (numpy.ndarray): Grayscale heatmap
        colormap: OpenCV colormap constant
        
    Returns:
        numpy.ndarray: Colored heatmap
    """
    # Ensure heatmap is in uint8 format
    if heatmap.dtype != np.uint8:
        heatmap = np.uint8(255 * heatmap)
    
    return cv2.applyColorMap(heatmap, colormap)


def create_overlay(base_image, overlay, alpha=0.6, beta=0.4):
    """
    Create a weighted overlay of two images.
    
    Args:
        base_image (numpy.ndarray): Base image
        overlay (numpy.ndarray): Overlay image
        alpha (float): Weight for base image
        beta (float): Weight for overlay
        
    Returns:
        numpy.ndarray: Combined image
    """
    return cv2.addWeighted(base_image, alpha, overlay, beta, 0)