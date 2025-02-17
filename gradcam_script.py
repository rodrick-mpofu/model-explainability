import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import os
import json
import requests


# Load the pre-trained ImageNet model (e.g., VGG16)
model = VGG16(weights='imagenet')

# Load ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = response.json()

def get_imagenet_label(index):
    """Retrieve the human-readable label for a given ImageNet index."""
    return imagenet_labels[str(index)][1]

def generate_gradcam(img_path, model, top_n=3, confidence_threshold=0.5, last_conv_layer_name="block5_conv3"):
    """Generate Grad-CAM for top predictions above a confidence threshold."""

    top_n = int(top_n)

    # Ensure we're saving images to the "www" folder (used by Shiny)
    output_dir = "www"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, (224, 224))  # Resized to 224x224 for VGG16
    img = np.expand_dims(original_img_resized, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    preds = preds[0].flatten()  # Flatten the predictions array to 1D
    top_n_preds = np.argsort(preds)[-top_n:][::-1]  # Get top N sorted predictions
    top_predictions = []

    # for pred_idx in top_n_preds:
    #     confidence = preds[pred_idx]
    # 
    #     # Skip predictions below the confidence threshold
    #     if confidence < confidence_threshold:
    #         continue
    # 
    #     # Compute Grad-CAM for the current prediction (pred_idx)
    #     with tf.GradientTape() as tape:
    #         last_conv_layer = model.get_layer(last_conv_layer_name)
    #         grads_model = Model([model.inputs], [last_conv_layer.output, model.output])
    #         conv_outputs, predictions = grads_model(img)
    #         top_class_channel = predictions[:, pred_idx]
    #         grads = tape.gradient(top_class_channel, conv_outputs)
    #         pooled_grads = np.mean(grads, axis=(0, 1, 2))
    #         conv_outputs = conv_outputs[0]
    #         conv_outputs = np.dot(conv_outputs, pooled_grads[..., np.newaxis])
    #         heatmap = np.maximum(conv_outputs, 0)
    #         heatmap /= np.max(heatmap)
    # 
    #     # Generate heatmap and overlay on original image
    #     heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    #     heatmap = np.uint8(255 * heatmap)
    #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #     gradcam_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    # 
    #     # Save Grad-CAM output in "www" folder for each prediction
    #     gradcam_path = os.path.join(output_dir, f"gradcam_output_{pred_idx}.png")
    #     cv2.imwrite(gradcam_path, gradcam_img)
    # 
    #     # Save the original resized image as well
    #     original_img_path = os.path.join(output_dir, "original_resized.png")
    #     cv2.imwrite(original_img_path, original_img_resized)
    # 
    #     # Append results
    #     top_predictions.append({
    #         "label": get_imagenet_label(pred_idx),  # Get human-readable label
    #         "confidence": confidence,
    #         "gradcam_img_path": gradcam_path  # Path to the saved Grad-CAM image
    #     })
    
    for pred_idx in top_n_preds:
      confidence = preds[pred_idx]
  
      # Skip predictions below the confidence threshold
      if confidence < confidence_threshold:
          continue
  
      # Compute Grad-CAM for the current prediction (pred_idx)
      with tf.GradientTape() as tape:
          # Get the last convolutional layer in the model using its name
          last_conv_layer = model.get_layer(last_conv_layer_name)
  
          # Create a new model that outputs both the activations of the last conv layer
          # and the predictions. This is needed to compute the gradients later on.
          grads_model = Model([model.inputs], [last_conv_layer.output, model.output])
  
          # Pass the input image through the model to get the feature maps (from the last conv layer)
          # and the final predictions (output layer of the model).
          conv_outputs, predictions = grads_model(img)
  
          # Extract the predicted score for the current class (pred_idx)
          # This is the "output" or class score that we are interested in.
          top_class_channel = predictions[:, pred_idx]
  
          # Compute the gradients of the predicted class score (top_class_channel) 
          # with respect to the feature maps (conv_outputs) of the last convolutional layer.
          # The GradientTape records all the operations needed to calculate these gradients.
          grads = tape.gradient(top_class_channel, conv_outputs)
  
          # Pooled gradients: Take the mean of the gradients across the height and width of the feature map
          # This gives us a single gradient value for each filter in the convolutional layer.
          pooled_grads = np.mean(grads, axis=(0, 1, 2))
  
          # The conv_outputs represent the feature maps from the last convolutional layer.
          # We extract the first element because the batch size is 1.
          conv_outputs = conv_outputs[0]
  
          # Multiply each feature map by its corresponding pooled gradient.
          # This step weights the importance of each feature map based on its contribution to the class score.
          conv_outputs = np.dot(conv_outputs, pooled_grads[..., np.newaxis])
  
          # ReLU (Rectified Linear Unit) operation: This ensures that only the positive contributions
          # to the class score are considered. We set all negative values to zero.
          heatmap = np.maximum(conv_outputs, 0)
  
          # Normalize the heatmap values between 0 and 1 for easier visualization
          # This makes the brightest regions in the heatmap more apparent.
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


# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
# import cv2
# 
# # Load pre-trained ResNet50 model from Keras applications
# def load_model():
#     model = ResNet50(weights='imagenet')
#     model.trainable = False  # Freeze the model for inference
#     return model
# 
# # Function to preprocess image and make predictions
# def preprocess_and_predict(image_path, model, top_n=5):
#     # Load and preprocess the image
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     
#     # Make predictions
#     predictions = model.predict(img_array)
#     
#     # Decode predictions
#     decoded_predictions = decode_predictions(predictions, top=top_n)
#     results = []
#     
#     for i, pred in enumerate(decoded_predictions[0]):
#         label = pred[1]  # Human-readable label
#         confidence = pred[2]  # Confidence score
#         results.append({
#             'label': label,
#             'confidence': confidence,
#             'gradcam_img_path': f"www/gradcam_output_{i}.png"
#         })
#     
#     return results
# 
# # Function to generate Grad-CAM visualization
# def generate_gradcam(image_path, model, top_n=5):
#     # Preprocess the image and get predictions
#     results = preprocess_and_predict(image_path, model, top_n=top_n)
#     
#     # Get last convolutional layer in ResNet50
#     last_conv_layer = model.get_layer('conv5_block3_out')
# 
#     # Create a new model that maps the input to the activations of the last conv layer and predictions
#     grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
# 
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
# 
#     with tf.GradientTape() as tape:
#         # Forward pass through the model and compute gradients
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, np.argmax(predictions[0])]
# 
#         # Compute gradients with respect to the conv layer output
#         grads = tape.gradient(loss, conv_outputs)
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# 
#     conv_outputs = conv_outputs[0]
# 
#     # Apply Grad-CAM by weighting the conv layer outputs with the gradients
#     for i in range(conv_outputs.shape[-1]):
#         conv_outputs[:, :, i] *= pooled_grads[i]
# 
#     # Generate heatmap
#     heatmap = np.mean(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     heatmap = cv2.resize(heatmap, (224, 224))
# 
#     # Load the original image
#     img = cv2.imread(image_path)
# 
#     # Resize to match the original image size (if necessary)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# 
#     # Convert the heatmap to RGB
#     heatmap = np.uint8(255 * heatmap)
# 
#     # Apply heatmap to the original image
#     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 
#     # Superimpose the heatmap on the original image
#     superimposed_img = cv2.addWeighted(heatmap_img, 0.4, img, 0.6, 0)
# 
#     # Save the Grad-CAM output
#     gradcam_output_path = f"www/gradcam_output_{np.argmax(predictions[0])}.png"
#     cv2.imwrite(gradcam_output_path, superimposed_img)
# 
#     return results
