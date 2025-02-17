import os
from model_script import generate_gradcam, get_imagenet_label  # Import functions from your model_script

# Define the path to an example image
test_image_path = "/srv/shiny-server/SYE/dog.jpg"  # Replace with an actual image path

# Ensure the test image exists
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

# Test Grad-CAM functionality
try:
    # Load the model (assuming the model is globally available in model_script)
    from tensorflow.keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet')

    # Generate Grad-CAM for the test image
    results = generate_gradcam(
        img_path=test_image_path,
        model=model,
        top_n=5,
        confidence_threshold=0.5
    )

    print("Grad-CAM Results:")
    for result in results:
        print(f"Label: {result['label']}, Confidence: {result['confidence']:.2f}")
        print(f"Grad-CAM Image Path: {result['gradcam_img_path']}")

    print("\nTest Completed Successfully!")

except Exception as e:
    print("An error occurred during testing:")
    print(e)
