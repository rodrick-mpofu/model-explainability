import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from shiny import App, ui, render

# Define Grad-CAM generation function
def generate_gradcam(img_path, model, last_conv_layer_name="block5_conv3"):
    # Load and resize the original image to 400x400 pixels
    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, (400, 400))  # Resize the image to 400x400
    cv2.imwrite("original_resized.png", original_img_resized)  # Save resized original image
    
    # Preprocess image for model input
    img = np.expand_dims(original_img_resized, axis=0)
    img = preprocess_input(img)

    # Create a model that maps the input image to the activations of the last conv layer and the model's output
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Compute the gradient of the top predicted class with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    # Get the gradients of the top predicted class with respect to the output feature map of the conv layer
    grads = tape.gradient(top_class_channel, conv_outputs)

    # Pool the gradients over all the axes of the feature map (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weigh the output feature map by the pooled gradients
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap to only keep positive values
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap to [0, 1] range for visualization
    heatmap /= np.max(heatmap)

    # Resize heatmap to match the fixed image size (400x400)
    heatmap = cv2.resize(heatmap.numpy(), (400, 400))  # Resize heatmap to 400x400

    # Convert the heatmap to an RGB image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original resized image
    output_img = cv2.addWeighted(original_img_resized, 0.6, heatmap, 0.4, 0)

    # Save the output Grad-CAM image (resized)
    cv2.imwrite("gradcam_output.png", output_img)
    
    # Get the predicted label and confidence score
    predicted_class_index = int(tf.argmax(predictions[0]))  # Predicted class index
    confidence_score = float(tf.reduce_max(predictions[0]))  # Confidence score

    # ImageNet labels (use a local labels file or a downloaded one)
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    import json
    import requests
    response = requests.get(LABELS_URL)
    imagenet_labels = response.json()

    def get_imagenet_label(index):
        return imagenet_labels[str(index)][1]

    predicted_label = get_imagenet_label(predicted_class_index)

    return {
        "output_img": output_img,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score
    }

# Define UI for the app
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload an Image", accept=["image/png", "image/jpeg"]),
        ui.input_slider("confidence", "Confidence Threshold", min=0, max=1, value=0.5),
        ui.input_action_button("submit", "Analyze Image")
    ),
    ui.navset_tab(
        ui.nav_panel("Original Image", ui.output_image("uploaded_image")),
        ui.nav_panel("Grad-CAM Output", ui.output_image("gradcam_plot")),
        ui.nav_panel("Prediction Results",
            ui.h4("Predicted Category:"),
            ui.output_text("predicted_label"),
            ui.h4("Confidence Score:"),
            ui.output_text("confidence_score")
        )
    )
)

# Define server logic
def server(input, output, session):
    
    @output
    @render.image
    def uploaded_image():
        return {"src": "original_resized.png", "alt": "Uploaded Image"}
    
    @output
    @render.image
    def gradcam_plot():
        return {"src": "gradcam_output.png", "alt": "Grad-CAM Output"}

    @output
    @render.text
    def predicted_label():
        return session.results["predicted_label"]
    
    @output
    @render.text
    def confidence_score():
        return f"{session.results['confidence_score'] * 100:.2f}%"
    
    @input.submit
    async def on_submit():
        # Wait for file upload
        file_info = input.file()
        if file_info is None:
            return
        
        # Save uploaded image to disk
        img_path = file_info["datapath"]
        
        # Call Grad-CAM function
        model = tf.keras.applications.VGG16(weights="imagenet")  # Use a pre-trained model
        session.results = generate_gradcam(img_path, model)

# Run the app
app = App(app_ui, server)

app

