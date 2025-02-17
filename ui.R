library(shiny)
library(shinydashboard)
library(reticulate)
library(png)
library(shinythemes)
library(shinycssloaders)

# Define UI
ui <- fluidPage(
  titlePanel("Object Recognition App"),
  
  sidebarLayout(
    sidebarPanel(
      # Upload input
      fileInput("file", "Upload an Image", accept = c("image/png", "image/jpeg")),

      # Choose Technique
      selectInput("explainability_technique", "Explanation Technique", 
		choices = c("Grad-CAM" = "gradcam", "SHAP" = "shap"), 
		selected = "gradcam"),      
      
      # Add a model selection input to the UI
      selectInput("model_choice", "Choose a Model:",
                  choices = list("VGG16" = "vgg16", 
                                 "ResNet50" = "resnet50", 
                                 "MobileNetV2" = "mobilenet_v2",
                                 "EfficientNetB0" = "efficientnetb0",
                                 "EfficientNetB7" = "efficientnetb7")),
      
      # Confidence threshold slider
      sliderInput("confidence", "Confidence Threshold", min = 0, max = 1, value = 0.5, step=0.05),
      # Action button for analysis
      actionButton("submit", "Analyze Image", class = "btn-primary")
    ),
    
    mainPanel(
      # Use tabsetPanel to create two tabs: one for Original Image, one for Prediction Results
      tabsetPanel(
        # First tab for the original image
        tabPanel("Original Image", 
                 wellPanel(
                   h4("Original Image:"),
                   # Add spinner for loading status
                   withSpinner(imageOutput("uploaded_image", height = "400px", width = "400px"))
                 )
        ),
        
        # Second tab for prediction results
        tabPanel("Prediction Results", 
                 wellPanel(
                   h3("Prediction Results"),
                   # Multiple predictions and Grad-CAM outputs
                   uiOutput("predicted_results")
                 )
        ),
        
        # Add new "How to Use" tab
        tabPanel(
          "How to Use",
          wellPanel(
            h3("Instructions: How to Use the App"),
            p("This application allows users to upload an image and get predictions of the object's category using a pre-trained neural network."),
            
            h4("Steps to Use the Application:"),
            tags$ol(
              tags$li("Upload an image by clicking on the 'Browse' button in the 'Original Image' tab."),
              tags$li("Adjust the confidence threshold using the slider. This allows you to filter out predictions that are below a certain confidence score."),
              tags$li("Click on the 'Analyze Image' button to start the prediction process."),
              tags$li("The results will be displayed in the 'Prediction Results' tab, along with Grad-CAM visualizations that highlight which parts of the image contributed to the prediction.")
            ),
            
            h4("Notes:"),
            tags$ul(
              tags$li("The model used for this application is VGG16, pre-trained on the ImageNet dataset."),
              tags$li("The confidence threshold slider helps in refining the displayed predictions based on confidence scores."),
              tags$li("Ensure your image is in .png or .jpeg format before uploading.")
            )
          )
        ),
        
        # How Grad-CAM Works Tab
        tabPanel(
          "How Grad-CAM Works",
          mainPanel(
            withMathJax(),  # Enables MathJax for rendering LaTeX math
            h3("Understanding Grad-CAM (Gradient-weighted Class Activation Mapping)"),
            
            p("Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used to visualize and interpret the decisions made by convolutional neural networks (CNNs), 
               particularly in image classification tasks. While CNNs are highly effective in extracting features from images and making predictions, they are often viewed as 'black boxes' 
               due to the difficulty in understanding what parts of an image contributed to a specific prediction. Grad-CAM addresses this challenge by generating a visual heatmap 
               that highlights the regions in an image that most influenced the network's prediction."),
            
            h4("Key Steps of Grad-CAM:"),
            tags$ol(
              tags$li("Input Image and Model Prediction: The process starts by feeding an input image through a CNN model (e.g., VGG16 or ResNet), 
                       which extracts features and outputs a prediction across different classes. For a chosen class, Grad-CAM seeks to identify 
                       which parts of the image contributed most to the score of that class."),
              
              tags$li("Choosing the Last Convolutional Layer: Grad-CAM focuses on the feature maps in the last convolutional layer of the CNN. 
                       This layer retains spatial information that is essential for visualizing 'where' certain features were detected in the image."),
              
              tags$li("Computing Gradients: Grad-CAM calculates the gradient of the class score with respect to each feature map in the last convolutional layer. 
                       Mathematically, this is represented as the gradient, $$ \\frac{\\partial y_c}{\\partial A_k} $$, where $$ y_c $$ is the score of the class of interest 
                       and $$ A_k $$ is the activation of the $$ k $$-th feature map in the selected layer. This gradient highlights the sensitivity of the prediction to changes in each feature map."),
              
              tags$li("Global Average Pooling of Gradients: The gradients for each feature map are averaged across all spatial locations to get a single importance weight for each map. 
                       This weight is computed as: $$ \\alpha_k^c = \\frac{1}{Z} \\sum_{i,j} \\frac{\\partial y_c}{\\partial A_{k}^{ij}} $$, 
                       where $$ Z $$ is the total number of spatial locations, and $$ \\alpha_k^c $$ represents the importance of each feature map $$ A_k $$ for the class $$ c $$."),
              
              tags$li("Generating the Heatmap: Grad-CAM produces a weighted combination of feature maps by multiplying each feature map $$ A_k $$ by its corresponding weight $$ \\alpha_k^c $$. 
                       The heatmap $$ L^c $$ is then calculated as: $$ L^c = \\text{ReLU} \\left( \\sum_k \\alpha_k^c A_k \\right) $$, 
                       where $$ \\text{ReLU} $$ sets negative values to zero, focusing on positive contributions to the prediction. This step effectively highlights areas in the image that most support the model’s classification decision."),
              
              tags$li("Overlaying the Heatmap: The heatmap is resized to match the original image and is superimposed to highlight areas that most influenced the model's decision. 
                       Brighter regions indicate the model's focus, helping users interpret the relevance of different parts of the image.")
            ),
            
            h4("Why Grad-CAM is Important:"),
            tags$ul(
              tags$li("Model Interpretability: Grad-CAM provides insights into the decision-making process of neural networks, making the model’s reasoning more transparent."),
              tags$li("Debugging Models: By showing where the model focuses, Grad-CAM can reveal if the model is relying on irrelevant or misleading parts of the image, 
                       helping identify issues in the model's understanding."),
              tags$li("Trust and Transparency: This technique builds trust in model predictions by explaining how the model arrived at its decision, which is especially useful in critical applications.")
            ),
            
            h4("Further Reading"),
            p("For a comprehensive explanation of Grad-CAM, you can refer to the original research paper:"),
            tags$a(href = "https://arxiv.org/abs/1610.02391", "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization")
          )
        )
      )
    )
  )
)

