library(reticulate)
use_virtualenv("/srv/shiny-server/SYE/.venv", required = TRUE)
py_config()  # Print the Python configuration in the logs

server <- function(input, output, session) {
  
  observeEvent(input$submit, {
    req(input$file)
    
    # Save the uploaded file to www folder with a fixed name for consistency
    original_image_path <- file.path("www", "original_resized.png")
    file.copy(input$file$datapath, original_image_path, overwrite = TRUE)
    
    # Load the Python script with load_model and generate_gradcam functions
    py_run_file("/srv/shiny-server/SYE/model_script.py")  
    
    if (is.null(py$load_model) || is.null(py$generate_gradcam)) {
      showNotification("Error loading Python functions. Check script path or function definitions.", type = "error")
      return()
    }
    
    # Show progress bar while the model is running
    withProgress(message = 'Analyzing image...', value = 0, {
      
      incProgress(0.2, detail = "Loading model")
      
      # Load the model, layer, preprocessing function, and input size
      python_model <- py$load_model(input$model_choice)
      model <- python_model[[1]]
      last_conv_layer_name <- python_model[[2]]
      preprocess <- python_model[[3]]
      input_size <- python_model[[4]]
      
      incProgress(0.3, detail = "Running Grad-CAM")
      
      # Call generate_gradcam or generate_shap with model and preprocess arguments
      if(input$explainability_technique == "gradcam"){
      results <- py$generate_gradcam(input$file$datapath, model, preprocess, top_n=5, confidence_threshold=input$confidence, last_conv_layer_name=last_conv_layer_name, input_size=input_size)
      } else if(input$explainability_technique == "shap"){
      results <- py$generate_shap(input$file$datapath, model, preprocess, top_n=5, confidence_threshold=input$confidence, input_size=input_size)
      }

      if (length(results) == 0) {
        showNotification("No predictions found above the confidence threshold.", type = "error")
      } else {
        incProgress(0.6, detail = "Rendering outputs")
        
        # Update the original image every time a new file is uploaded
        output$uploaded_image <- renderImage({
          list(src = original_image_path, alt = "Uploaded Image", width = "400px", height = "400px")
        }, deleteFile = FALSE)
        
        # Dynamically render predictions and their Grad-CAM images in the Prediction Results tab
        output$predicted_results <- renderUI({
          lapply(1:length(results), function(i) {
            result <- results[[i]]
            
            # Prepare Grad-CAM image rendering using renderImage
            output[[paste0("gradcam_output_", i)]] <- renderImage({
              # Ensure the image file exists before serving it
              if (file.exists(result$gradcam_img_path)) {
                list(src = result$gradcam_img_path, height = "300px", width = "300px", alt = "Grad-CAM Output")
              } else {
                NULL  # Return NULL if the file doesn't exist
              }
            }, deleteFile = FALSE)
            
            # Dynamically create UI elements for each prediction
            tagList(
              fluidRow(
                column(6,
                       h4(paste("Predicted Category:", result$label)),
                       h5(paste("Confidence Score:", round(result$confidence * 100, 2), "%"))
                ),
                column(6,
                       h4("Grad-CAM Output:"),
                       # Output the image using imageOutput instead of img()
                       imageOutput(paste0("gradcam_output_", i), height = "300px", width = "300px")
                )
              ),
              hr()  # Add a horizontal line between predictions
            )
          })
        })
        
        incProgress(1, detail = "Complete!")
      }
    })
  })
}
