# Model Explainability

A **Shiny-based web application** that enables users to upload images and analyze how deep learning models make predictions. The app provides **two interpretability techniques**:

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)** – Highlights important regions in an image that contributed to a model's prediction.
2. **SHAP (SHapley Additive exPlanations)** – A game-theoretic approach that assigns importance values to each pixel.

This project focuses on **XAI (Explainable AI)** techniques for deep learning with a clean, scalable architecture.

## 🔥 Features

- ✅ **Upload and Analyze Images** – Users can upload an image and visualize explanations
- ✅ **Multiple Model Support** – Select from models like **VGG16, ResNet50, MobileNetV2, and EfficientNet**
- ✅ **Grad-CAM Visualization** – See heatmaps overlaying important regions
- ✅ **SHAP Interpretability** – Generate pixel-based importance explanations
- ✅ **Confidence Threshold Slider** – Customize the minimum prediction confidence for analysis
- ✅ **Interactive UI** – Built with **Shiny and R**, integrating Python via `reticulate`
- ✅ **Modular Architecture** – Clean separation of concerns with scalable folder structure

## 📁 Project Structure

```
model-explainability/
├── src/
│   ├── python/                    # Python explainability modules
│   │   ├── models/                # Model loading and management
│   │   ├── explainability/        # Grad-CAM and SHAP implementations
│   │   ├── utils/                 # Utility functions
│   │   └── explainability_api.py  # Main API interface
│   └── shiny/                     # R Shiny application components
│       ├── ui.R                   # User interface
│       ├── server.R               # Server logic
│       └── modules/               # Modular Shiny components
├── tests/
│   ├── python/                    # Python unit tests
│   └── r/                         # R tests
├── assets/
│   ├── images/                    # Sample images and test data
│   └── docs/                      # Documentation and papers
├── config/                        # Configuration files
├── outputs/                       # Generated explanations (gitignored)
├── requirements.txt               # Python dependencies
├── app.R                         # Main Shiny app entry point
└── README.md
```

## 🛠️ Installation & Setup

### **Prerequisites**
- **R (>= 4.0)**
- **Python (>= 3.8)**
- **Shiny Server** (for deployment)

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/rodrick-mpofu/model-explainability.git
cd model-explainability
```

### **2️⃣ Install Python Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Install R Dependencies**
```r
# Install required R packages
install.packages(c("shiny", "shinydashboard", "shinythemes", 
                   "shinycssloaders", "reticulate", "png"))
```

### **4️⃣ Configure Python Environment**
```r
# In R, configure reticulate to use your Python environment
library(reticulate)
use_python("/path/to/your/python")  # or use_virtualenv()
```

### **5️⃣ Run the Application**
```r
# Run the Shiny app
shiny::runApp("app.R")
```

## 🎯 How It Works

1. **Upload an Image** 📷
2. **Select Model & Technique** (Grad-CAM or SHAP) 🏗️
3. **Adjust Confidence Threshold** 🎚️
4. **View Model Explanation** 🔥 (Heatmaps for Grad-CAM / SHAP values for SHAP)
5. **Interpret Results** ✅

## 🐍 Python API Usage

You can also use the Python API directly:

```python
from src.python.explainability_api import explain_image

# Generate Grad-CAM explanation
results = explain_image(
    img_path="assets/images/dog.jpg",
    model_name="vgg16",
    technique="gradcam",
    confidence_threshold=0.5
)

# Generate SHAP explanation  
results = explain_image(
    img_path="assets/images/dog.jpg",
    model_name="resnet50", 
    technique="shap",
    confidence_threshold=0.3
)
```

## 🧪 Testing

Run the test suite:

```bash
# Python tests
python -m pytest tests/python/

# R tests (if available)
Rscript -e "testthat::test_dir('tests/r')"
```

## 📚 Architecture Benefits

The new modular structure provides:

- **Separation of Concerns**: Clear separation between UI, business logic, and utilities
- **Scalability**: Easy to add new models and explanation techniques
- **Maintainability**: Well-organized code with proper imports and dependencies
- **Testability**: Dedicated test structure for both Python and R components
- **Reusability**: Python modules can be used independently of the Shiny app
- **Framework Conventions**: Follows best practices for both Python and R/Shiny

## 📬 Contact & Contributions

- 👤 **Rodrick Mpofu**
- 📧 **Email**: rodrickmpofu@gmail.com
- 🔗 **GitHub**: [@rodrick-mpofu](https://github.com/rodrick-mpofu)
- 🔗 **LinkedIn**: [Rodrick Mpofu](https://www.linkedin.com/in/rodrick-mpofu/)

## 📄 License

This project is available under the MIT License. See LICENSE file for details.







