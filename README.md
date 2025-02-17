# Model Explainability App

This is a **Shiny-based web application** that enables users to upload an image and analyze how deep learning models make predictions. The app provides **two interpretability techniques**:
1. **Grad-CAM (Gradient-weighted Class Activation Mapping)** – Highlights important regions in an image that contributed to a model's prediction.
2. **SHAP (SHapley Additive exPlanations)** – A game-theoretic approach that assigns importance values to each pixel.

This project is part of a **Senior Year Experience (SYE)**, focusing on **XAI (Explainable AI)** techniques for deep learning.

---

## 🔥 Features
- ✅ **Upload and Analyze Images** – Users can upload an image and visualize explanations.
- ✅ **Multiple Model Support** – Select from models like **VGG16, ResNet50, MobileNetV2, and EfficientNet**.
- ✅ **Grad-CAM Visualization** – See heatmaps overlaying important regions.
- ✅ **SHAP Interpretability** – Generate pixel-based importance explanations.
- ✅ **Confidence Threshold Slider** – Customize the minimum prediction confidence for analysis.
- ✅ **Interactive UI** – Built with **Shiny and R**, integrating Python via `reticulate`.

---

## 🛠️ Installation & Setup

### **Prerequisites**
- **R (>= 4.0)**
- **Python (>= 3.8)**
- **Shiny Server**
- **AWS EC2 Instance (if deploying remotely)**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/rodrick-mpofu/model-explainability.git
cd model-explainability
```

### 🎯 How It Works
- Upload an Image 📷
- Select Model & Technique (Grad-CAM or SHAP) 🏗️
- Adjust Confidence Threshold 🎚️
- View Model Explanation 🔥 (Heatmaps for Grad-CAM / SHAP values for SHAP)
- Interpret Results ✅

### 📬 Contact & Contributions
- 👤 Rodrick Mpofu
- 📧 Email: rodrickmpofu@gmail.com
- 🔗 GitHub: @rodrick-mpofu
- 🔗 LinkedIn: [Rodrick Mpofu](https://www.linkedin.com/in/rodrick-mpofu/)







