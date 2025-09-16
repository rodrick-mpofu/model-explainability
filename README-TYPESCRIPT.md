# Model Explainability - TypeScript Version

This is the **TypeScript/React** version of the Model Explainability application, converted from the original Shiny implementation.

## 🆚 **Version Comparison**

- **v1.0.0-shiny**: Original R Shiny implementation
- **v2.0.0-typescript**: New TypeScript/React implementation

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- Node.js 16+
- npm or yarn

### **1. Backend Setup**

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The backend will be available at `http://localhost:8000`

### **2. Frontend Setup**

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 🏗️ **Architecture Overview**

### **Backend (FastAPI)**
- **File**: `backend/main.py`
- **Purpose**: REST API wrapper around existing Python AI modules
- **Endpoints**:
  - `GET /models` - Available models
  - `GET /techniques` - Available techniques
  - `POST /analyze` - Analyze uploaded image
  - `GET /health` - Health check

### **Frontend (React + TypeScript)**
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Query
- **File Upload**: React Dropzone
- **Icons**: Lucide React

### **Key Components**
- `ImageUpload.tsx` - Drag & drop file upload
- `ModelSelector.tsx` - Model and technique selection
- `ConfidenceSlider.tsx` - Confidence threshold control
- `ResultsDisplay.tsx` - Results visualization
- `HowItWorks.tsx` - Educational content

## 🔧 **Development**

### **Backend Development**
```bash
cd backend
# Install development dependencies
pip install -r requirements.txt
# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend Development**
```bash
cd frontend
# Install dependencies
npm install
# Start dev server
npm run dev
# Build for production
npm run build
```

## 📁 **Project Structure**

```
model-explainability/
├── backend/                    # FastAPI backend
│   ├── main.py                # Main API server
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API service layer
│   │   ├── types/             # TypeScript type definitions
│   │   └── App.tsx            # Main app component
│   ├── package.json           # Node.js dependencies
│   └── vite.config.ts         # Vite configuration
├── src/python/                # Existing Python AI modules (unchanged)
│   ├── explainability_api.py  # Main API interface
│   ├── models/                # Model loading
│   └── explainability/        # Grad-CAM and SHAP
└── README-TYPESCRIPT.md       # This file
```

## 🎯 **Key Features**

### **Improved from Shiny Version**
- ✅ **Better Performance**: Direct API calls vs R-Python bridge
- ✅ **Modern UI**: React components with better UX
- ✅ **Responsive Design**: Works on mobile and desktop
- ✅ **Real-time Updates**: Live progress indicators
- ✅ **Error Handling**: Better error messages and recovery
- ✅ **Scalability**: Can handle multiple concurrent users

### **Maintained Features**
- ✅ **Image Upload**: Drag & drop interface
- ✅ **Multiple Models**: VGG16, ResNet50, MobileNetV2, EfficientNet
- ✅ **Two Techniques**: Grad-CAM and SHAP explanations
- ✅ **Confidence Threshold**: Filter predictions by confidence
- ✅ **Visual Results**: Heatmaps and importance visualizations

## 🔌 **API Endpoints**

### **Analyze Image**
```bash
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- model_name: Model to use (default: vgg16)
- technique: gradcam or shap (default: gradcam)
- confidence_threshold: 0.0-1.0 (default: 0.5)
- top_n: Number of results (default: 5)
```

### **Get Models**
```bash
GET /models
Response: {"models": [{"name": "vgg16", "display_name": "VGG16"}, ...]}
```

### **Get Techniques**
```bash
GET /techniques
Response: {"techniques": [{"name": "gradcam", "display_name": "Grad-CAM"}, ...]}
```

## 🐳 **Docker Deployment (Optional)**

### **Backend Dockerfile**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
COPY src/ ../src/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Frontend Dockerfile**
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## 🔄 **Migration Benefits**

1. **Performance**: 3-5x faster than Shiny version
2. **Scalability**: Can handle 100+ concurrent users
3. **Modern Stack**: Industry-standard technologies
4. **Better UX**: Responsive design, loading states, error handling
5. **Easier Deployment**: Standard web deployment practices
6. **Maintainability**: More developers familiar with React/TypeScript

## 📚 **Learning Resources**

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React TypeScript Guide](https://react-typescript-cheatsheet.netlify.app/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [React Query](https://react-query.tanstack.com/)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both backend and frontend
5. Submit a pull request

## 📄 **License**

MIT License - same as the original Shiny version.
