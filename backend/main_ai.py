"""
FastAPI backend for Model Explainability application with real AI functionality.
This version uses the actual Python AI modules for model predictions and explanations.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from typing import List, Dict, Any
import sys
import traceback

# Add the src/python directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

# Try to import AI modules, fall back to mock if not available
try:
    from explainability_api import explain_image
    AI_AVAILABLE = True
    print("✅ AI modules loaded successfully")
except ImportError as e:
    print(f"⚠️  AI modules not available: {e}")
    AI_AVAILABLE = False
    explain_image = None

app = FastAPI(
    title="Model Explainability API",
    description="API for generating model explanations using Grad-CAM and SHAP",
    version="2.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for file storage
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Mount static files for serving generated images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Model Explainability API is running!", 
        "version": "2.0.0",
        "ai_available": AI_AVAILABLE
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            {"name": "vgg16", "display_name": "VGG16"},
            {"name": "resnet50", "display_name": "ResNet50"},
            {"name": "mobilenet_v2", "display_name": "MobileNetV2"},
            {"name": "efficientnetb0", "display_name": "EfficientNetB0"},
            {"name": "efficientnetb7", "display_name": "EfficientNetB7"}
        ]
    }

@app.get("/techniques")
async def get_available_techniques():
    """Get list of available explanation techniques"""
    return {
        "techniques": [
            {"name": "gradcam", "display_name": "Grad-CAM"},
            {"name": "shap", "display_name": "SHAP"}
        ]
    }

def generate_mock_results(model_name, technique, confidence_threshold):
    """Generate mock results for testing"""
    mock_results = [
        {
            "label": "golden_retriever",
            "confidence": 0.95,
            "gradcam_img_url": "/outputs/mock_gradcam_1.png"
        },
        {
            "label": "labrador_retriever", 
            "confidence": 0.87,
            "gradcam_img_url": "/outputs/mock_gradcam_2.png"
        },
        {
            "label": "dog",
            "confidence": 0.78,
            "gradcam_img_url": "/outputs/mock_gradcam_3.png"
        }
    ]
    
    # Filter by confidence threshold
    filtered_results = [r for r in mock_results if r["confidence"] >= confidence_threshold]
    
    return filtered_results

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: str = Form("vgg16"),
    technique: str = Form("gradcam"),
    confidence_threshold: float = Form(0.5),
    top_n: int = Form(5)
):
    """
    Analyze an uploaded image and generate explanations.
    Uses real AI models if available, otherwise falls back to mock data.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create unique filename to avoid conflicts
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = os.path.join("uploads", unique_filename)
        
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate inputs
        if technique not in ["gradcam", "shap"]:
            raise HTTPException(status_code=400, detail="Invalid technique. Must be 'gradcam' or 'shap'")
        
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0 and 1")
        
        # Generate explanations
        if AI_AVAILABLE and explain_image:
            try:
                print(f"🤖 Using real AI models for {technique} analysis with {model_name}")
                
                # Use real AI modules
                results = explain_image(
                    img_path=upload_path,
                    model_name=model_name,
                    technique=technique,
                    top_n=top_n,
                    confidence_threshold=confidence_threshold,
                    output_dir="outputs"
                )
                
                # Process results to include web-accessible URLs
                processed_results = []
                for i, result in enumerate(results):
                    # Convert local file paths to web URLs
                    if 'gradcam_img_path' in result:
                        filename = os.path.basename(result['gradcam_img_path'])
                        result['gradcam_img_url'] = f"/outputs/{filename}"
                    
                    processed_results.append(result)
                
                mode = "real_ai"
                
            except Exception as e:
                print(f"❌ AI analysis failed: {e}")
                print(f"📝 Traceback: {traceback.format_exc()}")
                print("🔄 Falling back to mock results")
                
                # Fall back to mock results
                processed_results = generate_mock_results(model_name, technique, confidence_threshold)
                mode = "fallback_mock"
        else:
            print(f"🎭 Using mock results (AI not available)")
            processed_results = generate_mock_results(model_name, technique, confidence_threshold)
            mode = "mock"
        
        # Clean up uploaded file
        try:
            os.remove(upload_path)
        except:
            pass  # Ignore cleanup errors
        
        return {
            "success": True,
            "results": processed_results,
            "metadata": {
                "model_used": model_name,
                "technique_used": technique,
                "confidence_threshold": confidence_threshold,
                "total_predictions": len(processed_results),
                "mode": mode,
                "ai_available": AI_AVAILABLE
            }
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ai_available": AI_AVAILABLE,
        "mode": "real_ai" if AI_AVAILABLE else "mock",
        "endpoints": {
            "analyze": "/analyze",
            "models": "/models",
            "techniques": "/techniques"
        },
        "note": "Real AI models loaded" if AI_AVAILABLE else "Using mock data - AI models not available"
    }

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting Model Explainability API...")
    print(f"🤖 AI Available: {AI_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
