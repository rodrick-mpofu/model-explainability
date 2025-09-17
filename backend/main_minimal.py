"""
Minimal FastAPI backend for Model Explainability application.
This version can start without AI dependencies for testing purposes.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import tempfile
import uuid
from typing import List, Dict, Any

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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Model Explainability API is running!", "version": "2.0.0", "mode": "minimal"}

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            {"name": "vgg16", "display_name": "VGG16", "status": "available"},
            {"name": "resnet50", "display_name": "ResNet50", "status": "available"},
            {"name": "mobilenet_v2", "display_name": "MobileNetV2", "status": "available"},
            {"name": "efficientnetb0", "display_name": "EfficientNetB0", "status": "unavailable", "reason": "TensorFlow 2.20.0 compatibility issue"},
            {"name": "efficientnetb7", "display_name": "EfficientNetB7", "status": "unavailable", "reason": "TensorFlow 2.20.0 compatibility issue"}
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
    This is a mock implementation for testing the frontend.
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
        
        # Check if we should use real AI
        use_real_ai = os.environ.get("USE_REAL_AI", "false").lower() == "true"
        
        if use_real_ai:
            # Try to use real AI
            try:
                print(f"🤖 Attempting real AI analysis with {model_name} and {technique}")
                # Import and use real AI modules
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
                from explainability_api import explain_image
                
                results = explain_image(
                    img_path=upload_path,
                    model_name=model_name,
                    technique=technique,
                    top_n=top_n,
                    confidence_threshold=confidence_threshold,
                    output_dir="outputs"
                )
                
                # Process results and convert numpy types to Python types
                processed_results = []
                for result in results:
                    # Convert numpy types to Python types for JSON serialization
                    processed_result = {}
                    for key, value in result.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            processed_result[key] = value.item()
                        elif hasattr(value, 'tolist'):  # numpy array
                            processed_result[key] = value.tolist()
                        else:
                            processed_result[key] = value
                    
                    # Convert file paths to URLs
                    if 'gradcam_img_path' in processed_result:
                        filename = os.path.basename(processed_result['gradcam_img_path'])
                        processed_result['gradcam_img_url'] = f"/outputs/{filename}"
                    
                    processed_results.append(processed_result)
                
                mode = "real_ai"
                
            except Exception as e:
                print(f"❌ Real AI failed: {e}")
                print("🔄 Falling back to mock results")
                # Fall back to mock
                processed_results = [
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
                mode = "fallback_mock"
        else:
            # Use mock results
            processed_results = [
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
                "mode": mode
            },
            "note": f"Running in {mode} mode. {'Real AI models used!' if mode == 'real_ai' else 'Mock data for testing.'}"
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    use_real_ai = os.environ.get("USE_REAL_AI", "false").lower() == "true"
    mode = "hybrid" if use_real_ai else "minimal"
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "mode": mode,
        "ai_enabled": use_real_ai,
        "endpoints": {
            "analyze": "/analyze",
            "models": "/models", 
            "techniques": "/techniques"
        },
        "note": f"Running in {mode} mode. {'Real AI available via USE_REAL_AI=true' if use_real_ai else 'Mock data only. Set USE_REAL_AI=true to enable AI.'}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
