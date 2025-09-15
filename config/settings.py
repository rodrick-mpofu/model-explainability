# Configuration for model explainability application

# Default model settings
DEFAULT_MODEL = "vgg16"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_TOP_N = 3

# Supported models
SUPPORTED_MODELS = [
    "vgg16",
    "resnet50", 
    "mobilenet_v2",
    "efficientnetb0",
    "efficientnetb7"
]

# Supported explanation techniques
SUPPORTED_TECHNIQUES = [
    "gradcam",
    "shap"
]

# File paths
DEFAULT_OUTPUT_DIR = "outputs"
ASSETS_DIR = "assets"
SHINY_WWW_DIR = "www"

# Image processing settings
MAX_IMAGE_SIZE = (1024, 1024)  # Maximum image dimensions
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

# Grad-CAM settings
GRADCAM_OVERLAY_ALPHA = 0.6
GRADCAM_OVERLAY_BETA = 0.4

# SHAP settings
SHAP_DEBUG_LOG = "shap_debug.log"