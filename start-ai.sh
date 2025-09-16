#!/bin/bash

# Model Explainability - AI Mode Startup Script
echo "🚀 Starting Model Explainability with Real AI..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "backend/main_minimal.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to start backend with AI enabled
start_ai_backend() {
    echo "🤖 Starting backend with real AI enabled..."
    cd backend
    source venv/Scripts/activate
    export USE_REAL_AI=true
    python main_minimal.py
}

# Function to start backend in mock mode
start_mock_backend() {
    echo "🎭 Starting backend in mock mode..."
    cd backend
    source venv/Scripts/activate
    export USE_REAL_AI=false
    python main_minimal.py
}

# Function to start frontend
start_frontend() {
    echo "⚛️  Starting React frontend..."
    cd frontend
    npm run dev
}

# Main menu
echo "Choose your startup mode:"
echo "1) Real AI Mode (🤖 Uses actual TensorFlow models)"
echo "2) Mock Mode (🎭 Uses mock data for testing)"
echo "3) Frontend Only (⚛️  Just start the React app)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🤖 Starting in Real AI Mode..."
        start_ai_backend
        ;;
    2)
        echo "🎭 Starting in Mock Mode..."
        start_mock_backend
        ;;
    3)
        echo "⚛️  Starting Frontend Only..."
        start_frontend
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
