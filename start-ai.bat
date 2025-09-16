@echo off
echo 🚀 Starting Model Explainability with Real AI...
echo ==============================================

REM Check if we're in the right directory
if not exist "backend\main_minimal.py" (
    echo ❌ Please run this script from the project root directory
    pause
    exit /b 1
)

echo Choose your startup mode:
echo 1) Real AI Mode (🤖 Uses actual TensorFlow models)
echo 2) Mock Mode (🎭 Uses mock data for testing)
echo 3) Frontend Only (⚛️ Just start the React app)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo 🤖 Starting in Real AI Mode...
    cd backend
    call venv\Scripts\activate
    set USE_REAL_AI=true
    python main_minimal.py
) else if "%choice%"=="2" (
    echo 🎭 Starting in Mock Mode...
    cd backend
    call venv\Scripts\activate
    set USE_REAL_AI=false
    python main_minimal.py
) else if "%choice%"=="3" (
    echo ⚛️ Starting Frontend Only...
    cd frontend
    npm run dev
) else (
    echo ❌ Invalid choice. Please run the script again.
    pause
    exit /b 1
)

pause
