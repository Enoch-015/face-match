echo "🚀 Setting up LiveKit Face Recognition System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Remove existing venv if it has issues
if [ -d "venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade core tools first
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install system dependencies for face_recognition (if on Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "📚 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-dev
fi

# Install packages one by one to avoid conflicts
echo "🐍 Installing Python packages..."

# Install numpy first (required by many packages)
pip install "numpy>=1.21.0,<2.0.0"

# Install OpenCV
pip install opencv-python

# Install dlib (required for face_recognition)
pip install dlib

# Install face_recognition
pip install face-recognition

pip install livekit.api

# Install LiveKit
pip install livekit

pip install python-multipart

# Install other dependencies
pip install Pillow

# Install FastAPI + Uvicorn for serving frontend & token API
pip install fastapi "uvicorn[standard]"

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installations
echo "🔍 Verifying installations..."
python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" 2>/dev/null || echo "❌ OpenCV failed"
python -c "import face_recognition; print('✅ face_recognition: OK')" 2>/dev/null || echo "❌ face_recognition failed"
python -c "import livekit; print('✅ LiveKit: OK')" 2>/dev/null || echo "❌ LiveKit failed"
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy failed"

echo ""
echo "📝 Next steps:"
echo "1. Your reference image is already set: WhatsApp Image 2025-08-12 at 07.38.40_dec32353.jpg"
echo "2. Update LiveKit credentials in .env file"
echo "3. Run: ./run.sh"
echo ""
