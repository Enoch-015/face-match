#!/bin/bash

echo "🚀 Starting LiveKit Face Recognition System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if reference image exists (using your actual filename)
REFERENCE_IMAGE="/workspaces/face-match/reference/WhatsApp Image 2025-08-12 at 07.38.40_dec32353.jpg"
if [ ! -f "$REFERENCE_IMAGE" ]; then
    echo "❌ Reference image not found at: $REFERENCE_IMAGE"
    echo "Please check the file path and ensure the image exists"
    exit 1
fi

echo "✅ Found reference image: $REFERENCE_IMAGE"

# Load environment variables safely (avoiding issues with spaces in filenames)
if [ -f ".env" ]; then
    # Use a safer method to load environment variables
    while IFS= read -r line; do
        # Skip comments and empty lines
        if [[ $line =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then
            continue
        fi
        # Export the variable
        export "$line"
    done < .env
fi

# Set the reference image path as an environment variable
export REFERENCE_IMAGE="$REFERENCE_IMAGE"

echo "🎯 Starting face recognition with image: $(basename "$REFERENCE_IMAGE")"

# Start FastAPI server in the background
echo "🌐 Starting API server (FastAPI + Uvicorn) on http://0.0.0.0:8000 ..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Trap to clean up background server on exit
cleanup() {
    echo "\n🧹 Shutting down services..."
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        wait $API_PID 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# Run the main application (blocking)
python main.py