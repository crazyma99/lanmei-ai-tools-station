#!/bin/bash

# Configuration
PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/.venv"
BACKEND_LOG="$PROJECT_ROOT/backend.log"
FRONTEND_LOG="$PROJECT_ROOT/web/frontend.log"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
MODEL_FILE="$PROJECT_ROOT/models/bisenet/resnet18.onnx"

# Function to kill process using a port
kill_port() {
    local port=$1
    local pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)..."
        kill -9 $pid
    fi
}

# 0. Sync with Remote (Optional)
sync_code() {
    if [ -d ".git" ]; then
        echo ">>> Syncing with remote repository..."
        git pull origin main
    fi
}

# 1. Setup Backend Environment
setup_backend() {
    echo ">>> Setting up Backend..."
    
    # Check if model file exists
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Mandatory model file $MODEL_FILE not found."
        echo "Please ensure the model is placed in the models/bisenet/ directory."
        exit 1
    fi

    # Check if python3 is available
    if ! command -v python3 > /dev/null 2>&1; then
        echo "Error: python3 is not installed."
        exit 1
    fi

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi

    # Use venv pip/python directly for reliability
    VENV_PYTHON="$VENV_DIR/bin/python3"
    VENV_PIP="$VENV_DIR/bin/pip"

    # Upgrade pip
    $VENV_PIP install --upgrade pip

    # Install dependencies
    if [ -f "$REQUIREMENTS" ]; then
        echo "Installing/Verifying backend dependencies from $REQUIREMENTS..."
        # Use a more modern and PEP 503 compliant mirror to avoid HTML5 parsing issues
        $VENV_PIP install -r "$REQUIREMENTS" -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
        echo "Error: $REQUIREMENTS not found."
        exit 1
    fi
    
    # Verify installation
    if ! $VENV_PYTHON -c "import fastapi; import uvicorn; import cv2; import numpy; import onnxruntime" > /dev/null 2>&1; then
        echo "Error: Backend dependencies verification failed."
        exit 1
    fi
    echo "Backend dependencies verified."
}

# 2. Setup Frontend Environment
setup_frontend() {
    echo ">>> Setting up Frontend..."
    cd "$PROJECT_ROOT/web" || exit 1

    # Check if npm is available
    if ! command -v npm > /dev/null 2>&1; then
        echo "Error: npm is not installed."
        exit 1
    fi

    # Install dependencies if node_modules is missing
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    else
        echo "Frontend dependencies found."
    fi

    cd "$PROJECT_ROOT" || exit 1
}

# 3. Start Services
start_backend() {
    echo "Starting Backend (FastAPI)..."
    VENV_PYTHON="$VENV_DIR/bin/python3"
    nohup $VENV_PYTHON -m backend.main > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo "Backend started with PID $BACKEND_PID"
}

start_frontend() {
    echo "Starting Frontend (Vite)..."
    cd "$PROJECT_ROOT/web" || exit 1
    nohup npm run dev -- --host 0.0.0.0 > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo "Frontend started with PID $FRONTEND_PID"
    cd "$PROJECT_ROOT" || exit 1
}

# Main execution flow
echo "=================================================="
echo "Initializing Lanmei AI Tools Station..."
echo "=================================================="

# Cleanup
kill_port 8000
kill_port 5173

# Sync
sync_code

# Setup
setup_backend
setup_frontend

# Start
start_backend
start_frontend

echo "=================================================="
echo "Services started!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:5173"
echo "Starting daemon mode to monitor processes..."
echo "Press Ctrl+C to stop all services."
echo "=================================================="

# 4. Daemon / Guard Loop
# Use numeric signals for better compatibility with different shells (INT=2, TERM=15)
cleanup() {
    echo "Stopping services..."
    # Kill the processes if they are still running
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
    exit 0
}

# POSIX compliant trap: trap 'command' SIGNAL_NUMBER
trap 'cleanup' 2 15

while true; do
    # Check Backend
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "[$(date)] Backend (PID $BACKEND_PID) died. Restarting..."
        start_backend
    fi

    # Check Frontend
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "[$(date)] Frontend (PID $FRONTEND_PID) died. Restarting..."
        start_frontend
    fi

    sleep 5
done
