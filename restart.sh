#!/bin/bash

echo "=========================================="
echo "RBU AI ASSISTANT - RESTART SCRIPT"
echo "=========================================="

# Check if running on Windows or Linux/Mac
SYSTEM=$(uname -s)

if [[ "$SYSTEM" == "MINGW"* ]] || [[ "$SYSTEM" == "MSYS"* ]] || [[ "$SYSTEM" == "CYGWIN"* ]]; then
    # Windows
    echo "Detected Windows system..."
    
    echo "Stopping any existing Gradio processes..."
    taskkill /F /IM python.exe /FI "WINDOWTITLE eq Gradio" 2>/dev/null || true
    
    echo "Checking for existing processes on port 7860..."
    for /f "tokens=5" %a in ('netstat -ano ^| findstr :7860') do (
        echo "Killing process %a using port 7860..."
        taskkill /F /PID %a 2>/dev/null || true
    )
    
    echo "Checking for existing processes on port 5000..."
    for /f "tokens=5" %a in ('netstat -ano ^| findstr :5000') do (
        echo "Killing process %a using port 5000..."
        taskkill /F /PID %a 2>/dev/null || true
    )
    
    echo "Starting RBU AI Assistant..."
    cd "$(dirname "$0")" && python start.py
else
    # Linux/Mac
    echo "Detected UNIX-like system..."
    
    echo "Checking for existing processes on port 7860..."
    PIDS_7860=$(lsof -ti:7860)
    if [ ! -z "$PIDS_7860" ]; then
        echo "Killing processes using port 7860: $PIDS_7860"
        kill -9 $PIDS_7860 2>/dev/null || true
    else
        echo "No processes found using port 7860"
    fi
    
    echo "Checking for existing processes on port 5000..."
    PIDS_5000=$(lsof -ti:5000)
    if [ ! -z "$PIDS_5000" ]; then
        echo "Killing processes using port 5000: $PIDS_5000"
        kill -9 $PIDS_5000 2>/dev/null || true
    else
        echo "No processes found using port 5000"
    fi
    
    echo "Starting RBU AI Assistant..."
    cd "$(dirname "$0")" && python start.py
fi 