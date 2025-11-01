@echo off
echo ==========================================
echo RBU AI ASSISTANT - RESTART SCRIPT
echo ==========================================

echo Stopping any existing Python processes related to RBU AI Assistant...
taskkill /F /FI "WINDOWTITLE eq RBU AI Assistant" /IM python.exe 2>NUL
taskkill /F /FI "WINDOWTITLE eq *Gradio*" /IM python.exe 2>NUL

echo Checking for existing processes on port 7860...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do (
    echo Killing process %%a using port 7860...
    taskkill /F /PID %%a 2>NUL
)

echo Checking for existing processes on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    echo Killing process %%a using port 5000...
    taskkill /F /PID %%a 2>NUL
)

echo Starting RBU AI Assistant...
cd %~dp0
python start.py

pause 