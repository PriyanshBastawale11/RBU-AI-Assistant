#!/usr/bin/env python
"""
RBU AI Assistant - Startup Script
This script initializes and starts the RBU AI Assistant chatbot.
"""

import os
import sys
import webbrowser
import time
import threading
import socket
from pathlib import Path

# Ensure we're in the correct directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

print("=" * 60)
print("RBU AI ASSISTANT - STARTUP")
print("=" * 60)
print("Initializing the chatbot. This may take a few moments...")
print("The application will download necessary NLP models on first run.")
print("=" * 60)

def check_port_available(port):
    """Check if a port is available for use."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # If result is not 0, the port is available
    except:
        return False

try:
    # Check for Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required!")
        sys.exit(1)
    
    # Check for virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("Warning: It's recommended to run this in a virtual environment.")
        # Automatically continue without asking for input
        response = 'y'
        print("Continuing anyway...")
        if response.lower() != 'y':
            print("Exiting. Please set up a virtual environment before running.")
            sys.exit(0)
    
    # Check for required files
    required_files = [
        "app.py",
        "chatbot.py",
        "requirements.txt",
        "data/college_data.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Try to import required modules
    try:
        import flask
        import gradio
        import nltk
        import sklearn
        import numpy as np
        import textblob
    except ImportError as e:
        print(f"Error: Missing required Python packages: {e}")
        print("Please install the requirements using: pip install -r requirements.txt")
        sys.exit(1)
    
    # Optional imports - we'll still run without these
    try:
        import torch
        import transformers
        import sentence_transformers
        has_advanced_nlp = True
        print("[OK] Advanced NLP models available (sentence-transformers)")
    except ImportError:
        has_advanced_nlp = False
        print("[INFO] Running in basic mode - sentence-transformers not available")
        print("  Only TF-IDF matching will be used (no semantic search)")

    # Check if ports are available
    flask_port = int(os.environ.get('PORT', 5000))
    gradio_port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    
    if not check_port_available(flask_port):
        print(f"Warning: Port {flask_port} is already in use. Flask might not start properly.")
    
    if not check_port_available(gradio_port):
        print(f"Warning: Port {gradio_port} is already in use. Gradio might not start properly.")
    
    # Import the application modules
    from app import app, create_gradio_interface
    import gradio as gr
    
    # Create the Gradio interface
    interface = create_gradio_interface()
    
    # Function to start Gradio in a separate thread
    def start_gradio():
        try:
            print(f"Starting Gradio interface on port {gradio_port}...")
            # Launch Gradio without blocking
            interface.launch(
                server_name="0.0.0.0", 
                server_port=gradio_port,
                share=False,
                prevent_thread_lock=True,
                show_error=True,
                quiet=False
            )
            print(f"[OK] Gradio server started on port {gradio_port}")
        except Exception as e:
            print(f"[ERROR] Error starting Gradio: {e}")
    
    # Start Gradio server in a non-daemon thread
    gradio_thread = threading.Thread(target=start_gradio)
    gradio_thread.daemon = False
    gradio_thread.start()
    
    # Allow time for Gradio to start
    time.sleep(3)
    
    # Open browser after Gradio is started
    def open_browser():
        time.sleep(2)
        print("Opening browser to RBU AI Assistant...")
        
        # Try to open main Flask interface
        try:
            webbrowser.open(f'http://localhost:{flask_port}/')
            print(f"[OK] Browser opened to http://localhost:{flask_port}/")
        except Exception as e:
            print(f"[INFO] Could not open browser automatically: {e}")
            print(f"Please manually navigate to: http://localhost:{flask_port}/")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the Flask app
    print(f"Starting Flask application on port {flask_port}...")
    app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)
    
except KeyboardInterrupt:
    print("\nShutting down RBU AI Assistant...")
    sys.exit(0)
except Exception as e:
    print(f"Error starting the application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 