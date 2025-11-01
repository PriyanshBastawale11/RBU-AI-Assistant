import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect
import gradio as gr
from chatbot import RBUChatbot

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the chatbot
chatbot = RBUChatbot()

# Load feedback data file path
feedback_file = Path(__file__).parent / 'data' / 'feedback.json'

# Load or create feedback data
def load_feedback_data():
    if feedback_file.exists():
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return {"feedback": []}
    else:
        return {"feedback": []}

# Save feedback to the JSON file
def save_feedback(feedback_data):
    try:
        feedback_file.parent.mkdir(exist_ok=True)
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False

# Store user feedback
def add_feedback(query, response, rating, comment=""):
    feedback_data = load_feedback_data()
    feedback_data["feedback"].append({
        "query": query,
        "response": response,
        "rating": rating,
        "comment": comment,
        "source": response.get("source", "unknown")
    })
    return save_feedback(feedback_data)

# Define Gradio chat interface function
def gradio_chat(message, history):
    """Process user message and format response correctly for Gradio chatbot.
    
    Args:
        message (str): User's input message
        history (list): Chat history as a list of [user_msg, bot_msg] pairs
        
    Returns:
        list: Updated chat history with new message and response
    """
    # Process the user query
    response_data = chatbot.process_query(message)
    
    # Get the text response from the data
    bot_response = response_data['response']
    
    # Format message and response for Gradio Chatbot
    # Gradio chatbot expects a list of [user_msg, bot_msg] pairs
    history = history or []
    history.append([message, bot_response])
    
    return history

# Define speech-to-text function for voice input
def speech_to_text(audio):
    try:
        # With updated Gradio, we get the transcription directly from Whisper
        if audio is not None:
            # For numpy array input, this will be the transcribed text
            return audio["text"] if isinstance(audio, dict) and "text" in audio else audio
        return ""
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}")
        return ""

# Define function to get FAQ suggestions
def get_suggestions():
    suggestions = chatbot.get_faq_suggestions(5)
    return suggestions

# Gradio Interface setup
def create_gradio_interface():
    # Create theme with orange primary color
    theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="orange",
        neutral_hue="gray"
    )
    
    # Create Gradio chat interface
    with gr.Blocks() as interface:
        gr.Markdown("# RBU AI Assistant")
        gr.Markdown("Ask me anything about RBU - programs, admissions, fees, facilities, or any other university related queries!")
        
        # Chat interface
        chatbot_component = gr.Chatbot(
            label="Chat",
            show_label=True,
            height=400
        )
        
        # Input components
        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    placeholder="Type your question here...",
                    label="Message",
                    show_label=False
                )
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",  # Changed from "text" to "filepath"
                    label="Voice Input"
                )
        
        # Buttons
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        # FAQ Suggestions
        with gr.Accordion("Frequently Asked Questions", open=True):
            suggestion_btns = []
            for i in range(5):
                suggestion_btns.append(gr.Button("", visible=False))
        
        # Response Rating
        with gr.Accordion("Rate Response", open=False):
            with gr.Row():
                rating = gr.Radio(
                    ["👍 Helpful", "👎 Not Helpful"],
                    label="Was the response helpful?",
                    interactive=True
                )
                feedback_text = gr.Textbox(
                    placeholder="Additional feedback (optional)",
                    label="Feedback"
                )
            submit_feedback = gr.Button("Submit Feedback")
        
        # Performance Metrics
        with gr.Accordion("Chatbot Performance", open=False):
            refresh_metrics = gr.Button("Refresh Metrics")
            metrics_output = gr.JSON(chatbot.get_performance_metrics())
        
        # Set up callbacks
        submit_btn.click(
            gradio_chat,
            inputs=[msg, chatbot_component],
            outputs=[chatbot_component],
            queue=False
        ).then(
            lambda: "",
            None,
            [msg],
            queue=False
        )
        
        msg.submit(
            gradio_chat,
            inputs=[msg, chatbot_component],
            outputs=[chatbot_component],
            queue=False
        ).then(
            lambda: "",
            None,
            [msg],
            queue=False
        )
        
        audio_input.change(
            speech_to_text,
            inputs=[audio_input],
            outputs=[msg],
            queue=False
        )
        
        clear_btn.click(
            lambda: None,
            None,
            chatbot_component,
            queue=False
        )
        
        # Update FAQ suggestions
        suggestion_updater = gr.Button("Update Suggestions", visible=False)
        
        def update_suggestions():
            suggestions = get_suggestions()
            outputs = []
            for i, suggestion in enumerate(suggestions):
                if i < len(suggestion_btns):
                    outputs.append(gr.update(value=suggestion, visible=True))
                
            # If fewer suggestions than buttons, hide the extra buttons
            for i in range(len(suggestions), len(suggestion_btns)):
                outputs.append(gr.update(value="", visible=False))
            
            return outputs
        
        suggestion_updater.click(
            update_suggestions,
            inputs=[],
            outputs=suggestion_btns
        )
        
        # Initialize suggestions on load
        interface.load(
            update_suggestions,
            inputs=[],
            outputs=suggestion_btns
        )
        
        # Make the suggestion buttons fill the message box
        for btn in suggestion_btns:
            btn.click(
                lambda suggestion: suggestion,
                inputs=[btn],
                outputs=[msg]
            )
        
        # Handle rating submission
        submit_feedback.click(
            lambda x, y, z: "Feedback submitted. Thank you!",
            inputs=[rating, feedback_text, chatbot_component],
            outputs=[feedback_text]
        )
        
        # Refresh metrics
        refresh_metrics.click(
            lambda: chatbot.get_performance_metrics(),
            inputs=[],
            outputs=[metrics_output]
        )
    
    return interface

# Initialize Flask app
app = Flask(__name__)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gradio')
def gradio_redirect():
    # Direct redirect to port 7860 instead of using a template
    gradio_url = f"http://{request.host.split(':')[0]}:7860"
    return redirect(gradio_url)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Process the query
    response = chatbot.process_query(query)
    
    return jsonify(response)

@app.route('/api/feedback', methods=['POST'])
def feedback_api():
    data = request.json
    query = data.get('query', '')
    response = data.get('response', {})
    rating = data.get('rating', 0)
    comment = data.get('comment', '')
    
    success = add_feedback(query, response, rating, comment)
    
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to save feedback'}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics_api():
    metrics = chatbot.get_performance_metrics()
    return jsonify(metrics)

@app.route('/api/suggestions', methods=['GET'])
def suggestions_api():
    count = request.args.get('count', 5, type=int)
    suggestions = chatbot.get_faq_suggestions(count)
    return jsonify({'suggestions': suggestions})

@app.route('/api/reset-metrics', methods=['POST'])
def reset_metrics_api():
    chatbot.reset_metrics()
    return jsonify({'status': 'success'})

# Initialize and launch the application
if __name__ == "__main__":
    # Run the Flask app only (Gradio is started separately in start.py)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 