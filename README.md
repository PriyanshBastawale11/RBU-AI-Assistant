# RBU AI Assistant

A comprehensive AI-powered chatbot for Regional Benchmark University (RBU) that provides accurate information about university programs, admissions, facilities, and more using a hybrid NLP approach.

## Features

- **Hybrid NLP Processing**: Combines TF-IDF with cosine similarity and DistilBERT-based semantic search for optimal response accuracy
- **Voice Input Support**: Speak queries directly to the chatbot
- **FAQ Suggestions**: Quick access to commonly asked questions
- **User Feedback Collection**: Rate responses and provide comments for continuous improvement
- **Spell Checking**: Automatically corrects spelling errors in user queries
- **Performance Metrics**: Real-time tracking of accuracy and response time
- **Orange-themed UI**: Clean, responsive design with the university's orange branding
- **Dual Interface**: Flask API for integration and Gradio UI for direct interaction

## Architecture

The chatbot uses a two-step query processing approach:

1. **Keyword-based Matching**: First attempts to find matches using TF-IDF vectorization and cosine similarity
2. **Semantic Matching**: If keyword matching doesn't yield strong results, uses a pretrained DistilBERT model to find semantically similar responses
3. **Fallback Mechanism**: Provides default responses if no match is found

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Gradio
- **NLP Processing**: NLTK, scikit-learn, Sentence Transformers, Hugging Face Transformers
- **Additional Libraries**: TextBlob for spell checking, pandas for data manipulation

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation Steps

1. Clone the repository:
   ```
   git clone <repository-url>
   cd RBU_AI_Assistant
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   # On Windows
   restart.bat
   
   # On macOS/Linux
   bash restart.sh
   ```

5. Access the application:
   - Main web interface: http://localhost:5000/
   - Gradio chat interface: http://localhost:7860/

## Project Structure

```
RBU_AI_Assistant/
├── app.py                 # Main Flask application
├── chatbot.py             # Core chatbot logic with NLP components
├── requirements.txt       # Python dependencies
├── start.py               # Startup script
├── restart.bat            # Windows restart script
├── restart.sh             # Unix restart script
├── data/
│   └── college_data.json  # University information dataset
├── models/                # For storing NLP models
├── static/
│   ├── css/
│   │   └── style.css      # CSS styling
│   ├── js/
│   │   └── script.js      # Frontend JavaScript
│   └── img/               # Images for the UI
└── templates/
    └── index.html         # Main HTML template
    └── gradio_redirect.html  # Redirect template for Gradio
```

## Usage

### Chatbot Interface

1. Type your query in the text box or use the voice input feature
2. View the chatbot's response
3. Rate the response and provide feedback if desired
4. Use suggested FAQs for quick access to common information

### API Endpoints

- `POST /api/chat`: Submit a query to the chatbot
- `POST /api/feedback`: Submit feedback on a response
- `GET /api/metrics`: Get chatbot performance metrics
- `GET /api/suggestions`: Get FAQ suggestions
- `POST /api/reset-metrics`: Reset performance metrics

## Customization

### Adding New Data

To add new data to the chatbot:

1. Edit the `data/college_data.json` file
2. Add new FAQ entries with questions and answers
3. Restart the application to update the knowledge base

### Modifying the UI

- Update `static/css/style.css` to change the styling
- Edit `templates/index.html` to modify the web interface structure
- Customize the Gradio interface in `app.py` under the `create_gradio_interface` function

## Performance Evaluation

The chatbot tracks several performance metrics:

- Total number of queries processed
- Percentage of queries matched via TF-IDF
- Percentage of queries matched via semantic search
- Percentage of queries using fallback responses
- Average response time

View these metrics in the Chatbot Performance section of the Gradio interface or via the `/api/metrics` endpoint.

## Contributing

Contributions to improve the chatbot are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The chatbot was developed as a project for RamdeoBaba University
  
- Uses open-source NLP models from the Hugging Face Transformers library

- Gradio for the interactive chat interface 
