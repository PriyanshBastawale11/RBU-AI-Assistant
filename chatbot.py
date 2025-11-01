import json
import random
import re
import nltk
import numpy as np
import pandas as pd
import os
from pathlib import Path
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import logging
from dotenv import load_dotenv
import spacy
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from gensim.models import KeyedVectors
import warnings
import requests
from sentence_transformers import SentenceTransformer  # Add this import

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download necessary NLTK packages 
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/names')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('names')

class RBUChatbot:
    def __init__(self, data_file='data/college_data.json'):
        """Initialize the RBU Chatbot with data and NLP models."""
        # Initialize entity recognition availability early
        self.ner_available = False  # Ensure this is defined before any dependent methods are called

        # Load data and prepare components
        self.data_file = Path(__file__).parent / data_file
        self.load_data()

        # Thresholds for determining response confidence:
        # tfidf_threshold (0.15): Minimum similarity score for TF-IDF matches
        # - Lower values (<0.15) will return more matches but may be less accurate
        # - Higher values (>0.15) will be more strict, returning only close matches
        # - 0.15 provides a good balance between recall and precision
        self.tfidf_threshold = float(os.environ.get('DEFAULT_CONFIDENCE_THRESHOLD', 0.15))

        # semantic_threshold (0.25): Minimum similarity for semantic search matches
        # - Lower values (<0.25) will match more diverse but possibly less relevant responses
        # - Higher values (>0.25) ensure stronger semantic relevance but may miss valid matches
        # - 0.25 is calibrated for optimal semantic understanding while avoiding false positives
        self.semantic_threshold = float(os.environ.get('SEMANTIC_CONFIDENCE_THRESHOLD', 0.25))

        # high_confidence_threshold (0.7): Threshold for high-confidence direct matches
        # - Lower values (<0.7) risk accepting potentially incorrect matches
        # - Higher values (>0.7) ensure only very close matches are considered high confidence
        # - 0.7 represents a strict threshold where matches are highly likely to be correct
        self.high_confidence_threshold = float(os.environ.get('HIGH_CONFIDENCE_THRESHOLD', 0.7))

        # NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Initialize domain-specific dictionary for spell checking
        self.domain_dict = {
            "rbu": "Ramdeo Baba University",
            "RBU": "Ramdeo Baba University",
            "ramdeo": "Ramdeo Baba University",
            "nagpur": "Nagpur",
            "btech": "B.Tech",
            "mtech": "M.Tech",
            "bba": "BBA",
            "mba": "MBA",
            "bsc": "B.Sc",
            "msc": "M.Sc",
            "ba": "B.A",
            "ma": "M.A",
            "university": "Ramdeo Baba University",
            "college": "Ramdeo Baba University"
        }

        # Initialize entity recognition if enabled
        if os.environ.get('USE_ENTITY_RECOGNITION', 'True').lower() == 'true':
            try:
                logger.info("Loading SpaCy model for entity recognition...")
                self.nlp = spacy.load('en_core_web_sm')
                self.ner_available = True
                logger.info("SpaCy model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading SpaCy model: {e}")

        # Prepare data for TF-IDF
        self.prepare_tfidf_data()

        # Initialize word embeddings if enabled
        self.word_embeddings_available = False
        if os.environ.get('USE_WORD_EMBEDDINGS', 'True').lower() == 'true':
            try:
                word_vectors_path = Path(__file__).parent / 'models' / 'word_vectors.bin'
                if word_vectors_path.exists():
                    logger.info("Loading word embeddings...")
                    self.word_vectors = KeyedVectors.load(str(word_vectors_path))
                    self.word_embeddings_available = True
                    logger.info("Word embeddings loaded successfully.")
                else:
                    logger.error("Word embeddings file not found. Please ensure the file exists at 'models/word_vectors.bin'.")
            except Exception as e:
                logger.error(f"Error loading word embeddings: {e}")

        # Try to load DistilBERT model if semantic search is enabled
        self.semantic_search_available = False
        if os.environ.get('USE_SEMANTIC_SEARCH', 'True').lower() == 'true':
            try:
                logger.info("Loading SentenceTransformer model for semantic matching...")
                self.semantic_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
                self.semantic_embeddings = self.generate_semantic_embeddings(self.questions)
                self.semantic_search_available = True
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                self.semantic_search_available = False
        
        # Performance metrics
        self.query_count = 0
        self.fallbacks = 0
        self.response_times = []
        self.structured_data_matches = 0
        self.tfidf_matches = 0
        self.semantic_matches = 0
        self.api_fallback_count = 0
        self.frequent_queries = {}
        self.source_distribution = {
            'structured_data': 0,
            'tfidf': 0,
            'hybrid': 0,
            'semantic': 0,
            'api_fallback': 0,
            'fallback': 0,
            'quick_response': 0
        }

    def load_data(self):
        """Load and prepare the data from JSON file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Data loaded successfully from {self.data_file}")
            self.faqs = self.data.get('faqs', [])
            self.questions = [faq['question'] for faq in self.faqs]
            self.answers = [faq['answer'] for faq in self.faqs]
            self.quick_responses = self.data.get('quick_responses', {})
            self.university_info = self.data.get('university_info', {})
            self.college_info = self.data.get('college_info', {})
            self.faculty_info = self.data.get('faculty', {})
            self.courses_info = self.data.get('courses', {})
            self.facilities_info = self.data.get('facilities', {})
            self.events_info = self.data.get('events', {})
            self.research_info = self.data.get('research', {})
            self.generate_qa_pairs_from_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.data = {"faqs": [], "quick_responses": {
                "default": ["I'm sorry, I couldn't access the university data. Please try again later."]
            }}
            self.faqs = []
            self.questions = []
            self.answers = []
            self.quick_responses = self.data.get("quick_responses", {})

    def prepare_tfidf_data(self):
        """Prepare TF-IDF vectorizer and matrix."""
        try:
            if not self.questions:
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(["dummy text"])
                logger.warning("No questions found, using dummy TF-IDF matrix")
                return
            processed_questions = [self.preprocess_text(q) for q in self.questions]
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_questions)
            logger.info("TF-IDF vectorizer and matrix prepared successfully.")
        except Exception as e:
            logger.error(f"Error preparing TF-IDF data: {e}")
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(["dummy text"])

    def preprocess_text(self, text):
        """Preprocess text for better matching."""
        if not text:
            return ""

        # Use SpaCy for tokenization, lemmatization, and entity recognition
        if self.ner_available:
            try:
                doc = self.nlp(text.lower())  # Lowercase the text
                tokens = []
                for token in doc:
                    # Exclude stopwords and punctuation, but keep important entities
                    if not token.is_stop and not token.is_punct:
                        if token.ent_type_ and token.ent_type_ in ["ORG", "GPE", "PERSON", "PROGRAM"]:
                            tokens.append(token.text)  # Preserve named entities
                        else:
                            tokens.append(token.lemma_)  # Use lemmatized form for other tokens
                return ' '.join(tokens)
            except Exception as e:
                logger.error(f"Error during SpaCy preprocessing: {e}")
                pass  # Fallback to basic preprocessing if SpaCy fails

        # Basic preprocessing as a fallback
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        tokens = word_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def expand_with_synonyms(self, text):
        """Expand text with synonyms from WordNet.
        Args:
            text (str): Input text
        Returns:
            str: Text expanded with synonyms
        """
        tokens = text.split()
        expanded_tokens = tokens.copy()

        # Custom domain-specific synonyms
        domain_synonyms = {
            'fee': ['fees', 'cost', 'price', 'tuition'],
            'course': ['program', 'degree', 'major', 'specialization'],
            'admission': ['enrollment', 'registration', 'application'],
            'hostel': ['dormitory', 'accommodation', 'residence'],
            'faculty': ['professor', 'teacher', 'instructor', 'staff'],
            'exam': ['examination', 'test', 'assessment'],
            'placement': ['job', 'career', 'employment', 'recruitment'],
            'scholarship': ['financial aid', 'grant', 'fellowship'],
            'principal': ['director', 'head', 'chief'],
            'hod': ['head of department', 'department head', 'chairperson'],
            'canteen': ['cafeteria', 'mess', 'food court', 'dining'],
            'library': ['book center', 'resource center', 'study center'],
            'lab': ['laboratory', 'workshop', 'practical room'],
            'sports': ['games', 'athletics', 'physical activities'],
            'research': ['innovation', 'development', 'investigation', 'study'],
            'seminar': ['workshop', 'conference', 'symposium', 'lecture'],
            'alumni': ['graduate', 'former student', 'ex-student'],
            'infrastructure': ['facilities', 'amenities', 'buildings', 'campus'],
            'ramdeo': ['rbu', 'university'],
            'nagpur': ['city'],
        }

        # Add domain-specific synonyms
        for token in tokens:
            if token in domain_synonyms:
                expanded_tokens.extend(domain_synonyms[token])
                continue

            # Add WordNet synonyms (only for nouns and verbs to avoid too much expansion)
            synonyms = []
            for pos in [wordnet.NOUN, wordnet.VERB]:
                for synset in wordnet.synsets(token, pos=pos):
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token and synonym not in synonyms:
                            synonyms.append(synonym)

            # Limit to top 3 synonyms to avoid excessive expansion
            expanded_tokens.extend(synonyms[:3])

        # Remove duplicates and join
        expanded_text = ' '.join(list(dict.fromkeys(expanded_tokens)))
        return expanded_text

    def spell_check(self, text):
        """Apply basic spell checking with domain-specific terms."""
        # Only replace full words, not parts of words
        processed_text = text
        for term, replacement in self.domain_dict.items():
            processed_text = re.sub(r'\b' + term + r'\b', replacement, processed_text, flags=re.IGNORECASE)
        return processed_text

    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and len(synonym) > 3:  # Only add if different and not too short
                        synonyms.add(synonym)
            
            # Add domain-specific synonyms
            if word.lower() == "university":
                synonyms.add("college")
            elif word.lower() == "program":
                synonyms.add("course")
                synonyms.add("degree")
            elif word.lower() == "student":
                synonyms.add("learner")
                synonyms.add("pupil")
            
            return list(synonyms)[:3]  # Limit to top 3 synonyms to avoid noise
        except Exception as e:
            logger.error(f"Error getting synonyms: {e}")
            return []
    
    def expand_query(self, query):
        """Expand query with synonyms. This is now handled by expand_with_synonyms."""
        return query  # Just return the original query as this is handled elsewhere
    
    def extract_entities(self, text):
        """Extract named entities from text using SpaCy."""
        if not self.ner_available:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = {}
            
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            # Add custom entity extraction for university-specific terms
            programs = ["B.Tech", "M.Tech", "BBA", "MBA", "B.Sc", "M.Sc", "B.A", "M.A", "Ph.D"]
            for program in programs:
                if program.lower() in text.lower():
                    if "PROGRAM" not in entities:
                        entities["PROGRAM"] = []
                    entities["PROGRAM"].append(program)
            
            return entities
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {}
    
    def extract_intent(self, query):
        """Extract intent from user query."""
        query_lower = query.lower()
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in query_lower for greeting in greetings):
            return 'greeting'
        
        # Check for thanks
        thanks = ['thank you', 'thanks', 'appreciate', 'grateful']
        if any(thank in query_lower for thank in thanks):
            return 'thanks'
        
        # Check for goodbye
        goodbyes = ['bye', 'goodbye', 'see you', 'talk to you later', 'cya']
        if any(goodbye in query_lower for goodbye in goodbyes):
            return 'goodbye'
        
        # Otherwise, it's a general query
        return 'query'
    
    def get_tfidf_response(self, query):
        """Get response using TF-IDF and cosine similarity.
        Args:
            query (str): User query
        Returns:
            tuple: (response, confidence)
        """
        try:
            # Preprocess the query
            processed_query = self.preprocess_text(query)
            
            # Apply spell check if enabled
            if os.environ.get('USE_SPELL_CHECK', 'True').lower() == 'true':
                processed_query = self.spell_check(processed_query)
            
            # Expand query with synonyms if enabled
            if os.environ.get('USE_QUERY_EXPANSION', 'True').lower() == 'true':
                processed_query = self.expand_query(processed_query)
            
            # Transform the query using the TF-IDF vectorizer
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate cosine similarity between the query and all questions
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Find the index of the question with the highest similarity
            best_match_index = cosine_similarities.argmax()
            best_match_score = cosine_similarities[best_match_index]
            
            # If the best match score is above the threshold, return the corresponding answer
            if best_match_score >= self.tfidf_threshold:
                return self.answers[best_match_index], float(best_match_score)
            
            # No good match found
            return "", 0.0
        except Exception as e:
            logger.error(f"Error in TF-IDF response: {e}")
            return "", 0.0

    def generate_semantic_embeddings(self, texts):
        """Generate semantic embeddings for a list of texts."""
        try:
            return self.semantic_model.encode(texts, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error generating semantic embeddings: {e}")
            return None

    def get_semantic_response(self, query):
        """Get response using semantic search with SentenceTransformer."""
        if not self.semantic_search_available:
            return "", 0.0

        try:
            # Generate embedding for the query
            query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)

            # Calculate cosine similarity between the query and all questions
            cosine_similarities = cosine_similarity(query_embedding.unsqueeze(0), self.semantic_embeddings).flatten()

            # Find the index of the question with the highest similarity
            best_match_index = cosine_similarities.argmax()
            best_match_score = cosine_similarities[best_match_index]

            # If the best match score is above the threshold, return the corresponding answer
            if best_match_score >= self.semantic_threshold:
                return self.answers[best_match_index], float(best_match_score)

            # No good match found
            return "", 0.0
        except Exception as e:
            logger.error(f"Error in semantic response: {e}")
            return "", 0.0

    def get_hybrid_response(self, query):
        """Get response using a weighted combination of TF-IDF and semantic search scores."""
        tfidf_response, tfidf_confidence = self.get_tfidf_response(query)

        if self.semantic_search_available:
            semantic_response, semantic_confidence = self.get_semantic_response(query)
        else:
            semantic_response, semantic_confidence = "", 0.0

        tfidf_weight = float(os.environ.get('TFIDF_WEIGHT', 0.4))
        semantic_weight = float(os.environ.get('SEMANTIC_WEIGHT', 0.6))
        total_weight = tfidf_weight + semantic_weight
        tfidf_weight /= total_weight
        semantic_weight /= total_weight

        weighted_confidence = (tfidf_weight * tfidf_confidence) + (semantic_weight * semantic_confidence)
        response = semantic_response if semantic_confidence * semantic_weight > tfidf_confidence * tfidf_weight else tfidf_response

        if weighted_confidence < float(os.environ.get('HYBRID_SIMILARITY_THRESHOLD', 0.7)):
            return "", 0.0

        return response, weighted_confidence

    def get_fallback_response(self):
        """Return a random fallback response when no match is found."""
        self.fallbacks += 1
        fallback_responses = self.quick_responses.get('default', [
            "I'm sorry, I couldn't find information related to your query. Please try asking something about our programs, admissions, fees, or campus facilities.",
            "Sorry, I was unable to find a good match for your question. Could you please rephrase or ask about another topic related to the university?",
            "I don't have specific information on that topic yet. Feel free to ask about our courses, admission process, facilities, or faculty instead."
        ])
        return random.choice(fallback_responses)

    def fetch_online_data(self, query):
        """Fetch data from external API when confidence is low."""
        if os.environ.get('USE_API_FALLBACK', 'True').lower() != 'true':
            return ""

        try:
            university_name = os.environ.get('UNIVERSITY_NAME', 'Ramdeo Baba University')
            university_location = os.environ.get('UNIVERSITY_LOCATION', 'Nagpur')
            contextualized_query = f"{query} {university_name} {university_location}"

            serpapi_key = os.environ.get('SERPAPI_KEY')
            if serpapi_key and serpapi_key != 'your_serpapi_key_here':
                return self.fetch_from_serpapi(contextualized_query)

            logger.info("SerpAPI key not configured. Skipping SerpAPI.")
            return "API fallback is not configured. Please check your environment variables."
        except Exception as e:
            logger.error(f"Error fetching online data: {e}")
            return "An error occurred while fetching online data. Please try again later."

    def fetch_from_serpapi(self, query):
        """Fetch data from SerpAPI."""
        try:
            api_key = os.environ.get('SERPAPI_KEY')
            if not api_key or api_key == 'your_serpapi_key_here':
                logger.error("SerpAPI key is missing or invalid.")
                return "SerpAPI key is not configured. Please check your environment variables."

            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": api_key,
                "engine": "google"
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"SerpAPI request failed with status code {response.status_code}: {response.text}")
                return "Failed to fetch data from SerpAPI. Please try again later."

            data = response.json()

            # Extract organic results
            if 'organic_results' in data and len(data['organic_results']) > 0:
                top_result = data['organic_results'][0]

                # Format the response
                title = top_result.get('title', 'No title available')
                snippet = top_result.get('snippet', 'No snippet available')
                link = top_result.get('link', 'No link available')

                formatted_response = f"{snippet}\n\nSource: {title} ({link})"
                return formatted_response

            logger.info("No organic results found in SerpAPI response.")
            return "No relevant results found. Please try rephrasing your query."
        except Exception as e:
            logger.error(f"Error in SerpAPI request: {e}")
            return "An error occurred while fetching data from SerpAPI. Please try again later."

    def get_quick_response(self, intent):
        """Return a quick response based on the detected intent."""
        if intent in self.quick_responses:
            responses = self.quick_responses[intent]
            return random.choice(responses)
        return None

    def format_direct_matches(self, results):
        """Format direct matches from structured data into user-friendly responses."""
        if not results:
            return "I'm sorry, I couldn't find any relevant information."
        unique_results = list(dict.fromkeys(results))  # Remove duplicates
        if len(unique_results) == 1:
            return self.format_single_response(unique_results[0])
        grouped_results = self.group_results_by_category(unique_results)
        return self.format_grouped_results(grouped_results)

    def format_single_response(self, response):
        """Format a single response to be more user-friendly."""
        # Remove technical path information
        if " > " in response:
            parts = response.split(" > ")
            if ":" in parts[-1]:
                key, value = parts[-1].split(":", 1)
                response = f"{key.strip()}: {value.strip()}"
        return response.strip()

    def group_results_by_category(self, results):
        """Group results by their categories."""
        categories = {
            'fee': [],
            'hostel': [],
            'scholarship': [],
            'admission': [],
            'course': [],
            'faculty': [],
            'facility': [],
            'time_table': [],
            'sports': [],
            'other': [],
        }
        for result in results:
            result_lower = result.lower()
            if 'fee' in result_lower or 'cost' in result_lower or 'price' in result_lower:
                categories['fee'].append(result)
            elif 'hostel' in result_lower or 'accommodation' in result_lower:
                categories['hostel'].append(result)
            elif 'scholarship' in result_lower:
                categories['scholarship'].append(result)
            elif 'admission' in result_lower or 'apply' in result_lower:
                categories['admission'].append(result)
            elif 'course' in result_lower or 'program' in result_lower or 'degree' in result_lower:
                categories['course'].append(result)
            elif 'faculty' in result_lower or 'professor' in result_lower or 'teacher' in result_lower:
                categories['faculty'].append(result)
            elif 'facility' in result_lower or 'amenity' in result_lower or 'infrastructure' in result_lower:
                categories['facility'].append(result)
            elif 'time' in result_lower or 'schedule' in result_lower:
                categories['time_table'].append(result)
            elif 'sports' in result_lower or 'ground' in result_lower:
                categories['sports'].append(result)
            else:
                categories['other'].append(result)
        return {k: v for k, v in categories.items() if v}
    
    def format_grouped_results(self, grouped_results):
        """Format grouped results into a readable response."""
        if not grouped_results:
            return "I'm sorry, I couldn't find any relevant information."
        response = "Here's what I found:\n\n"
        for category, items in grouped_results.items():
            if category == 'fee':
                response += "Fee Information:\n"
            elif category == 'hostel':
                response += "Hostel Information:\n"
            elif category == 'scholarship':
                response += "Scholarship Information:\n"
            elif category == 'admission':
                response += "Admission Information:\n"
            elif category == 'course':
                response += "Course Information:\n"
            elif category == 'faculty':
                response += "Faculty Information:\n"
            elif category == 'facility':
                response += "Facility Information:\n"
            elif category == 'time_table':
                response += "Time Table Information:\n"
            elif category == 'sports':
                response += "Sports Information:\n"
            else:
                response += "Other Information:\n"
            for item in items:
                if " > " in item:
                    parts = item.split(" > ")
                    if ":" in parts[-1]:
                        key, value = parts[-1].split(":", 1)
                        item = f"{key.strip()}: {value.strip()}"
                response += f"• {item}\n"
            response += "\n"
        return response.strip()

    def process_query(self, query):
        """Process user query and return appropriate response."""
        start_time = time.time()
        self.query_count += 1
        self.frequent_queries[query] = self.frequent_queries.get(query, 0) + 1

        # Preprocess the query
        processed_query = self.preprocess_text(query)

        # Extract intent
        intent = self.extract_intent(query)
        if intent in self.quick_responses:
            response = self.get_quick_response(intent)
            return self._build_response(response, 'quick_response', 1.0, start_time, query)

        # Search in structured data
        all_results = []
        for section_name, section_data in [
            ('college_info', self.college_info),
            ('faculty_info', self.faculty_info),
            ('courses_info', self.courses_info),
            ('facilities_info', self.facilities_info),
            ('events_info', self.events_info),
            ('research_info', self.research_info),
        ]:
            if section_data:
                results = self.search_in_data_structure(query, section_data)
                all_results.extend(results)

        if all_results:
            best_result, confidence = max(all_results, key=lambda x: x[1])
            if confidence >= 0.5:  # Confidence threshold for structured data
                self.structured_data_matches += 1
                return self._build_response(best_result, 'structured_data', confidence, start_time, query)

        # Get response using hybrid approach (TF-IDF + Semantic)
        response, confidence = self.get_hybrid_response(query)
        if confidence >= 0.8:  # High semantic match threshold
            source = 'hybrid' if self.semantic_search_available else 'tfidf'
            if source == 'tfidf':
                self.tfidf_matches += 1
            elif source == 'hybrid':
                self.semantic_matches += 1
            return self._build_response(response, source, confidence, start_time, query)

        # Handle rule-based logic for specific intents
        if "fees" in processed_query or "course list" in processed_query:
            rule_based_response = self.handle_using_rule_based_logic(processed_query)
            if rule_based_response:
                return self._build_response(rule_based_response, 'rule_based', 0.7, start_time, query)

        # Fetch data from SerpAPI if enabled
        serp_api_key = os.environ.get('SERPAPI_KEY')
        if serp_api_key:
            api_response = self.get_answer_from_serpapi(query)
            if api_response:
                self.api_fallback_count += 1
                return self._build_response(api_response, 'api_fallback', 0.5, start_time, query)

        # Return fallback response
        fallback_response = self.get_fallback_response()
        return self._build_response(fallback_response, 'fallback', 0.0, start_time, query)

    def handle_using_rule_based_logic(self, query):
        """Handle specific intents using rule-based logic."""
        if "fees" in query:
            return "The fee structure varies by program. Please specify the program you're interested in."
        elif "course list" in query:
            return "RBU offers a variety of courses including B.Tech, MBA, BBA, and more. Please specify your area of interest."
        return None

    def get_answer_from_serpapi(self, query):
        """Fetch an answer from SerpAPI."""
        serp_api_key = os.environ.get('SERPAPI_KEY')
        if not serp_api_key:
            logger.error("SerpAPI key is missing or invalid.")
            return "I couldn't find an answer on the web."

        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": serp_api_key
            }
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Extract snippet from organic results
            if "organic_results" in data:
                for result in data["organic_results"]:
                    if "snippet" in result:
                        return result["snippet"]

            # Fallback if no good result
            return "I couldn't find an answer on the web."
        except Exception as e:
            logger.error(f"Error fetching data from SerpAPI: {e}")
            return "An error occurred while fetching data from the web."

    def _build_response(self, response, source, confidence, start_time, query):
        """Helper method to build a response dictionary."""
        response_time = time.time() - start_time
        self.response_times.append(response_time)

        # Dynamically add the source key to source_distribution if it doesn't exist
        if source not in self.source_distribution:
            self.source_distribution[source] = 0
        self.source_distribution[source] += 1

        return {
            'response': response,
            'source': source,
            'confidence': confidence,
            'response_time': response_time,
            'original_query': query,
        }
    
    def get_performance_metrics(self):
        """Return comprehensive performance metrics of the chatbot."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        successful_responses = self.structured_data_matches + self.tfidf_matches + self.semantic_matches
        success_rate = (successful_responses / self.query_count) * 100 if self.query_count > 0 else 0
        top_queries = sorted(self.frequent_queries.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            'query_count': self.query_count,
            'successful_matches': successful_responses,  # Updated to reflect successful matches
            'structured_data_matches': self.structured_data_matches,
            'tfidf_matches': self.tfidf_matches,
            'semantic_matches': self.semantic_matches,
            'api_fallback_count': self.api_fallback_count,
            'fallbacks': self.fallbacks,
            'success_rate': f"{success_rate:.2f}%",  # Updated to calculate success rate
            'average_response_time': f"{avg_response_time:.4f}s",
            'source_distribution': self.source_distribution,
            'top_queries': dict(top_queries),
        }
    
    def get_faq_suggestions(self, count=5):
        """Return a list of suggested FAQs for quick access.
        Args:
            count (int): Number of suggestions to return
        Returns:
            list: List of suggested questions
        """
        if not self.questions:
            return [
                "What programs does Ramdeo Baba University offer?",
                "What is the fee structure for B.Tech at RBU?",
                "How can I apply to Ramdeo Baba University?",
                "What are the hostel facilities at RBU?",
                "When do admissions start at Ramdeo Baba University?"
            ]
        if self.frequent_queries:
            top_queries = sorted(self.frequent_queries.items(), key=lambda x: x[1], reverse=True)
            if len(top_queries) >= count:
                return [query for query, _ in top_queries[:count]]

        return random.sample(self.questions, min(count, len(self.questions)))

    def reset_metrics(self):
        """Reset performance metrics."""
        self.query_count = 0
        self.fallbacks = 0
        self.response_times = []
        self.structured_data_matches = 0
        self.tfidf_matches = 0
        self.semantic_matches = 0
        self.api_fallback_count = 0
        self.frequent_queries = {}
        self.source_distribution = {
            'structured_data': 0,
            'tfidf': 0,
            'hybrid': 0,
            'semantic': 0,
            'api_fallback': 0,
            'fallback': 0,
            'quick_response': 0,
        }
        logger.info("Performance metrics reset")

    def generate_bert_embeddings(self, texts):
        """Generate embeddings for a list of texts using DistilBERT."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().numpy()
    
    def generate_qa_pairs_from_data(self):
        """Generate additional question-answer pairs from structured data."""
        additional_pairs = []

        # Process college_info - only extract core information
        if self.college_info:
            if 'name' in self.college_info:
                additional_pairs.append({
                    'question': f"What is the full name of RBU?",
                    'answer': f"The full name is {self.college_info.get('name')}.",
                    'category': 'general'
                })
            if 'established' in self.college_info:
                additional_pairs.append({
                    'question': f"When was RBU established?",
                    'answer': f"RBU was established in {self.college_info.get('established')}.",
                    'category': 'general'
                })

        # Add the additional pairs to existing FAQs
        self.faqs.extend(additional_pairs)

        # Update questions and answers lists
        self.questions = [faq['question'] for faq in self.faqs]
        self.answers = [faq['answer'] for faq in self.faqs]

    def search_in_data_structure(self, query, section, key_path=None, max_results=3):
        """Search for information in nested data structures with improved accuracy."""
        if key_path is None:
            key_path = []

        # Process query to extract key terms
        processed_query = self.preprocess_text(query)
        query_terms = set(processed_query.split())

        results = []

        # If section is a dictionary
        if isinstance(section, dict):
            for key, value in section.items():
                processed_key = self.preprocess_text(key.replace('_', ' '))
                key_terms = set(processed_key.split())

                # Calculate term overlap
                overlap = len(query_terms.intersection(key_terms)) / len(query_terms) if query_terms else 0

                if overlap > 0.4:  # Increased overlap threshold for better precision
                    if isinstance(value, (str, int, float)):
                        context = " > ".join([str(k).replace('_', ' ').title() for k in key_path])
                        result = f"{context} > {key.replace('_', ' ').title()}: {value}" if context else f"{key.replace('_', ' ').title()}: {value}"
                        results.append((result, overlap))  # Include confidence score
                        if len(results) >= max_results:
                            return results
                    elif isinstance(value, (dict, list)):
                        nested_results = self.search_in_data_structure(query, value, key_path + [key], max_results - len(results))
                        results.extend(nested_results)
                        if len(results) >= max_results:
                            return results

        # If section is a list
        elif isinstance(section, list):
            for i, item in enumerate(section):
                if isinstance(item, (dict, list)):
                    nested_results = self.search_in_data_structure(query, item, key_path + [i], max_results - len(results))
                    results.extend(nested_results)
                    if len(results) >= max_results:
                        return results
                elif isinstance(item, str):
                    processed_item = self.preprocess_text(item)
                    item_terms = set(processed_item.split())
                    overlap = len(query_terms.intersection(item_terms)) / len(query_terms) if query_terms else 0

                    if overlap > 0.4:  # Increased overlap threshold
                        context = " > ".join([str(k).replace('_', ' ').title() for k in key_path])
                        result = f"{context}: {item}" if context else item
                        results.append((result, overlap))  # Include confidence score
                        if len(results) >= max_results:
                            return results

        return results

# For testing purposes
if __name__ == "__main__":
    chatbot = RBUChatbot()
    # Test the chatbot with a few queries
    test_queries = [
        "Hello there",
        "What programs does Ramdeo Baba University offer?",
        "How much is the MBA fee at RBU?",
        "When do admissions start at Ramdeo Baba University?",
        "What are the hostel facilities at RBU?",
        "Thank you for your help",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.process_query(query)
        try:
            print(f"Response: {response['response']}")
        except UnicodeEncodeError:
            # Replace problematic characters with their ASCII equivalents
            safe_response = response['response'].replace('₹', 'Rs.')
            print(f"Response: {safe_response}")
        print(f"Source: {response['source']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Response time: {response['response_time']}")

    # Print performance metrics
    print("\nPerformance Metrics:")
    metrics = chatbot.get_performance_metrics()
    for key, value in metrics.items():
        if key == 'source_distribution' or key == 'top_queries':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")