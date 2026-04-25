"""
Emotion Classification API Server
Tích hợp vào web forum để phân tích cảm xúc comments

Usage:
    python api_server.py
    
API Endpoints:
    POST /predict - Phân tích cảm xúc từ text
    GET /health - Health check
"""

import fix_encoding  # Fix Windows emoji encoding
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import numpy as np
from transformers import BertTokenizer
import os
import logging

from config import Config
from model import BERTEmotionClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web integration

# Global variables for model
model = None
tokenizer = None
device = None

def load_model():
    """Load trained model and tokenizer"""
    global model, tokenizer, device
    
    try:
        model_path = "saved_model/"
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load model
        model = BERTEmotionClassifier(num_labels=len(Config.EMOTION_LABELS))
        model_file = os.path.join(model_path, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        logger.info("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def predict_emotions(text, threshold=0.4):
    """
    Predict emotions from text
    
    Args:
        text (str): Input text
        threshold (float): Confidence threshold
        
    Returns:
        dict: Prediction results with emotions and sentiment
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Process results
        emotions = {}
        detected_emotions = []
        
        for i, emotion in enumerate(Config.EMOTION_LABELS):
            score = float(probabilities[i])
            emotions[emotion] = {
                'score': score,
                'vietnamese': Config.EMOTION_LABELS_VI.get(emotion, emotion),
                'detected': score >= threshold
            }
            
            if score >= threshold:
                detected_emotions.append({
                    'emotion': emotion,
                    'vietnamese': Config.EMOTION_LABELS_VI.get(emotion, emotion),
                    'score': score
                })
        
        # Calculate sentiment (positive vs negative)
        positive_emotions = ['joy', 'trust', 'surprise', 'love', 'proud', 'excited', 'calm']
        negative_emotions = ['fear', 'sadness', 'disgust', 'anger', 'worried', 'disappointed', 'embarrassed', 'jealous']
        
        positive_score = np.mean([emotions[e]['score'] for e in positive_emotions])
        negative_score = np.mean([emotions[e]['score'] for e in negative_emotions])
        neutral_score = 1 - max(positive_score, negative_score)
        
        # Normalize sentiment scores
        total = positive_score + negative_score + neutral_score
        sentiment = {
            'positive': float(positive_score / total),
            'negative': float(negative_score / total),
            'neutral': float(neutral_score / total)
        }
        
        # Determine overall sentiment
        if positive_score > negative_score and positive_score > 0.3:
            overall_sentiment = 'positive'
        elif negative_score > positive_score and negative_score > 0.3:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'success': True,
            'text': text,
            'emotions': emotions,
            'detected_emotions': detected_emotions,
            'sentiment': sentiment,
            'overall_sentiment': overall_sentiment,
            'threshold': threshold
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/', methods=['GET'])
def index():
    """Serve the web demo interface"""
    try:
        with open('web_demo.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except Exception as e:
        return f"Error loading web_demo.html: {e}", 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotions from text
    
    Request JSON:
        {
            "text": "Tôi rất vui với sản phẩm này!",
            "threshold": 0.4  // optional
        }
    
    Response JSON:
        {
            "success": true,
            "emotions": {...},
            "sentiment": {...},
            "detected_emotions": [...]
        }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing text field in request'
            }), 400
        
        text = data['text'].strip()
        threshold = data.get('threshold', 0.4)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Empty text'
            }), 400
        
        # Predict emotions
        result = predict_emotions(text, threshold)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/emotions/list', methods=['GET'])
def list_emotions():
    """Get list of supported emotions"""
    emotions_list = []
    for emotion in Config.EMOTION_LABELS:
        emotions_list.append({
            'english': emotion,
            'vietnamese': Config.EMOTION_LABELS_VI.get(emotion, emotion)
        })
    
    return jsonify({
        'emotions': emotions_list,
        'total': len(emotions_list)
    })

if __name__ == '__main__':
    print("🚀 Starting Emotion Classification API Server...")
    
    # Load model
    if load_model():
        print("✅ Model loaded successfully!")
        print("🌐 API Server starting on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  POST /predict - Analyze emotions")
        print("  GET /health - Health check")
        print("  GET /emotions/list - List supported emotions")
        
        # Start server
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to load model. Please check saved_model/ directory.")