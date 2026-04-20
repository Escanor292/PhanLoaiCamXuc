"""
Prediction module for Multi-label Emotion Classification system.

This module provides functions for predicting emotions from comment text using
a trained BERT-based model. It supports both single and batch predictions, and
includes an interactive command-line interface for user interaction.
"""

import torch
import numpy as np
from transformers import BertTokenizer
from config import Config
from utils import clean_text, load_model


def predict_emotions(text, model, tokenizer, device, threshold=None):
    """
    Predicts emotions for a single comment.
    
    This function takes a raw comment text, preprocesses it, tokenizes it,
    passes it through the trained model, and returns predicted emotions along
    with confidence scores for all 16 emotion labels.
    
    Args:
        text (str): Input comment text to analyze for emotions.
        model: Trained BERTEmotionClassifier instance.
        tokenizer: BERT tokenizer instance.
        device (torch.device or str): Device to run inference on ('cpu' or 'cuda').
        threshold (float, optional): Confidence threshold for emotion prediction.
                                     Emotions with scores >= threshold are included
                                     in the predicted emotions list.
                                     Defaults to Config.PREDICTION_THRESHOLD (0.5).
    
    Returns:
        dict: Dictionary containing:
            - 'emotions' (list): List of predicted emotion labels (strings)
                                that exceed the threshold. Empty list if no
                                emotions exceed threshold.
            - 'scores' (dict): Dictionary mapping all 16 emotion labels to
                              their confidence scores (float, range [0, 1]).
    
    Raises:
        ValueError: If input text is empty or invalid.
        RuntimeError: If model inference fails.
    
    Examples:
        >>> model, tokenizer = load_model('saved_model/', device='cpu')
        >>> result = predict_emotions("I love this!", model, tokenizer, 'cpu')
        >>> print(result['emotions'])
        ['joy', 'love', 'excited']
        >>> print(result['scores']['joy'])
        0.92
        
        >>> # With custom threshold
        >>> result = predict_emotions("Great!", model, tokenizer, 'cpu', threshold=0.7)
        >>> print(result['emotions'])
        ['joy']  # Only emotions with score >= 0.7
    
    Notes:
        - Text is automatically preprocessed (cleaned and normalized)
        - If no emotions exceed threshold, returns empty emotions list
        - All 16 confidence scores are always returned regardless of threshold
        - Model is automatically set to evaluation mode
    """
    # Input validation
    if not text or not isinstance(text, str):
        raise ValueError(
            "Input text must be a non-empty string. "
            f"Received: {type(text).__name__}"
        )
    
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty or whitespace only.")
    
    # Use default threshold if not provided
    if threshold is None:
        threshold = Config.PREDICTION_THRESHOLD
    
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Threshold must be between 0 and 1. Got: {threshold}"
        )
    
    # Preprocess text
    cleaned_text = clean_text(text)
    
    # Handle case where cleaning removes all content
    if not cleaned_text:
        # Return neutral prediction with low confidence
        return {
            'emotions': [],
            'scores': {label: 0.0 for label in Config.EMOTION_LABELS}
        }
    
    # Tokenize input
    try:
        encoding = tokenizer(
            cleaned_text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    except Exception as e:
        raise RuntimeError(f"Tokenization failed: {str(e)}")
    
    # Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    try:
        with torch.no_grad():
            # Forward pass to get logits
            logits = model(input_ids, attention_mask)
            
            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(logits)
            
            # Move to CPU and convert to numpy
            scores = probabilities.cpu().numpy()[0]  # Shape: (16,)
    
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {str(e)}")
    
    # Create scores dictionary mapping emotion labels to confidence scores
    scores_dict = {
        label: float(score) 
        for label, score in zip(Config.EMOTION_LABELS, scores)
    }
    
    # Apply threshold to determine predicted emotions
    predicted_emotions = [
        label 
        for label, score in scores_dict.items() 
        if score >= threshold
    ]
    
    # Sort predicted emotions by confidence score (descending)
    predicted_emotions.sort(key=lambda label: scores_dict[label], reverse=True)
    
    return {
        'emotions': predicted_emotions,
        'scores': scores_dict
    }


def predict_emotions_batch(texts, model, tokenizer, device, threshold=None):
    """
    Predicts emotions for multiple comments efficiently in batch mode.
    
    This function extends single prediction to handle multiple comments
    simultaneously, which is more efficient than processing them one by one.
    It's particularly useful for analyzing large volumes of comments.
    
    Args:
        texts (list of str): List of input comment texts to analyze.
        model: Trained BERTEmotionClassifier instance.
        tokenizer: BERT tokenizer instance.
        device (torch.device or str): Device to run inference on ('cpu' or 'cuda').
        threshold (float, optional): Confidence threshold for emotion prediction.
                                     Defaults to Config.PREDICTION_THRESHOLD (0.5).
    
    Returns:
        list of dict: List of prediction dictionaries, one per input text.
                     Each dictionary contains:
                         - 'emotions' (list): Predicted emotion labels
                         - 'scores' (dict): All confidence scores
                     Order matches input texts order.
    
    Raises:
        ValueError: If texts is empty or contains invalid entries.
        RuntimeError: If batch inference fails.
    
    Examples:
        >>> model, tokenizer = load_model('saved_model/', device='cpu')
        >>> texts = ["I love this!", "This is terrible.", "Feeling calm."]
        >>> results = predict_emotions_batch(texts, model, tokenizer, 'cpu')
        >>> for i, result in enumerate(results):
        ...     print(f"Text {i+1}: {result['emotions']}")
        Text 1: ['joy', 'love', 'excited']
        Text 2: ['anger', 'disappointed', 'disgust']
        Text 3: ['calm', 'trust']
        
        >>> # Access individual results
        >>> print(results[0]['scores']['joy'])
        0.92
    
    Notes:
        - More efficient than calling predict_emotions() in a loop
        - All texts are preprocessed and tokenized together
        - Batch size is limited by available memory
        - For very large datasets, consider processing in chunks
    """
    # Input validation
    if not texts or not isinstance(texts, list):
        raise ValueError(
            "Input texts must be a non-empty list of strings. "
            f"Received: {type(texts).__name__}"
        )
    
    if len(texts) == 0:
        raise ValueError("Input texts list cannot be empty.")
    
    # Validate all texts are strings
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"All texts must be strings. "
                f"Text at index {i} is {type(text).__name__}"
            )
    
    # Use default threshold if not provided
    if threshold is None:
        threshold = Config.PREDICTION_THRESHOLD
    
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Threshold must be between 0 and 1. Got: {threshold}"
        )
    
    # Preprocess all texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Handle empty texts after cleaning
    # Replace empty strings with a placeholder to maintain batch alignment
    cleaned_texts = [text if text else "[empty]" for text in cleaned_texts]
    
    # Tokenize all texts in batch
    try:
        encodings = tokenizer(
            cleaned_texts,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    except Exception as e:
        raise RuntimeError(f"Batch tokenization failed: {str(e)}")
    
    # Move tensors to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform batch inference
    try:
        with torch.no_grad():
            # Forward pass to get logits for all texts
            logits = model(input_ids, attention_mask)
            
            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(logits)
            
            # Move to CPU and convert to numpy
            scores_batch = probabilities.cpu().numpy()  # Shape: (batch_size, 16)
    
    except Exception as e:
        raise RuntimeError(f"Batch inference failed: {str(e)}")
    
    # Process results for each text
    results = []
    for i, scores in enumerate(scores_batch):
        # Create scores dictionary for this text
        scores_dict = {
            label: float(score) 
            for label, score in zip(Config.EMOTION_LABELS, scores)
        }
        
        # Apply threshold to determine predicted emotions
        predicted_emotions = [
            label 
            for label, score in scores_dict.items() 
            if score >= threshold
        ]
        
        # Sort predicted emotions by confidence score (descending)
        predicted_emotions.sort(key=lambda label: scores_dict[label], reverse=True)
        
        # Handle case where original text was empty
        if texts[i].strip() == "":
            predicted_emotions = []
            scores_dict = {label: 0.0 for label in Config.EMOTION_LABELS}
        
        results.append({
            'emotions': predicted_emotions,
            'scores': scores_dict
        })
    
    return results


def display_prediction(text, result, show_all_scores=False):
    """
    Displays prediction results in a readable format.
    
    Args:
        text (str): Original input text.
        result (dict): Prediction result from predict_emotions().
        show_all_scores (bool): If True, display all 16 emotion scores.
                               If False, only show predicted emotions.
    """
    print("\n" + "="*70)
    print("EMOTION PREDICTION RESULTS")
    print("="*70)
    print(f"\nInput Text: \"{text}\"")
    print("\n" + "-"*70)
    
    if result['emotions']:
        print(f"\nPredicted Emotions ({len(result['emotions'])}):")
        print("-" * 70)
        for emotion in result['emotions']:
            score = result['scores'][emotion]
            # Create a visual bar
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {emotion:15s} [{bar}] {score:.3f}")
    else:
        print("\nPredicted Emotions: None")
        print("(No emotions exceeded the confidence threshold)")
    
    if show_all_scores:
        print("\n" + "-"*70)
        print("\nAll Emotion Confidence Scores:")
        print("-" * 70)
        # Sort by score descending
        sorted_scores = sorted(
            result['scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for emotion, score in sorted_scores:
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            marker = "✓" if emotion in result['emotions'] else " "
            print(f"{marker} {emotion:15s} [{bar}] {score:.3f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """
    Interactive command-line interface for emotion prediction.
    
    This function provides a user-friendly interface that:
    1. Loads the trained model once at startup
    2. Prompts user for input text
    3. Displays predicted emotions and confidence scores
    4. Supports multiple predictions without reloading model
    5. Handles errors gracefully with clear messages
    
    Commands:
        - Enter text: Predict emotions for the text
        - 'quit' or 'exit': Exit the program
        - 'help': Show help message
    
    Examples:
        $ python predict.py
        
        Loading model...
        Model loaded successfully!
        
        Enter a comment (or 'quit' to exit): I love this product!
        
        ======================================================================
        EMOTION PREDICTION RESULTS
        ======================================================================
        
        Input Text: "I love this product!"
        
        ----------------------------------------------------------------------
        
        Predicted Emotions (3):
        ----------------------------------------------------------------------
          joy             [████████████████████████████████████░░░░] 0.920
          love            [██████████████████████████████░░░░░░░░░░] 0.850
          excited         [████████████████████████░░░░░░░░░░░░░░░] 0.720
        
        ======================================================================
    """
    print("\n" + "="*70)
    print("MULTI-LABEL EMOTION CLASSIFICATION SYSTEM")
    print("="*70)
    print("\nThis system predicts emotions from comment text using BERT.")
    print(f"Supported emotions: {', '.join(Config.EMOTION_LABELS[:8])},")
    print(f"                    {', '.join(Config.EMOTION_LABELS[8:])}")
    print("\nCommands:")
    print("  - Enter text to predict emotions")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for more information")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model from:", Config.MODEL_SAVE_DIR)
    print("This may take a moment...")
    
    try:
        model, tokenizer = load_model(Config.MODEL_SAVE_DIR, Config.DEVICE)
        print(f"✓ Model loaded successfully on device: {Config.DEVICE}")
        print(f"✓ Prediction threshold: {Config.PREDICTION_THRESHOLD}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPlease train the model first by running:")
        print("  python train.py")
        return
    except Exception as e:
        print(f"\n✗ Error loading model: {str(e)}")
        print("\nPlease ensure the model was trained correctly.")
        return
    
    print("\n" + "="*70)
    print("Ready for predictions!")
    print("="*70 + "\n")
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("Enter a comment (or 'quit' to exit): ").strip()
            
            # Handle empty input
            if not user_input:
                print("⚠ Please enter some text.\n")
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Emotion Classification System!")
                print("Goodbye!\n")
                break
            
            if user_input.lower() == 'help':
                print("\n" + "="*70)
                print("HELP")
                print("="*70)
                print("\nHow to use:")
                print("  1. Enter any comment text when prompted")
                print("  2. The system will analyze and predict emotions")
                print("  3. Results show emotions that exceed the confidence threshold")
                print(f"     (current threshold: {Config.PREDICTION_THRESHOLD})")
                print("\nTips:")
                print("  - Longer, more expressive text generally works better")
                print("  - The system can detect multiple emotions simultaneously")
                print("  - Confidence scores range from 0.0 (unlikely) to 1.0 (very likely)")
                print("\nCommands:")
                print("  quit, exit, q - Exit the program")
                print("  help          - Show this help message")
                print("="*70 + "\n")
                continue
            
            # Perform prediction
            try:
                result = predict_emotions(
                    user_input, 
                    model, 
                    tokenizer, 
                    Config.DEVICE,
                    threshold=Config.PREDICTION_THRESHOLD
                )
                
                # Display results
                display_prediction(user_input, result, show_all_scores=False)
                
                # Ask if user wants to see all scores
                show_all = input("Show all emotion scores? (y/n): ").strip().lower()
                if show_all in ['y', 'yes']:
                    display_prediction(user_input, result, show_all_scores=True)
            
            except ValueError as e:
                print(f"\n✗ Input Error: {str(e)}\n")
            except RuntimeError as e:
                print(f"\n✗ Prediction Error: {str(e)}\n")
            except Exception as e:
                print(f"\n✗ Unexpected Error: {str(e)}\n")
                print("Please try again with different input.\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            print("Thank you for using the Emotion Classification System!")
            print("Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n✗ Unexpected Error: {str(e)}\n")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
