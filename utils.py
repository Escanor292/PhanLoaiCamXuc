"""
Utility functions for Multi-label Emotion Classification system.

This module contains helper functions for text preprocessing, data loading,
evaluation metrics, visualization, and model persistence.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix for Windows/Headless blocking
import matplotlib.pyplot as plt
from config import Config


def clean_text(text):
    """
    Cleans and normalizes comment text for emotion classification.
    
    This function performs the following operations:
    1. Removes or normalizes URLs (http://, https://, www.)
    2. Removes non-meaningful special characters while preserving emoticons
    3. Normalizes excessive whitespace to single spaces
    4. Preserves emoticons and emoji that convey emotional meaning
    5. Converts text to lowercase for consistency
    
    Args:
        text (str): Raw comment text to be cleaned
    
    Returns:
        str: Cleaned and normalized text suitable for tokenization
    
    Examples:
        >>> clean_text("Check this https://example.com out!")
        "check this out!"
        
        >>> clean_text("I'm   so    HAPPY :)")
        "i'm so happy :)"
        
        >>> clean_text("Amazing product!!! ❤️")
        "amazing product!!! ❤️"
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs (http://, https://, www.)
    # Pattern matches: http://..., https://..., www....
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Preserve common emoticons by temporarily replacing them with placeholders
    # We need to do this before removing special characters
    # Format: (emoticon_string, placeholder, lowercase_version)
    emoticon_map = [
        (':)', '__SMILE__'),
        (':(', '__SAD__'),
        (':D', '__LAUGH__'),
        (':d', '__LAUGH_LOWER__'),
        (';)', '__WINK__'),
        (':P', '__TONGUE__'),
        (':p', '__TONGUE_LOWER__'),
        (':/', '__SKEPTICAL__'),
        (':o', '__SURPRISED_LOWER__'),
        (':O', '__SURPRISED__'),
        ('<3', '__HEART__'),
        (':|', '__NEUTRAL__'),
        (':-)', '__SMILE_NOSE__'),
        (':-(', '__SAD_NOSE__'),
        (':-D', '__LAUGH_NOSE__'),
        (':-d', '__LAUGH_NOSE_LOWER__'),
        (';-)', '__WINK_NOSE__'),
        (':-P', '__TONGUE_NOSE__'),
        (':-p', '__TONGUE_NOSE_LOWER__'),
        (':-/', '__SKEPTICAL_NOSE__'),
        (':-o', '__SURPRISED_NOSE_LOWER__'),
        (':-O', '__SURPRISED_NOSE__'),
        (':-|', '__NEUTRAL_NOSE__'),
        (":'(", '__CRYING__'),
        (":')", '__HAPPY_TEARS__'),
        ('XD', '__LAUGH_EYES__'),
        ('xD', '__LAUGH_EYES_LOWER__'),
        ('Xd', '__LAUGH_EYES_MIX1__'),
        ('xd', '__LAUGH_EYES_MIX2__'),
    ]
    
    # Replace emoticons with placeholders (order matters - longer patterns first)
    emoticon_map_sorted = sorted(emoticon_map, key=lambda x: len(x[0]), reverse=True)
    for emoticon, placeholder in emoticon_map_sorted:
        text = text.replace(emoticon, placeholder)
    
    # Note: Emoji (like ❤️, 😊, 😢, etc.) are Unicode characters and will be preserved
    # as they don't match the special character removal pattern below
    
    # Remove non-meaningful special characters
    # Keep: letters, numbers, spaces, basic punctuation (.,!?'), emoticon placeholders, and Unicode emoji
    # This pattern preserves alphanumeric, spaces, basic punctuation, underscores (for placeholders), and Unicode characters
    text = re.sub(r'[^\w\s.,!?\'\u0080-\uFFFF-]', '', text)
    
    # Normalize excessive whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Restore emoticons from placeholders (now in lowercase)
    # Create a mapping of lowercase placeholder to lowercase emoticon
    for emoticon, placeholder in emoticon_map_sorted:
        text = text.replace(placeholder.lower(), emoticon.lower())
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def load_data(file_path):
    """
    Loads emotion classification data from CSV file with validation.
    
    This function loads a CSV file containing comment text and emotion labels,
    performs comprehensive validation, and returns the data in a format suitable
    for model training and evaluation.
    
    Expected CSV format:
    - 'text' column: Comment text (string)
    - 16 emotion label columns: Binary values (0 or 1) for each emotion
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
    
    Returns:
        tuple: (texts, labels) where:
            - texts (list): List of comment text strings
            - labels (np.ndarray): Binary label matrix of shape (N, 16)
                                   where N is the number of samples
    
    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path
        ValueError: If the CSV is empty, missing required columns, or contains
                   invalid label values (non-binary)
    
    Examples:
        >>> texts, labels = load_data('data/sample_comments.csv')
        >>> print(f"Loaded {len(texts)} comments")
        Loaded 100 comments
        >>> print(f"Label shape: {labels.shape}")
        Label shape: (100, 16)
    
    Validation performed:
        1. File existence check
        2. Empty dataset check
        3. 'text' column presence
        4. All 16 emotion label columns presence
        5. Binary values (0 or 1) in all emotion columns
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{file_path}'. "
            f"Please ensure the file exists or check the file path."
        )
    
    # Load CSV file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"CSV file at '{file_path}' is empty. "
            f"Please provide a dataset with at least one sample."
        )
    except Exception as e:
        raise ValueError(
            f"Failed to read CSV file at '{file_path}': {str(e)}"
        )
    
    # Check if dataset is empty
    if len(df) == 0:
        raise ValueError(
            f"Dataset at '{file_path}' is empty. "
            f"Please provide data with at least one sample."
        )
    
    # Validate 'text' column exists
    if 'text' not in df.columns:
        raise ValueError(
            f"CSV file at '{file_path}' is missing required 'text' column. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Validate all 16 emotion label columns exist
    required_columns = ['text'] + Config.EMOTION_LABELS
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV file at '{file_path}' is missing required emotion label columns: {sorted(missing_columns)}. "
            f"Expected columns: {required_columns}"
        )
    
    # Validate that emotion columns contain only binary values (0 or 1)
    for label in Config.EMOTION_LABELS:
        # Check if all values are either 0 or 1
        if not df[label].isin([0, 1]).all():
            # Find invalid values for better error message
            invalid_values = df[label][~df[label].isin([0, 1])].unique()
            raise ValueError(
                f"Column '{label}' contains non-binary values: {invalid_values}. "
                f"All emotion labels must be 0 or 1."
            )
    
    # Extract texts as list of strings
    texts = df['text'].astype(str).tolist()
    
    # Extract labels as numpy array (N, 16)
    labels = df[Config.EMOTION_LABELS].values.astype(np.float32)
    
    return texts, labels


def compute_metrics(predictions, labels, threshold=0.5):
    """
    Computes comprehensive evaluation metrics for multi-label emotion classification.
    
    This function calculates various performance metrics to evaluate the quality of
    emotion predictions, including per-label metrics, micro/macro-averaged metrics,
    and hamming loss.
    
    Args:
        predictions (np.ndarray): Predicted probabilities of shape (N, 16)
                                  where N is the number of samples.
                                  Values should be in range [0, 1].
        labels (np.ndarray): True binary labels of shape (N, 16).
                            Values should be 0 or 1.
        threshold (float): Confidence threshold for converting probabilities
                          to binary predictions. Default is 0.5.
                          Predictions >= threshold are classified as 1,
                          otherwise 0.
    
    Returns:
        dict: Dictionary containing the following metrics:
            - 'per_label_precision': List of precision scores for each of 16 emotions
            - 'per_label_recall': List of recall scores for each of 16 emotions
            - 'per_label_f1': List of F1-scores for each of 16 emotions
            - 'micro_precision': Micro-averaged precision across all labels
            - 'micro_recall': Micro-averaged recall across all labels
            - 'micro_f1': Micro-averaged F1-score across all labels
            - 'macro_precision': Macro-averaged precision across all labels
            - 'macro_recall': Macro-averaged recall across all labels
            - 'macro_f1': Macro-averaged F1-score across all labels
            - 'hamming_loss': Hamming loss (fraction of incorrect predictions)
    
    Examples:
        >>> predictions = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
        >>> labels = np.array([[1, 0, 1], [0, 1, 0]])
        >>> metrics = compute_metrics(predictions, labels, threshold=0.5)
        >>> print(f"Micro F1: {metrics['micro_f1']:.3f}")
        Micro F1: 1.000
        >>> print(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
        Hamming Loss: 0.000
    
    Notes:
        - Micro-averaging aggregates contributions from all labels equally,
          giving more weight to frequent labels
        - Macro-averaging computes metrics for each label independently and
          takes the average, treating all labels equally
        - Hamming loss measures the fraction of incorrectly predicted labels
          (lower is better, 0 is perfect)
        - Per-label metrics help identify which emotions are predicted well
          and which need improvement
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        hamming_loss as sklearn_hamming_loss
    )
    
    # Convert predictions to binary using threshold
    # predictions >= threshold become 1, otherwise 0
    binary_predictions = (predictions >= threshold).astype(np.int32)
    
    # Ensure labels are integer type for sklearn metrics
    labels = labels.astype(np.int32)
    
    # Calculate per-label precision, recall, F1-score
    # average=None returns metrics for each label separately
    # zero_division=0 sets metric to 0 when there are no positive predictions/labels
    per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(
        labels,
        binary_predictions,
        average=None,
        zero_division=0
    )
    
    # Calculate micro-averaged metrics
    # Micro-averaging aggregates all TP, FP, FN across all labels
    # then computes metrics globally
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels,
        binary_predictions,
        average='micro',
        zero_division=0
    )
    
    # Calculate macro-averaged metrics
    # Macro-averaging computes metrics for each label independently
    # then takes the unweighted mean
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        binary_predictions,
        average='macro',
        zero_division=0
    )
    
    # Calculate hamming loss
    # Hamming loss is the fraction of labels that are incorrectly predicted
    # It's calculated as: (1 / (N * L)) * sum of (y_ij != y_hat_ij)
    # where N is number of samples and L is number of labels
    hamming = sklearn_hamming_loss(labels, binary_predictions)
    
    # Return comprehensive metrics dictionary
    return {
        # Per-label metrics (lists of length 16)
        'per_label_precision': per_label_precision.tolist(),
        'per_label_recall': per_label_recall.tolist(),
        'per_label_f1': per_label_f1.tolist(),
        
        # Micro-averaged metrics (single values)
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        
        # Macro-averaged metrics (single values)
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        
        # Hamming loss (single value)
        'hamming_loss': float(hamming)
    }


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plots and saves training and validation loss curves over epochs.
    
    This function creates a line plot showing how training and validation losses
    change over the course of training. This visualization helps identify:
    - Whether the model is learning (loss decreasing)
    - Whether the model is overfitting (train loss << val loss)
    - Whether training has converged (losses plateau)
    
    Args:
        train_losses (list of float): Training loss values for each epoch.
                                      Length should equal number of epochs.
        val_losses (list of float): Validation loss values for each epoch.
                                    Length should equal number of epochs.
        save_path (str): File path where the plot will be saved.
                        Should include filename and extension (e.g., 'plot.png').
                        Supported formats: png, jpg, pdf, svg.
    
    Returns:
        None: The function saves the plot to disk and closes the figure.
    
    Raises:
        ValueError: If train_losses and val_losses have different lengths.
        OSError: If the save_path directory doesn't exist or is not writable.
    
    Examples:
        >>> train_losses = [0.8, 0.6, 0.4, 0.3, 0.25]
        >>> val_losses = [0.75, 0.65, 0.55, 0.5, 0.48]
        >>> plot_training_curves(train_losses, val_losses, 'saved_model/training_curves.png')
        # Creates a plot showing both curves and saves to specified path
    
    Notes:
        - The function automatically closes the figure after saving to free memory
        - If save_path directory doesn't exist, it will be created
        - The plot includes a grid for easier reading of values
        - Training loss is plotted in blue, validation loss in orange
    """
    # Validate input lengths
    if len(train_losses) != len(val_losses):
        raise ValueError(
            f"train_losses and val_losses must have the same length. "
            f"Got train_losses: {len(train_losses)}, val_losses: {len(val_losses)}"
        )
    
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Create epoch numbers (1-indexed for better readability)
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    
    # Plot validation loss
    plt.plot(epochs, val_losses, 'orange', linestyle='-', marker='s', 
             label='Validation Loss', linewidth=2, markersize=6)
    
    # Add title and labels
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add legend
    plt.legend(loc='best', fontsize=11)
    
    # Add grid for easier reading
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure layout fits well
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    except Exception as e:
        raise OSError(f"Failed to save plot to '{save_path}': {str(e)}")
    finally:
        # Close the figure to free memory
        plt.close()


def save_model(model, tokenizer, save_dir, training_config=None):
    """
    Saves model checkpoint, tokenizer, and training configuration.
    
    This function persists the trained model state, tokenizer configuration,
    and optional training hyperparameters to disk for later inference or
    continued training.
    
    Saved files:
    - pytorch_model.bin: Model state_dict (weights and biases)
    - tokenizer files: Tokenizer configuration and vocabulary
    - training_config.json: Training hyperparameters (if provided)
    
    Args:
        model: BERTEmotionClassifier instance to save.
               Must have a state_dict() method.
        tokenizer: BERT tokenizer instance to save.
                   Must have a save_pretrained() method.
        save_dir (str): Directory path where files will be saved.
                       Will be created if it doesn't exist.
        training_config (dict, optional): Dictionary containing training
                                         hyperparameters (learning rate,
                                         batch size, epochs, etc.).
                                         If None, no config file is saved.
    
    Returns:
        None: Files are saved to disk.
    
    Raises:
        OSError: If save_dir cannot be created or is not writable.
        AttributeError: If model or tokenizer don't have required methods.
    
    Examples:
        >>> from transformers import BertTokenizer
        >>> model = BERTEmotionClassifier(num_labels=16)
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> config = {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 5}
        >>> save_model(model, tokenizer, 'saved_model/', training_config=config)
        Model saved to: saved_model/pytorch_model.bin
        Tokenizer saved to: saved_model/
        Training config saved to: saved_model/training_config.json
    
    Notes:
        - Creates save_dir if it doesn't exist
        - Overwrites existing files in save_dir
        - Model is saved in PyTorch format (not Hugging Face format)
        - Tokenizer is saved using Hugging Face's save_pretrained() method
    """
    import torch
    import json
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")
        except Exception as e:
            raise OSError(f"Failed to create save directory '{save_dir}': {str(e)}")
    
    # Validate that save_dir is writable
    if not os.access(save_dir, os.W_OK):
        raise OSError(f"Save directory '{save_dir}' is not writable.")
    
    # Save model state_dict
    model_path = os.path.join(save_dir, 'pytorch_model.bin')
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    except AttributeError:
        raise AttributeError(
            "Model must have a state_dict() method. "
            "Ensure you're passing a PyTorch nn.Module instance."
        )
    except Exception as e:
        raise OSError(f"Failed to save model to '{model_path}': {str(e)}")
    
    # Save tokenizer
    try:
        tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer saved to: {save_dir}")
    except AttributeError:
        raise AttributeError(
            "Tokenizer must have a save_pretrained() method. "
            "Ensure you're passing a Hugging Face tokenizer instance."
        )
    except Exception as e:
        raise OSError(f"Failed to save tokenizer to '{save_dir}': {str(e)}")
    
    # Save training configuration if provided
    if training_config is not None:
        config_path = os.path.join(save_dir, 'training_config.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(training_config, f, indent=4)
            print(f"Training config saved to: {config_path}")
        except Exception as e:
            raise OSError(f"Failed to save training config to '{config_path}': {str(e)}")


def load_model(save_dir, device='cpu'):
    """
    Loads saved model checkpoint and tokenizer.
    
    This function restores a previously saved model and tokenizer from disk,
    enabling inference or continued training. The model is moved to the
    specified device (CPU or GPU).
    
    Expected files in save_dir:
    - pytorch_model.bin: Model state_dict
    - tokenizer files: Tokenizer configuration and vocabulary
    
    Args:
        save_dir (str): Directory path containing saved model files.
                       Must contain pytorch_model.bin and tokenizer files.
        device (str or torch.device): Device to load model onto.
                                     Options: 'cpu', 'cuda', or torch.device object.
                                     Default is 'cpu'.
    
    Returns:
        tuple: (model, tokenizer) where:
            - model: BERTEmotionClassifier instance with loaded weights,
                    moved to specified device and set to eval mode
            - tokenizer: Loaded BERT tokenizer instance
    
    Raises:
        FileNotFoundError: If save_dir doesn't exist or required files are missing.
        RuntimeError: If model loading fails (corrupted checkpoint, architecture mismatch).
        ValueError: If device is invalid.
    
    Examples:
        >>> model, tokenizer = load_model('saved_model/', device='cuda')
        Model loaded from: saved_model/pytorch_model.bin
        Tokenizer loaded from: saved_model/
        Model moved to device: cuda
        >>> # Now ready for inference
        >>> predictions = model(input_ids, attention_mask)
    
    Notes:
        - Model architecture must match the saved checkpoint
        - Model is automatically set to evaluation mode (model.eval())
        - If GPU is specified but not available, an error is raised
        - Tokenizer is loaded using Hugging Face's from_pretrained() method
    """
    import torch
    from transformers import BertTokenizer
    # Note: Import model class when model.py is implemented
    # For now, we'll add a placeholder comment
    
    # Validate save_dir exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(
            f"Save directory '{save_dir}' does not exist. "
            f"Please ensure the model has been trained and saved first."
        )
    
    # Validate model checkpoint exists
    model_path = os.path.join(save_dir, 'pytorch_model.bin')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            f"Please train the model first by running: python train.py"
        )
    
    # Validate device
    if isinstance(device, str):
        if device not in ['cpu', 'cuda']:
            raise ValueError(
                f"Invalid device '{device}'. Must be 'cpu' or 'cuda'."
            )
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available. "
                "Please use device='cpu' or install CUDA support."
            )
        device = torch.device(device)
    
    # Load tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(save_dir)
        print(f"Tokenizer loaded from: {save_dir}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load tokenizer from '{save_dir}': {str(e)}. "
            f"Ensure tokenizer files exist in the directory."
        )
    
    # Load model
    # TODO: Import BERTEmotionClassifier from model.py when it's implemented
    # For now, we'll add a placeholder that will work once model.py exists
    try:
        # Import model classes
        from model import BERTEmotionClassifier
        try:
            from model_phobert import PhoBERTEmotionClassifier, HybridEmotionClassifier
        except ImportError:
            PhoBERTEmotionClassifier = None
            HybridEmotionClassifier = None
        
        # Load state_dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Determine model architecture from state_dict shape
        if 'classifier.weight' in state_dict:
            weight_shape = state_dict['classifier.weight'].shape
            # Hybrid mode: 768 (CLS) + 256*2 (BiLSTM) = 1280
            if weight_shape[1] == 1280 and HybridEmotionClassifier:
                model = HybridEmotionClassifier(num_labels=Config.NUM_LABELS, dropout_rate=Config.DROPOUT_RATE)
            # BiLSTM+Attention mode: 256*2 = 512
            elif weight_shape[1] == 512 and PhoBERTEmotionClassifier:
                model = PhoBERTEmotionClassifier(num_labels=Config.NUM_LABELS, dropout_rate=Config.DROPOUT_RATE)
            else:
                # Base model: 768
                model = BERTEmotionClassifier(num_labels=Config.NUM_LABELS, dropout_rate=Config.DROPOUT_RATE)
        else:
            model = BERTEmotionClassifier(num_labels=Config.NUM_LABELS, dropout_rate=Config.DROPOUT_RATE)
        
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {model_path} (Type: {model.__class__.__name__})")
        
    except ImportError:
        raise ImportError(
            "Could not import BERTEmotionClassifier from model.py. "
            "Please ensure model.py exists and contains the BERTEmotionClassifier class."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{model_path}': {str(e)}. "
            f"The checkpoint may be corrupted or incompatible. "
            f"Please retrain the model."
        )
    
    # Move model to specified device
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer
