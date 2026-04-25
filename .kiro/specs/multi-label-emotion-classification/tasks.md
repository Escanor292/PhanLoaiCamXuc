# Implementation Plan: Multi-label Emotion Classification

## Overview

This implementation plan breaks down the Multi-label Emotion Classification system into discrete coding tasks. The system uses BERT (bert-base-uncased) for analyzing Vietnamese and English comments to predict up to 16 emotions simultaneously. The implementation follows a modular architecture with clear separation between configuration, data handling, model definition, training, and inference.

**Implementation Language**: Python 3.9+

**Key Technologies**: PyTorch, Hugging Face Transformers, BERT

## Tasks

- [x] 1. Set up project structure and configuration
  - Create directory structure: `data/`, `saved_model/`, root files
  - Create `.gitignore` file to exclude model checkpoints and data files
  - Create `requirements.txt` with all dependencies (torch, transformers, pandas, numpy, scikit-learn, matplotlib, tqdm)
  - Create `config.py` with Config class containing all hyperparameters, paths, and emotion labels list
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 10.1, 10.2_

- [ ] 2. Implement utility functions module
  - [x] 2.1 Create `utils.py` with text preprocessing functions
    - Implement `clean_text()` function to remove URLs, normalize whitespace, preserve emoticons, convert to lowercase
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [x]* 2.2 Write unit tests for text preprocessing
    - Test URL removal, whitespace normalization, emoticon preservation, lowercase conversion
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 2.3 Implement data loading functions in `utils.py`
    - Implement `load_data()` function to load CSV files with validation
    - Validate CSV contains 'text' column and 16 emotion label columns
    - Implement error handling for missing files, empty datasets, invalid label values
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 14.1, 14.2_
  
  - [x] 2.4 Implement evaluation metrics functions in `utils.py`
    - Implement `compute_metrics()` function to calculate precision, recall, F1-score per label
    - Calculate micro-averaged and macro-averaged F1-scores
    - Calculate hamming loss
    - Apply prediction threshold to convert probabilities to binary predictions
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  
  - [x] 2.5 Write unit tests for metrics calculation
    - Test metrics with perfect predictions, all wrong predictions, and mixed cases
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 2.6 Implement visualization functions in `utils.py`
    - Implement `plot_training_curves()` function to plot training and validation loss over epochs
    - Save plots to file in saved_model directory
    - _Requirements: 6.6_
  
  - [x] 2.7 Implement model persistence functions in `utils.py`
    - Implement `save_model()` function to save model checkpoint, tokenizer, and training config
    - Implement `load_model()` function to load saved model and tokenizer
    - Include error handling for missing checkpoints
    - _Requirements: 5.6, 14.2_

- [ ] 3. Implement dataset module
  - [x] 3.1 Create `dataset.py` with EmotionDataset class
    - Implement PyTorch Dataset class that accepts texts, labels, and tokenizer
    - Implement `__init__()`, `__len__()`, and `__getitem__()` methods
    - Tokenize text on-the-fly in `__getitem__()` with truncation and padding
    - Return dictionary with input_ids, attention_mask, and labels as tensors
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x]* 3.2 Write unit tests for EmotionDataset
    - Test dataset length, getitem format, tokenization output shapes
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 4. Implement model architecture
  - [x] 4.1 Create `model.py` with BERTEmotionClassifier class
    - Implement PyTorch nn.Module class
    - Load pre-trained bert-base-uncased model
    - Add dropout layer with configurable rate
    - Add linear classification head (768 → 16)
    - Implement forward() method that returns logits (no sigmoid)
    - Include error handling for BERT download failures
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 14.2_
  
  - [x]* 4.2 Write unit tests for model architecture
    - Test model initialization, forward pass output shapes, model save/load
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 5. Generate sample data
  - [x] 5.1 Create sample data generation script or function
    - Generate at least 100 sample comments in English and Vietnamese
    - Assign realistic multi-label emotion combinations
    - Ensure each comment has at least one emotion label
    - Save to `data/sample_comments.csv` with proper format
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [x]* 5.2 Validate sample data format
    - Load generated CSV and verify it has correct columns and format
    - _Requirements: 2.1, 2.4_

- [x] 6. Checkpoint - Verify core components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement training pipeline
  - [x] 7.1 Create `train.py` with training functions
    - Implement `train_epoch()` function for one epoch of training
    - Use BCEWithLogitsLoss as loss function
    - Use AdamW optimizer with configurable learning rate
    - Implement gradient clipping (max norm 1.0)
    - Display training progress with tqdm progress bars
    - _Requirements: 6.1, 6.2, 6.7, 14.5_
  
  - [x] 7.2 Implement evaluation function in `train.py`
    - Implement `evaluate()` function to compute loss on validation/test sets
    - Return average loss, predictions, and true labels
    - _Requirements: 6.4, 7.1, 7.2, 7.3, 7.4_
  
  - [x] 7.3 Implement main training loop in `train.py`
    - Load data from CSV and split into train/val/test (70/15/15)
    - Create EmotionDataset instances and DataLoaders
    - Initialize model, optimizer, and loss function
    - Set random seeds for reproducibility
    - Detect and use GPU if available, fallback to CPU with warning
    - Train for configurable number of epochs
    - Evaluate on validation set after each epoch
    - Save model checkpoint with lowest validation loss
    - Generate and save training curves plot
    - Log all hyperparameters and final metrics
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.8, 1.5, 15.1, 15.2, 15.3, 15.4, 15.5, 14.3_
  
  - [x]* 7.4 Write integration test for training pipeline
    - Test end-to-end training with minimal sample data for 1 epoch
    - Verify model checkpoint is saved
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Implement prediction/inference module
  - [x] 8.1 Create `predict.py` with prediction functions
    - Implement `predict_emotions()` function for single comment prediction
    - Load saved model checkpoint and tokenizer
    - Preprocess and tokenize input text
    - Generate logits and apply sigmoid to get confidence scores
    - Apply prediction threshold to determine predicted emotions
    - Return dictionary with predicted emotions list and all confidence scores
    - Handle case when no emotions exceed threshold
    - Include input validation and error handling
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 14.4_
  
  - [x] 8.2 Implement batch prediction support in `predict.py`
    - Extend prediction function to handle multiple comments efficiently
    - _Requirements: 8.6_
  
  - [x] 8.3 Create interactive command-line interface in `predict.py`
    - Implement main() function that prompts user for input
    - Display predicted emotions and confidence scores in readable format
    - Support multiple predictions without reloading model
    - Handle errors gracefully with clear messages
    - _Requirements: 13.2, 13.3, 13.4, 13.5_
  
  - [x]* 8.4 Write integration test for prediction pipeline
    - Test end-to-end prediction with sample comments
    - Verify output format and score ranges
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 9. Checkpoint - Verify training and prediction work end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Create comprehensive documentation
  - [x] 10.1 Create `README.md` with project documentation
    - Write project description explaining multi-label emotion classification
    - List and explain all 16 emotion labels
    - Provide installation instructions with requirements.txt
    - Provide step-by-step usage examples for training (python train.py)
    - Provide step-by-step usage examples for prediction (python predict.py)
    - Explain evaluation metrics (precision, recall, F1-score, micro-F1, macro-F1, hamming loss)
    - Include expected directory structure
    - Add troubleshooting section for common errors
    - _Requirements: 10.3, 10.4, 10.5, 10.6_
  
  - [x] 10.2 Add docstrings to all functions and classes
    - Add comprehensive docstrings explaining purpose, parameters, and return values
    - Add inline comments for complex logic
    - _Requirements: 10.7, 10.8_

- [ ] 11. Final validation and testing
  - [x]* 11.1 Run complete test suite
    - Execute all unit tests and integration tests
    - Verify test coverage is adequate
    - _Requirements: All testing requirements_
  
  - [x]* 11.2 Perform manual testing with diverse examples
    - Test with positive emotion comments
    - Test with negative emotion comments
    - Test with mixed emotion comments
    - Test with neutral/calm comments
    - Test with Vietnamese text
    - Document any edge cases or limitations
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  
  - [x] 11.3 Verify error handling
    - Test with missing data files
    - Test with missing model checkpoint
    - Test with invalid CSV format
    - Test with empty input text
    - Verify all error messages are clear and helpful
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 12. Final checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The implementation uses Python with PyTorch and Hugging Face Transformers
- BERT model (bert-base-uncased) will be downloaded automatically on first run
- GPU is recommended for training but system will fallback to CPU if unavailable
- Sample data generation allows demonstration without requiring a real dataset
- All configuration is centralized in config.py for easy experimentation
- The system includes comprehensive error handling and user-friendly messages
- Testing strategy focuses on unit tests, integration tests, and manual validation (no property-based tests as explained in design document)
