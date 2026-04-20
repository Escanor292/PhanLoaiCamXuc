# Requirements Document

## Introduction

This document specifies requirements for a Multi-label Emotion Classification system that analyzes Vietnamese and English comment text and predicts multiple emotions simultaneously. The system uses BERT-based deep learning to identify up to 16 different emotions present in user comments, enabling nuanced sentiment analysis for applications such as social media monitoring, customer feedback analysis, and content moderation.

## Glossary

- **System**: The Multi-label Emotion Classification system
- **Comment**: User-generated text input (review, feedback, or social media comment)
- **Emotion_Label**: One of 16 predefined emotion categories (joy, trust, fear, surprise, sadness, disgust, anger, anticipation, love, worried, disappointed, proud, embarrassed, jealous, calm, excited)
- **Multi_Label_Prediction**: Classification output where multiple Emotion_Labels can be assigned to a single Comment
- **BERT_Model**: Bidirectional Encoder Representations from Transformers pre-trained language model
- **Confidence_Score**: Probability value between 0 and 1 indicating likelihood of an Emotion_Label
- **Prediction_Threshold**: Minimum Confidence_Score (default 0.5) required to assign an Emotion_Label
- **Training_Dataset**: CSV file containing Comments with ground-truth Emotion_Label annotations
- **Model_Checkpoint**: Saved state of trained BERT_Model parameters
- **Text_Preprocessor**: Component that cleans and normalizes Comment text
- **Tokenizer**: BERT tokenizer that converts text into model input format
- **Classification_Head**: Fully connected neural network layer that outputs 16 Emotion_Label predictions
- **BCEWithLogitsLoss**: Binary Cross Entropy loss function for multi-label classification
- **Evaluation_Metrics**: Performance measures including precision, recall, F1-score, micro-F1, macro-F1, and hamming loss

## Requirements

### Requirement 1: Data Loading and Management

**User Story:** As a data scientist, I want to load comment data from CSV files, so that I can train and evaluate the emotion classification model.

#### Acceptance Criteria

1. WHEN a CSV file path is provided, THE System SHALL load the file into a structured data format
2. THE System SHALL validate that the CSV contains a text column for Comments
3. THE System SHALL validate that the CSV contains 16 binary columns corresponding to all Emotion_Labels
4. IF the CSV file is missing or corrupted, THEN THE System SHALL return a descriptive error message
5. THE System SHALL split the Training_Dataset into training, validation, and test sets with configurable ratios

### Requirement 2: Sample Data Generation

**User Story:** As a developer, I want to generate sample comment data, so that I can demonstrate the system without requiring a real dataset.

#### Acceptance Criteria

1. THE System SHALL generate a sample CSV file with at least 100 Comments
2. THE System SHALL include Comments in both English and Vietnamese languages
3. WHEN generating sample data, THE System SHALL assign realistic multi-label Emotion_Label combinations
4. THE System SHALL save the generated sample data to the data/ directory as sample_comments.csv
5. FOR ALL generated Comments, at least one Emotion_Label SHALL be assigned

### Requirement 3: Text Preprocessing

**User Story:** As a machine learning engineer, I want to preprocess comment text, so that the model receives clean and normalized input.

#### Acceptance Criteria

1. WHEN a Comment contains URLs, THE Text_Preprocessor SHALL remove or normalize them
2. WHEN a Comment contains special characters that are not linguistically meaningful, THE Text_Preprocessor SHALL remove them
3. WHEN a Comment contains excessive whitespace, THE Text_Preprocessor SHALL normalize it to single spaces
4. THE Text_Preprocessor SHALL preserve emoticons and emoji that convey emotional meaning
5. THE Text_Preprocessor SHALL convert text to lowercase for consistency
6. WHEN preprocessing is complete, THE Text_Preprocessor SHALL return cleaned text suitable for tokenization

### Requirement 4: BERT Tokenization

**User Story:** As a machine learning engineer, I want to tokenize preprocessed text using BERT tokenizer, so that comments can be converted into model input format.

#### Acceptance Criteria

1. THE System SHALL use bert-base-uncased Tokenizer for text tokenization
2. WHEN tokenizing a Comment, THE Tokenizer SHALL produce input_ids, attention_mask, and token_type_ids
3. THE Tokenizer SHALL truncate Comments longer than 512 tokens to fit BERT input constraints
4. THE Tokenizer SHALL pad Comments shorter than the maximum length with padding tokens
5. WHEN tokenizing batches of Comments, THE Tokenizer SHALL return tensors compatible with PyTorch

### Requirement 5: Model Architecture

**User Story:** As a machine learning engineer, I want to define a BERT-based multi-label classification model, so that I can train it to predict emotions.

#### Acceptance Criteria

1. THE System SHALL use bert-base-uncased as the base BERT_Model
2. THE BERT_Model SHALL be followed by a dropout layer with configurable dropout rate
3. THE Classification_Head SHALL be a fully connected layer with 16 output units corresponding to Emotion_Labels
4. THE System SHALL apply sigmoid activation to Classification_Head outputs to produce Confidence_Scores
5. THE System SHALL support loading pre-trained BERT_Model weights from Hugging Face
6. THE System SHALL support saving and loading Model_Checkpoint files

### Requirement 6: Model Training

**User Story:** As a machine learning engineer, I want to train the emotion classification model, so that it learns to predict emotions from comments.

#### Acceptance Criteria

1. THE System SHALL use BCEWithLogitsLoss as the loss function for multi-label classification
2. THE System SHALL use AdamW optimizer with configurable learning rate
3. THE System SHALL train for a configurable number of epochs (default 3-5 epochs)
4. WHEN training, THE System SHALL compute loss on both training and validation sets each epoch
5. THE System SHALL save the Model_Checkpoint with the lowest validation loss
6. WHEN training completes, THE System SHALL generate a loss curve plot showing training and validation loss over epochs
7. THE System SHALL display training progress including current epoch, batch number, and loss values
8. IF GPU is available, THE System SHALL use GPU acceleration for training

### Requirement 7: Model Evaluation

**User Story:** As a data scientist, I want to evaluate model performance with comprehensive metrics, so that I can assess the quality of emotion predictions.

#### Acceptance Criteria

1. WHEN evaluating on test data, THE System SHALL compute per-label precision, recall, and F1-score
2. THE System SHALL compute micro-averaged F1-score across all Emotion_Labels
3. THE System SHALL compute macro-averaged F1-score across all Emotion_Labels
4. THE System SHALL compute hamming loss to measure prediction accuracy
5. THE System SHALL generate a classification report displaying all Evaluation_Metrics
6. THE System SHALL apply the Prediction_Threshold to convert Confidence_Scores to binary predictions
7. WHEN evaluation completes, THE System SHALL save the classification report to a text file

### Requirement 8: Emotion Prediction

**User Story:** As an end user, I want to input a comment and receive emotion predictions, so that I can understand the emotional content of the text.

#### Acceptance Criteria

1. WHEN a new Comment is provided, THE System SHALL preprocess and tokenize the text
2. THE System SHALL load the saved Model_Checkpoint for inference
3. THE System SHALL generate Confidence_Scores for all 16 Emotion_Labels
4. THE System SHALL apply the Prediction_Threshold to determine which Emotion_Labels are present
5. THE System SHALL display both the Confidence_Scores and the selected Emotion_Labels
6. THE System SHALL support batch prediction for multiple Comments
7. WHEN no emotions exceed the Prediction_Threshold, THE System SHALL indicate that no strong emotions were detected

### Requirement 9: Configuration Management

**User Story:** As a developer, I want to manage system configuration in a centralized file, so that I can easily adjust hyperparameters and settings.

#### Acceptance Criteria

1. THE System SHALL define all configurable parameters in config.py
2. THE System SHALL include configuration for model architecture (dropout rate, hidden dimensions)
3. THE System SHALL include configuration for training (learning rate, batch size, epochs)
4. THE System SHALL include configuration for data paths (dataset location, model save directory)
5. THE System SHALL include configuration for the 16 Emotion_Labels as a constant list
6. THE System SHALL include configuration for Prediction_Threshold
7. WHEN configuration values are modified, THE System SHALL use the updated values without code changes

### Requirement 10: Project Structure and Documentation

**User Story:** As a developer, I want clear project organization and documentation, so that I can easily understand, install, and run the system.

#### Acceptance Criteria

1. THE System SHALL organize code into the following structure: data/, saved_model/, config.py, dataset.py, model.py, train.py, predict.py, utils.py, requirements.txt, README.md
2. THE System SHALL include a requirements.txt file listing all Python dependencies with versions
3. THE System SHALL include a README.md file with project description, installation instructions, and usage examples
4. THE README.md SHALL explain the multi-label classification problem and the 16 Emotion_Labels
5. THE README.md SHALL provide step-by-step instructions for training and prediction
6. THE README.md SHALL explain all Evaluation_Metrics and their interpretation
7. WHEN code files contain functions or classes, THE System SHALL include docstrings explaining purpose and parameters
8. THE System SHALL include inline comments for complex logic to aid understanding

### Requirement 11: Dataset Module

**User Story:** As a machine learning engineer, I want a PyTorch Dataset class for emotion classification, so that I can efficiently load and batch data during training.

#### Acceptance Criteria

1. THE System SHALL implement a PyTorch Dataset class in dataset.py
2. WHEN initialized, THE Dataset SHALL accept Comments, Emotion_Label arrays, and Tokenizer
3. THE Dataset SHALL tokenize Comments on-the-fly during data loading
4. THE Dataset SHALL return dictionaries containing input_ids, attention_mask, and labels for each sample
5. THE Dataset SHALL be compatible with PyTorch DataLoader for batching and shuffling

### Requirement 12: Utility Functions

**User Story:** As a developer, I want reusable utility functions, so that I can avoid code duplication and maintain consistency.

#### Acceptance Criteria

1. THE System SHALL implement text cleaning functions in utils.py
2. THE System SHALL implement metric calculation functions in utils.py
3. THE System SHALL implement visualization functions for plotting training curves in utils.py
4. THE System SHALL implement model saving and loading helper functions in utils.py
5. WHEN utility functions are called, THE System SHALL handle edge cases and invalid inputs gracefully

### Requirement 13: Command-Line Interface

**User Story:** As a user, I want to run training and prediction from the command line, so that I can easily execute the system without modifying code.

#### Acceptance Criteria

1. WHEN train.py is executed, THE System SHALL train the model using the Training_Dataset
2. WHEN predict.py is executed, THE System SHALL prompt the user to input a Comment
3. WHEN predict.py receives a Comment, THE System SHALL display predicted emotions and Confidence_Scores
4. THE System SHALL support running predict.py multiple times without reloading the model
5. WHEN training or prediction encounters an error, THE System SHALL display a clear error message

### Requirement 14: Error Handling and Validation

**User Story:** As a developer, I want robust error handling, so that the system fails gracefully with informative messages.

#### Acceptance Criteria

1. IF the Training_Dataset file is not found, THEN THE System SHALL display an error message with the expected file path
2. IF the Model_Checkpoint is not found during prediction, THEN THE System SHALL display an error message instructing the user to train first
3. IF GPU is requested but not available, THEN THE System SHALL fall back to CPU with a warning message
4. IF input Comment is empty or invalid, THEN THE System SHALL display an error message and prompt for valid input
5. WHEN any file I/O operation fails, THE System SHALL display a descriptive error message including the file path

### Requirement 15: Reproducibility

**User Story:** As a researcher, I want reproducible results, so that I can verify model performance and compare experiments.

#### Acceptance Criteria

1. THE System SHALL set random seeds for Python, NumPy, and PyTorch
2. THE System SHALL document the random seed value in config.py
3. WHEN the same Training_Dataset and configuration are used, THE System SHALL produce consistent results across runs
4. THE System SHALL log all hyperparameters used during training
5. THE System SHALL save training configuration alongside the Model_Checkpoint
