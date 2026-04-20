# Multi-label Emotion Classification

A BERT-based deep learning system that analyzes Vietnamese and English comment text to predict multiple emotions simultaneously. The system can identify up to 16 different emotions in user-generated content, enabling nuanced sentiment analysis for social media monitoring, customer feedback analysis, and content moderation.

## Overview

Unlike traditional single-label sentiment analysis, this multi-label approach recognizes that human emotions are complex and often overlapping. A single comment can express multiple emotions simultaneously (e.g., "joy" and "excited" together), providing richer emotional understanding of text.

### Key Features

- **Multi-label Classification**: Predicts multiple emotions per comment
- **BERT-based Architecture**: Leverages pre-trained `bert-base-uncased` for contextual understanding
- **16 Emotion Labels**: Comprehensive emotion coverage
- **Bilingual Support**: Handles both English and Vietnamese text
- **Easy-to-use CLI**: Interactive command-line interface for predictions
- **Comprehensive Metrics**: Detailed evaluation with precision, recall, F1-scores, and more

## Emotion Labels

The system can detect 16 distinct emotions:

| Emotion | Description | Example |
|---------|-------------|---------|
| **joy** | Happiness, delight, pleasure | "This is wonderful!" |
| **trust** | Confidence, reliability, faith | "I believe in this product" |
| **fear** | Anxiety, worry, apprehension | "I'm scared this won't work" |
| **surprise** | Astonishment, amazement, shock | "Wow, I didn't expect this!" |
| **sadness** | Sorrow, grief, unhappiness | "This makes me feel down" |
| **disgust** | Revulsion, distaste, aversion | "This is repulsive" |
| **anger** | Rage, frustration, irritation | "I'm furious about this!" |
| **anticipation** | Expectation, hope, looking forward | "Can't wait to try this" |
| **love** | Affection, adoration, fondness | "I absolutely love this!" |
| **worried** | Concern, unease, nervousness | "I'm concerned about quality" |
| **disappointed** | Let down, dissatisfied, discouraged | "This didn't meet expectations" |
| **proud** | Pride, satisfaction, accomplishment | "I'm proud of this achievement" |
| **embarrassed** | Shame, awkwardness, discomfort | "This is so awkward" |
| **jealous** | Envy, resentment, covetousness | "I wish I had that" |
| **calm** | Peaceful, relaxed, tranquil | "Everything is fine" |
| **excited** | Enthusiasm, eagerness, thrill | "This is so exciting!" |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd multi-label-emotion-classification

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following packages:

- **torch** (>=2.0.0): PyTorch deep learning framework
- **transformers** (>=4.30.0): Hugging Face transformers library for BERT
- **pandas** (>=2.0.0): Data manipulation and CSV handling
- **numpy** (>=1.24.0): Numerical operations
- **scikit-learn** (>=1.3.0): Evaluation metrics
- **matplotlib** (>=3.7.0): Training curve visualization
- **tqdm** (>=4.65.0): Progress bars

### Step 3: Verify Installation

```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Usage

### Training the Model

#### Step 1: Prepare Your Dataset

Create a CSV file in the `data/` directory with the following format:

```csv
text,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,love,worried,disappointed,proud,embarrassed,jealous,calm,excited
"This product is amazing!",1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1
"I'm really disappointed",0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0
```

**CSV Requirements:**
- **text** column: Comment text (string)
- **16 emotion columns**: Binary labels (0 or 1) for each emotion

**Or use the sample data generator:**

```bash
python generate_sample_data.py
```

This creates `data/sample_comments.csv` with 100 sample comments.

#### Step 2: Configure Training Parameters (Optional)

Edit `config.py` to adjust hyperparameters:

```python
# Training Configuration
LEARNING_RATE = 2e-5      # Learning rate for optimizer
BATCH_SIZE = 16           # Batch size for training
NUM_EPOCHS = 5            # Number of training epochs
MAX_LENGTH = 512          # Maximum token sequence length

# Model Configuration
DROPOUT_RATE = 0.3        # Dropout rate for regularization

# Prediction Configuration
PREDICTION_THRESHOLD = 0.5  # Confidence threshold for predictions
```

#### Step 3: Run Training

```bash
python train.py
```

**Training Process:**

1. Loads data from `data/sample_comments.csv`
2. Splits data into train (70%), validation (15%), test (15%)
3. Initializes BERT model with classification head
4. Trains for specified number of epochs
5. Evaluates on validation set after each epoch
6. Saves best model checkpoint to `saved_model/`
7. Generates training curves plot

**Expected Output:**

```
======================================================================
Multi-label Emotion Classification - Training Pipeline
======================================================================

[1/10] Setting random seeds for reproducibility...
Random seed: 42

[2/10] Detecting device...
Device: cuda (NVIDIA GeForce RTX 3080)

[3/10] Loading data...
Loaded 1000 samples
Label shape: (1000, 16)

[4/10] Splitting data...
Train: 700 samples
Validation: 150 samples
Test: 150 samples

[5/10] Creating datasets and dataloaders...
Train batches: 44
Validation batches: 10
Test batches: 10

[6/10] Initializing model...
Model: BERTEmotionClassifier
Parameters: 109,498,384
Trainable parameters: 109,498,384
Optimizer: AdamW (lr=2e-05)
Loss function: BCEWithLogitsLoss

[7/10] Starting training...
Epochs: 5
Batch size: 16
======================================================================

Epoch 1/5
----------------------------------------------------------------------
Training: 100%|████████████████████| 44/44 [00:45<00:00]
Evaluating: 100%|██████████████████| 10/10 [00:03<00:00]

Epoch 1 Summary:
  Train Loss: 0.4523
  Val Loss:   0.3845
  Val Micro-F1: 0.6234
  Val Macro-F1: 0.5891
  Val Hamming Loss: 0.0823

  ✓ New best validation loss! Saving model...

...

Training complete!
```

**Training Time Estimates:**
- **CPU**: ~10-15 minutes per epoch (1000 samples)
- **GPU (RTX 3080)**: ~2-3 minutes per epoch (1000 samples)

### Making Predictions

#### Interactive Mode

```bash
python predict.py
```

**Interactive Session:**

```
======================================================================
MULTI-LABEL EMOTION CLASSIFICATION SYSTEM
======================================================================

This system predicts emotions from comment text using BERT.
Supported emotions: joy, trust, fear, surprise, sadness, disgust, anger, anticipation,
                    love, worried, disappointed, proud, embarrassed, jealous, calm, excited

Commands:
  - Enter text to predict emotions
  - Type 'quit' or 'exit' to exit
  - Type 'help' for more information
======================================================================

Loading model from: saved_model/
This may take a moment...
✓ Model loaded successfully on device: cuda
✓ Prediction threshold: 0.5

======================================================================
Ready for predictions!
======================================================================

Enter a comment (or 'quit' to exit): I love this product! It exceeded my expectations!

======================================================================
EMOTION PREDICTION RESULTS
======================================================================

Input Text: "I love this product! It exceeded my expectations!"

----------------------------------------------------------------------

Predicted Emotions (4):
----------------------------------------------------------------------
  joy             [████████████████████████████████████░░░░] 0.920
  love            [██████████████████████████████░░░░░░░░░░] 0.850
  excited         [████████████████████████░░░░░░░░░░░░░░░] 0.720
  surprise        [██████████████████░░░░░░░░░░░░░░░░░░░░░] 0.650

======================================================================

Show all emotion scores? (y/n): n

Enter a comment (or 'quit' to exit): quit

Thank you for using the Emotion Classification System!
Goodbye!
```

#### Programmatic Usage

```python
from predict import predict_emotions
from utils import load_model
from config import Config

# Load model once
model, tokenizer = load_model(Config.MODEL_SAVE_DIR, Config.DEVICE)

# Single prediction
text = "This is amazing!"
result = predict_emotions(text, model, tokenizer, Config.DEVICE)

print("Predicted emotions:", result['emotions'])
print("Confidence scores:", result['scores'])

# Batch prediction
from predict import predict_emotions_batch

texts = [
    "I love this!",
    "This is terrible.",
    "Feeling calm today."
]
results = predict_emotions_batch(texts, model, tokenizer, Config.DEVICE)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['emotions']}")
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics to assess model performance:

### Per-Label Metrics

Calculated independently for each of the 16 emotions:

- **Precision**: Of all predictions for this emotion, how many were correct?
  - Formula: `TP / (TP + FP)`
  - High precision = few false positives

- **Recall**: Of all actual instances of this emotion, how many did we detect?
  - Formula: `TP / (TP + FN)`
  - High recall = few false negatives

- **F1-Score**: Harmonic mean of precision and recall
  - Formula: `2 × (Precision × Recall) / (Precision + Recall)`
  - Balances precision and recall

### Aggregate Metrics

#### Micro-F1 Score

Calculates metrics globally by counting total true positives, false positives, and false negatives across all emotions.

- **When to use**: Overall performance across all predictions
- **Characteristic**: Dominated by frequent emotions
- **Interpretation**: Higher is better (0.0 to 1.0)

```
Micro-Precision = Σ TP / (Σ TP + Σ FP)
Micro-Recall = Σ TP / (Σ TP + Σ FN)
Micro-F1 = 2 × (Micro-Precision × Micro-Recall) / (Micro-Precision + Micro-Recall)
```

#### Macro-F1 Score

Calculates metrics for each emotion independently, then averages them.

- **When to use**: Balanced evaluation across all emotions
- **Characteristic**: Treats all emotions equally (not dominated by frequent ones)
- **Interpretation**: Higher is better (0.0 to 1.0)

```
Macro-F1 = mean([F1_emotion_i for i in range(16)])
```

#### Hamming Loss

Fraction of incorrect label predictions (both false positives and false negatives).

- **Formula**: `(1 / N×L) × Σᵢ Σⱼ (yᵢⱼ ≠ ŷᵢⱼ)`
  - N = number of samples
  - L = number of labels (16)
- **Interpretation**: Lower is better (0.0 = perfect, 1.0 = all wrong)
- **Example**: If 5% of all label predictions are incorrect, Hamming Loss = 0.05

### Example Evaluation Output

```
Test Set Results:
  Test Loss: 0.2456
  Test Micro-F1: 0.7823
  Test Macro-F1: 0.7234
  Test Hamming Loss: 0.0456

Per-Label F1 Scores:
  joy            : 0.8523
  trust          : 0.7891
  fear           : 0.7234
  surprise       : 0.6892
  sadness        : 0.8012
  disgust        : 0.7456
  anger          : 0.7823
  anticipation   : 0.6734
  love           : 0.8234
  worried        : 0.7123
  disappointed   : 0.7567
  proud          : 0.6891
  embarrassed    : 0.6523
  jealous        : 0.6234
  calm           : 0.7345
  excited        : 0.7891
```

## Project Structure

```
multi-label-emotion-classification/
│
├── data/                           # Data directory
│   ├── .gitkeep                    # Keeps directory in git
│   └── sample_comments.csv         # Generated sample data
│
├── saved_model/                    # Model checkpoints directory
│   ├── .gitkeep                    # Keeps directory in git
│   ├── pytorch_model.bin           # Trained model weights
│   ├── config.json                 # Model configuration
│   ├── tokenizer_config.json       # Tokenizer configuration
│   ├── vocab.txt                   # BERT vocabulary
│   ├── training_config.json        # Training hyperparameters
│   └── training_curves.png         # Loss curves plot
│
├── config.py                       # Configuration parameters
├── dataset.py                      # EmotionDataset class
├── model.py                        # BERTEmotionClassifier model
├── train.py                        # Training script
├── predict.py                      # Prediction script
├── utils.py                        # Utility functions
├── generate_sample_data.py         # Sample data generator
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

### File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration for all hyperparameters and settings |
| `dataset.py` | PyTorch Dataset class for loading and preprocessing data |
| `model.py` | BERT-based multi-label emotion classification model |
| `train.py` | Training pipeline with evaluation and checkpointing |
| `predict.py` | Inference functions and interactive CLI |
| `utils.py` | Helper functions (text cleaning, metrics, visualization, I/O) |
| `generate_sample_data.py` | Generates sample CSV data for demonstration |
| `requirements.txt` | Python package dependencies with versions |

## Troubleshooting

### Common Errors and Solutions

#### 1. FileNotFoundError: Dataset not found

**Error:**
```
FileNotFoundError: Dataset file not found at 'data/sample_comments.csv'
```

**Solution:**
```bash
# Generate sample data
python generate_sample_data.py

# Or place your own CSV file in data/ directory
cp your_dataset.csv data/sample_comments.csv
```

#### 2. Model checkpoint not found during prediction

**Error:**
```
FileNotFoundError: Model checkpoint not found at 'saved_model/'
```

**Solution:**
```bash
# Train the model first
python train.py

# This will create the saved_model/ directory with checkpoints
```

#### 3. CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**Option 1: Reduce batch size** (edit `config.py`):
```python
BATCH_SIZE = 8  # Reduce from 16 to 8
```

**Option 2: Use CPU instead** (edit `config.py`):
```python
DEVICE = "cpu"  # Force CPU usage
```

**Option 3: Reduce sequence length** (edit `config.py`):
```python
MAX_LENGTH = 256  # Reduce from 512 to 256
```

#### 4. Missing columns in CSV

**Error:**
```
ValueError: Missing required columns: {'joy', 'trust', ...}
```

**Solution:**

Ensure your CSV has exactly these columns:
- `text` (comment text)
- 16 emotion columns: `joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`, `love`, `worried`, `disappointed`, `proud`, `embarrassed`, `jealous`, `calm`, `excited`

Example CSV structure:
```csv
text,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,love,worried,disappointed,proud,embarrassed,jealous,calm,excited
"Your comment here",1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1
```

#### 5. Slow training on CPU

**Issue:**
Training takes very long on CPU (10-15 minutes per epoch).

**Solutions:**

**Option 1: Use GPU** (if available):
- Install CUDA toolkit
- Install PyTorch with CUDA support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

**Option 2: Reduce dataset size**:
- Use fewer samples for faster experimentation
- Edit data loading in `train.py` to limit samples

**Option 3: Reduce epochs** (edit `config.py`):
```python
NUM_EPOCHS = 3  # Reduce from 5 to 3
```

#### 6. Import errors

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install transformers>=4.30.0
```

#### 7. Empty predictions (no emotions detected)

**Issue:**
Model returns empty emotion list for all inputs.

**Possible Causes & Solutions:**

**Cause 1: Threshold too high**
```python
# In config.py, lower the threshold
PREDICTION_THRESHOLD = 0.3  # Reduce from 0.5
```

**Cause 2: Model not trained properly**
```bash
# Retrain the model
python train.py
```

**Cause 3: Input text too short or unclear**
```python
# Try more expressive text
"I absolutely love this product! It's amazing and exceeded all my expectations!"
```

#### 8. Inconsistent results across runs

**Issue:**
Different results when running training multiple times.

**Solution:**

Ensure reproducibility settings in `config.py`:
```python
RANDOM_SEED = 42  # Fixed seed for reproducibility
```

Also check that training script sets all seeds:
```python
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
```

#### 9. Vietnamese text not working well

**Issue:**
Poor performance on Vietnamese comments.

**Explanation:**

`bert-base-uncased` is primarily trained on English text. For better Vietnamese support:

**Option 1: Use multilingual BERT**

Edit `config.py`:
```python
MODEL_NAME = "bert-base-multilingual-cased"
```

**Option 2: Use XLM-RoBERTa**

Edit `config.py`:
```python
MODEL_NAME = "xlm-roberta-base"
```

Then update imports in `model.py`:
```python
from transformers import XLMRobertaModel, XLMRobertaTokenizer
```

#### 10. Permission denied when saving model

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'saved_model/'
```

**Solution:**
```bash
# Create directory with proper permissions
mkdir -p saved_model
chmod 755 saved_model

# Or run with appropriate permissions
sudo python train.py  # Not recommended
```

### Getting Help

If you encounter issues not covered here:

1. **Check the error message carefully** - it often contains the solution
2. **Verify your Python version** - requires Python 3.8+
3. **Check installed package versions** - run `pip list`
4. **Review configuration** - ensure `config.py` has valid values
5. **Check data format** - ensure CSV matches expected structure
6. **Try with sample data** - run `generate_sample_data.py` first

## Advanced Configuration

### Adjusting the Prediction Threshold

The threshold determines which emotions are included in predictions:

```python
# In config.py
PREDICTION_THRESHOLD = 0.5  # Default

# Lower threshold = more emotions predicted (higher recall, lower precision)
PREDICTION_THRESHOLD = 0.3

# Higher threshold = fewer emotions predicted (lower recall, higher precision)
PREDICTION_THRESHOLD = 0.7
```

### Customizing Training

```python
# In config.py

# Faster training (less accurate)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
BATCH_SIZE = 32

# More accurate training (slower)
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
BATCH_SIZE = 8
```

### Using Different BERT Models

```python
# In config.py

# Larger model (better accuracy, more memory)
MODEL_NAME = "bert-large-uncased"
HIDDEN_SIZE = 1024

# Multilingual support
MODEL_NAME = "bert-base-multilingual-cased"

# Smaller model (faster, less accurate)
MODEL_NAME = "distilbert-base-uncased"
HIDDEN_SIZE = 768
```

## Multi-Person Training

This project supports collaborative training where multiple people can work on experiments simultaneously.

### Quick Start for Teams

See [QUICK_START_MULTI_PERSON.md](QUICK_START_MULTI_PERSON.md) for a 5-minute setup guide.

### Training with Custom Configuration

Use `train_with_args.py` to train with different configurations without modifying `config.py`:

```bash
# Person A - Baseline
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001_baseline/ \
    --experiment-name "Person A - Baseline" \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5

# Person B - Higher Learning Rate
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001_higher_lr/ \
    --experiment-name "Person B - Higher LR" \
    --epochs 5 \
    --lr 5e-5

# Person C - Larger Batch Size
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_c/exp001_larger_batch/ \
    --experiment-name "Person C - Larger Batch" \
    --epochs 5 \
    --batch-size 32
```

### Comparing Experiments

Compare results from multiple experiments:

```bash
python compare_experiments.py \
    experiments/person_a/exp001_baseline/ \
    experiments/person_b/exp001_higher_lr/ \
    experiments/person_c/exp001_larger_batch/
```

This generates:
- Console output with comparison tables
- `comparison.csv` - Results in CSV format
- `comparison.md` - Markdown report with recommendations

### Experiment Tracking

All experiments should be documented in `experiments/experiment_log.md`:

```markdown
## Exp001 - Baseline (Person A)
- **Date:** 2026-04-20
- **Status:** ✅ Completed
- **Results:** Macro F1 = 0.756
- **Path:** experiments/person_a/exp001_baseline/
```

### Detailed Guides

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete guide for multi-person training
  - Git workflow
  - Distributed training
  - Cloud services (Colab, Kaggle, AWS)
  - Best practices

- **[QUICK_START_MULTI_PERSON.md](QUICK_START_MULTI_PERSON.md)** - Quick setup guide
  - 5-minute setup
  - Common scenarios
  - Example commands

- **[experiments/README.md](experiments/README.md)** - Experiment tracking guide
  - Naming conventions
  - Documentation templates
  - Best practices

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Hugging Face Transformers**: For providing easy-to-use BERT implementations
- **PyTorch**: For the deep learning framework

---

**Questions or Issues?** Please refer to the Troubleshooting section above or review the code documentation in each module.
