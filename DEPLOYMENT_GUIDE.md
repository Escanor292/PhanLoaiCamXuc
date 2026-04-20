# Hướng Dẫn Triển Khai Cho Nhiều Người Training

## Mục Lục
1. [Chia Sẻ Project Qua Git](#1-chia-sẻ-project-qua-git)
2. [Training Trên Nhiều Máy Khác Nhau](#2-training-trên-nhiều-máy-khác-nhau)
3. [Training Phân Tán (Distributed Training)](#3-training-phân-tán-distributed-training)
4. [Sử Dụng Cloud Services](#4-sử-dụng-cloud-services)
5. [Best Practices](#5-best-practices)

---

## 1. Chia Sẻ Project Qua Git

### Bước 1: Khởi tạo Git Repository

```bash
# Khởi tạo git (nếu chưa có)
git init

# Add tất cả files (đã có .gitignore để loại trừ model checkpoints)
git add .

# Commit
git commit -m "Initial commit: Multi-label Emotion Classification"

# Tạo repository trên GitHub/GitLab
# Sau đó push lên remote
git remote add origin https://github.com/your-username/emotion-classification.git
git push -u origin main
```

### Bước 2: Người Khác Clone Project

```bash
# Clone repository
git clone https://github.com/your-username/emotion-classification.git
cd emotion-classification

# Cài đặt dependencies
pip install -r requirements.txt

# Chuẩn bị dữ liệu (nếu có dữ liệu riêng)
# Hoặc sử dụng sample data có sẵn
```

### Bước 3: Mỗi Người Training Độc Lập

Mỗi người có thể:
- Training trên máy riêng của họ
- Sử dụng dữ liệu riêng hoặc dữ liệu chung
- Lưu model checkpoints vào thư mục riêng

```bash
# Người A training
python train.py

# Model sẽ được lưu vào saved_model/
# Mỗi người có thể đổi tên model checkpoint của mình
```

---

## 2. Training Trên Nhiều Máy Khác Nhau

### Phương Án A: Chia Dữ Liệu

Mỗi người training trên một phần dữ liệu khác nhau:

```python
# Tạo script split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load toàn bộ dữ liệu
df = pd.read_csv('data/all_comments.csv')

# Chia thành N phần (ví dụ: 3 người)
n_splits = 3
split_size = len(df) // n_splits

for i in range(n_splits):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < n_splits - 1 else len(df)
    
    df_split = df.iloc[start_idx:end_idx]
    df_split.to_csv(f'data/comments_part_{i+1}.csv', index=False)
    print(f"Part {i+1}: {len(df_split)} samples")
```

**Người 1:**
```bash
# Sửa config.py hoặc truyền argument
python train.py --data data/comments_part_1.csv --output saved_model/model_part1/
```

**Người 2:**
```bash
python train.py --data data/comments_part_2.csv --output saved_model/model_part2/
```

**Người 3:**
```bash
python train.py --data data/comments_part_3.csv --output saved_model/model_part3/
```

### Phương Án B: Thử Nghiệm Hyperparameters Khác Nhau

Mỗi người thử các hyperparameters khác nhau:

**Người 1 - Learning Rate Cao:**
```python
# Sửa config.py
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
DROPOUT_RATE = 0.3
```

**Người 2 - Learning Rate Thấp:**
```python
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
```

**Người 3 - Batch Size Lớn:**
```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 64
DROPOUT_RATE = 0.4
```

### Phương Án C: Chia Theo Ngôn Ngữ

**Người 1 - Training cho tiếng Anh:**
```python
# Filter English comments
df_en = df[df['language'] == 'en']
```

**Người 2 - Training cho tiếng Việt:**
```python
# Filter Vietnamese comments
df_vi = df[df['language'] == 'vi']
```

---

## 3. Training Phân Tán (Distributed Training)

### Sử Dụng PyTorch Distributed

Tạo file `train_distributed.py`:

```python
"""
Distributed Training Script for Multi-GPU or Multi-Machine Setup
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from config import Config
from model import BERTEmotionClassifier
from dataset import EmotionDataset
from utils import load_data
from transformers import BertTokenizer


def setup(rank, world_size):
    """
    Initialize distributed training.
    
    Args:
        rank: Unique identifier for each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_distributed(rank, world_size):
    """
    Training function for each process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    print(f"Running distributed training on rank {rank}")
    setup(rank, world_size)
    
    # Load data
    texts, labels = load_data(Config.DATA_FILE)
    
    # Create dataset
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = EmotionDataset(texts, labels, tokenizer, Config.MAX_LENGTH)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model and move to GPU
    model = BERTEmotionClassifier(
        num_labels=len(Config.EMOTION_LABELS),
        dropout_rate=Config.DROPOUT_RATE
    )
    model = model.to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if rank == 0:  # Only print from main process
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save model (only from rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), 'saved_model/distributed_model.pt')
        print("Model saved!")
    
    cleanup()


def main():
    """Main function to launch distributed training."""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Warning: Distributed training requires at least 2 GPUs")
        print(f"Found {world_size} GPU(s)")
        return
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
```

### Chạy Distributed Training

**Trên 1 máy với nhiều GPU:**
```bash
python train_distributed.py
```

**Trên nhiều máy (Multi-Node):**

**Máy 1 (Master Node):**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    train_distributed.py
```

**Máy 2 (Worker Node):**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=12355 \
    train_distributed.py
```

---

## 4. Sử Dụng Cloud Services

### A. Google Colab (Miễn Phí)

**Bước 1:** Upload project lên Google Drive

**Bước 2:** Tạo notebook `train_colab.ipynb`:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/emotion-classification

# Install dependencies
!pip install -r requirements.txt

# Train model
!python train.py

# Download trained model
from google.colab import files
files.download('saved_model/pytorch_model.bin')
```

**Ưu điểm:**
- Miễn phí GPU (Tesla T4)
- Dễ chia sẻ notebook
- Mỗi người có thể chạy riêng

**Nhược điểm:**
- Giới hạn thời gian (12 giờ)
- Cần reconnect thường xuyên

### B. Kaggle Notebooks

Tương tự Google Colab nhưng:
- GPU: Tesla P100 (mạnh hơn Colab)
- Thời gian: 30 giờ/tuần
- Có thể tạo private notebooks

### C. AWS SageMaker / Azure ML / Google Cloud AI

**Ví dụ với AWS SageMaker:**

```python
# train_sagemaker.py
import sagemaker
from sagemaker.pytorch import PyTorch

# Define training job
estimator = PyTorch(
    entry_point='train.py',
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='1.12',
    py_version='py38',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 2e-5
    }
)

# Start training
estimator.fit({'training': 's3://my-bucket/data/'})
```

---

## 5. Best Practices

### A. Quản Lý Experiments

Sử dụng **MLflow** hoặc **Weights & Biases** để track experiments:

```python
# Thêm vào train.py
import mlflow

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", Config.LEARNING_RATE)
mlflow.log_param("batch_size", Config.BATCH_SIZE)
mlflow.log_param("num_epochs", Config.NUM_EPOCHS)

# Log metrics
for epoch in range(Config.NUM_EPOCHS):
    # ... training code ...
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)

# Log model
mlflow.pytorch.log_model(model, "model")

mlflow.end_run()
```

### B. Chia Sẻ Model Checkpoints

**Option 1: Google Drive**
```bash
# Upload model
rclone copy saved_model/ gdrive:emotion-models/

# Download model
rclone copy gdrive:emotion-models/ saved_model/
```

**Option 2: Hugging Face Hub**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="saved_model",
    repo_id="your-username/emotion-classifier",
    repo_type="model"
)
```

### C. Versioning Models

```bash
# Đặt tên model theo version
saved_model/
├── v1.0_baseline/
├── v1.1_improved_lr/
├── v2.0_multilingual/
└── v2.1_final/
```

### D. Documentation

Mỗi người nên document experiments của mình:

```markdown
# experiments/experiment_log.md

## Experiment 1 - Baseline
- **Date:** 2026-04-20
- **Person:** Người A
- **Data:** 10,000 samples
- **Hyperparameters:**
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Epochs: 5
- **Results:**
  - Train Loss: 0.234
  - Val Loss: 0.289
  - Macro F1: 0.756
- **Notes:** Baseline model, good starting point

## Experiment 2 - Higher Learning Rate
- **Date:** 2026-04-21
- **Person:** Người B
- **Changes:** Increased LR to 5e-5
- **Results:**
  - Train Loss: 0.198
  - Val Loss: 0.312
  - Macro F1: 0.742
- **Notes:** Overfitting, need regularization
```

---

## 6. Workflow Đề Xuất

### Workflow Cho Team Nhỏ (2-5 người)

```
1. Setup Repository
   ├── Người A: Setup Git repo
   └── Tất cả: Clone và setup environment

2. Phân Công Tasks
   ├── Người A: Baseline model
   ├── Người B: Hyperparameter tuning
   ├── Người C: Data augmentation
   └── Người D: Model evaluation

3. Training Parallel
   ├── Mỗi người train trên máy riêng
   └── Share results qua Git/Drive

4. Model Selection
   ├── Compare results
   ├── Select best model
   └── Deploy

5. Documentation
   └── Update README với best practices
```

### Workflow Cho Team Lớn (5+ người)

```
1. Infrastructure Setup
   ├── Setup cloud training (AWS/GCP)
   ├── Setup experiment tracking (MLflow)
   └── Setup model registry

2. Data Pipeline
   ├── Team A: Data collection
   ├── Team B: Data cleaning
   └── Team C: Data validation

3. Model Development
   ├── Team D: Model architecture
   ├── Team E: Training pipeline
   └── Team F: Evaluation metrics

4. Distributed Training
   ├── Use multi-GPU/multi-node
   └── Parallel hyperparameter search

5. Model Deployment
   ├── Team G: Model serving
   └── Team H: Monitoring
```

---

## 7. Troubleshooting

### Vấn Đề: Xung Đột Khi Merge Code

**Giải pháp:**
```bash
# Tạo branch riêng cho mỗi experiment
git checkout -b experiment/person-a-baseline
git checkout -b experiment/person-b-tuning

# Merge sau khi review
git checkout main
git merge experiment/person-a-baseline
```

### Vấn Đề: Model Checkpoints Quá Lớn

**Giải pháp:**
- Sử dụng Git LFS cho large files
- Hoặc lưu trên cloud storage
- Chỉ commit config và code, không commit models

```bash
# Setup Git LFS
git lfs install
git lfs track "*.bin"
git lfs track "*.pt"
```

### Vấn Đề: Khác Biệt Về Environment

**Giải pháp:**
```bash
# Sử dụng Docker
docker build -t emotion-classifier .
docker run -v $(pwd):/app emotion-classifier python train.py
```

---

## 8. Checklist Trước Khi Bắt Đầu

- [ ] Repository đã được setup và shared
- [ ] Tất cả mọi người có access vào data
- [ ] Environment đã được test (requirements.txt)
- [ ] Phân công tasks rõ ràng
- [ ] Setup experiment tracking
- [ ] Định nghĩa success metrics
- [ ] Setup communication channel (Slack/Discord)
- [ ] Document workflow và best practices

---

## Liên Hệ & Support

Nếu có vấn đề khi setup multi-person training, tham khảo:
- PyTorch Distributed: https://pytorch.org/tutorials/beginner/dist_overview.html
- MLflow: https://mlflow.org/docs/latest/index.html
- Hugging Face Hub: https://huggingface.co/docs/hub/index

---

**Cập nhật:** 2026-04-20  
**Version:** 1.0
