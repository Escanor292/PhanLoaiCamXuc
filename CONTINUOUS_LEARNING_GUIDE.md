# Continuous Learning System - Model Tự Động Nâng Cấp

## 🎯 Mục Tiêu

Tạo hệ thống để model **tự động cải thiện theo thời gian thực** khi nhiều người training:

```
Người A training → Model A (F1: 0.75)
                                        ↘
Người B training → Model B (F1: 0.78)  → Auto-select Best → Deploy Model B
                                        ↗
Người C training → Model C (F1: 0.72)
```

## 📊 So Sánh Các Phương Pháp

| Phương Pháp | Tự Động? | Độ Phức Tạp | Phù Hợp Cho |
|-------------|----------|-------------|-------------|
| **Manual Selection** | ❌ Không | Thấp | Team nhỏ, research |
| **Continuous Training** | ✅ Có | Trung bình | Production, team lớn |
| **Federated Learning** | ✅ Có | Cao | Privacy-sensitive, distributed |
| **Online Learning** | ✅ Có | Cao | Real-time updates |
| **Model Ensemble** | ⚠️ Bán tự động | Trung bình | High accuracy needs |

---

## 1️⃣ Manual Selection (Hiện Tại)

### Cách Hoạt Động

```
1. Người A training → Save model A
2. Người B training → Save model B
3. Người C training → Save model C
4. Compare results manually
5. Select best model
6. Deploy manually
```

### Ưu Điểm
- ✅ Đơn giản, dễ hiểu
- ✅ Kiểm soát hoàn toàn
- ✅ Không cần infrastructure phức tạp

### Nhược Điểm
- ❌ Không tự động
- ❌ Chậm (phải đợi compare)
- ❌ Dễ miss best model

### Khi Nào Dùng
- Team nhỏ (2-5 người)
- Research projects
- Không cần real-time updates

---

## 2️⃣ Continuous Training Pipeline (Khuyến Nghị)

### Kiến Trúc

```
┌─────────────┐
│ Person A    │──┐
│ Training    │  │
└─────────────┘  │
                 │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
┌─────────────┐  │    │              │    │              │    │              │
│ Person B    │──┼───→│ Model        │───→│ Auto         │───→│ Production   │
│ Training    │  │    │ Registry     │    │ Evaluation   │    │ Deployment   │
└─────────────┘  │    │              │    │              │    │              │
                 │    └──────────────┘    └──────────────┘    └──────────────┘
┌─────────────┐  │           ↓                    ↓                   ↓
│ Person C    │──┘      Save Models        Compare Metrics      Deploy Best
│ Training    │
└─────────────┘
```

### Implementation

✅ **ĐÃ TÍCH HỢP SẴN!** Bạn chỉ cần sử dụng flag `--register-model` khi training.

#### Cách Sử Dụng

**Person A training:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001/ \
    --experiment-name "Person A - Baseline" \
    --register-model
```

**Person B training:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001/ \
    --experiment-name "Person B - Higher LR" \
    --lr 5e-5 \
    --register-model
```

**Auto-select best model:**
```bash
# Xem tất cả models
python model_registry.py list

# Deploy best model
python model_registry.py deploy --model-id <best_model_id>
```

**Enable auto-deployment:**
```bash
# Set environment variable
export AUTO_DEPLOY=true

# Bây giờ mỗi khi có model tốt hơn, tự động deploy
python train_with_args.py --register-model ...
```

#### Chi Tiết Implementation

Hệ thống đã được tích hợp sẵn với 2 components chính:

**1. Model Registry (`model_registry.py`):**
- Tracks all trained models with metrics
- Automatically identifies best model by macro F1
- Supports auto-deployment when `AUTO_DEPLOY=true`
- Backs up previous production models
- CLI interface for management

**2. Training Script Integration (`train_with_args.py`):**
- Added `--register-model` flag
- Automatically registers model after training completes
- Captures all metrics (macro F1, micro F1, test loss, hamming loss)
- Captures metadata (person, experiment name, hyperparameters)
- Triggers auto-evaluation and deployment if enabled

#### Code Reference

Xem implementation trong:
- `model_registry.py` - Complete registry system (lines 1-400+)
- `train_with_args.py` - Integration code (import + registration section at end of main())

Không cần tạo thêm code - chỉ cần sử dụng!

#### Code Reference

Xem implementation trong:
- `model_registry.py` - Complete registry system (lines 1-400+)
- `train_with_args.py` - Integration code (import + registration section at end of main())

Không cần tạo thêm code - chỉ cần sử dụng!

### Ưu Điểm
- ✅ Tự động select best model
- ✅ Track tất cả models
- ✅ Có thể auto-deploy
- ✅ Backup models cũ

### Nhược Điểm
- ⚠️ Cần shared storage
- ⚠️ Cần coordination

---

## 3️⃣ Federated Learning (Advanced)

### Cách Hoạt Động

```
┌─────────────┐
│ Person A    │ ──┐
│ Local Data  │   │
└─────────────┘   │
                  │    ┌──────────────┐
┌─────────────┐   │    │              │
│ Person B    │ ──┼───→│ Central      │
│ Local Data  │   │    │ Server       │
└─────────────┘   │    │              │
                  │    └──────────────┘
┌─────────────┐   │           │
│ Person C    │ ──┘           │
│ Local Data  │               ↓
└─────────────┘        Aggregate Models
                       → Global Model
```

### Implementation với Flower

```bash
pip install flwr
```

**Server (`fl_server.py`):**

```python
import flwr as fl

def weighted_average(metrics):
    """Aggregate metrics from clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
    ),
)
```

**Client (`fl_client.py`):**

```python
import flwr as fl
from model import BERTEmotionClassifier
# ... imports ...

class EmotionClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train model
        train_loss = train_epoch(self.model, self.train_loader, ...)
        return self.get_parameters(config={}), len(self.train_loader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluate model
        loss, accuracy = evaluate(self.model, self.val_loader, ...)
        return loss, len(self.val_loader), {"accuracy": accuracy}

# Start client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=EmotionClient(model, train_loader, val_loader),
)
```

**Chạy:**

```bash
# Terminal 1: Start server
python fl_server.py

# Terminal 2: Person A
python fl_client.py --data data/person_a.csv

# Terminal 3: Person B
python fl_client.py --data data/person_b.csv

# Terminal 4: Person C
python fl_client.py --data data/person_c.csv
```

### Ưu Điểm
- ✅ Privacy-preserving (data không rời máy)
- ✅ Tự động aggregate models
- ✅ Scalable

### Nhược Điểm
- ❌ Phức tạp
- ❌ Cần server infrastructure
- ❌ Communication overhead

---

## 4️⃣ Online Learning (Real-time)

### Cách Hoạt Động

```
New Data → Update Model → Deploy → Repeat
```

### Implementation

```python
"""
Online Learning - Continuous model updates
"""

class OnlineLearner:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
    
    def update(self, new_data_batch):
        """
        Update model with new data batch.
        
        Args:
            new_data_batch: Dict with 'input_ids', 'attention_mask', 'labels'
        """
        self.model.train()
        
        input_ids = new_data_batch['input_ids']
        attention_mask = new_data_batch['attention_mask']
        labels = new_data_batch['labels']
        
        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def should_deploy(self, val_metrics, threshold=0.01):
        """
        Decide if model should be deployed based on improvement.
        
        Args:
            val_metrics: Current validation metrics
            threshold: Minimum improvement required
            
        Returns:
            bool: Whether to deploy
        """
        if not hasattr(self, 'best_f1'):
            self.best_f1 = 0
        
        current_f1 = val_metrics['macro_f1']
        improvement = current_f1 - self.best_f1
        
        if improvement > threshold:
            self.best_f1 = current_f1
            return True
        
        return False

# Usage
learner = OnlineLearner(model, optimizer)

# Continuous learning loop
while True:
    # Get new data (from queue, database, etc.)
    new_batch = get_new_data_batch()
    
    # Update model
    loss = learner.update(new_batch)
    
    # Evaluate periodically
    if should_evaluate():
        val_metrics = evaluate(model, val_loader)
        
        # Deploy if improved
        if learner.should_deploy(val_metrics):
            save_model(model, 'saved_model/')
            print("✓ Model updated and deployed!")
```

### Ưu Điểm
- ✅ Real-time updates
- ✅ Adapts to new data quickly

### Nhược Điểm
- ❌ Risk of catastrophic forgetting
- ❌ Needs careful monitoring
- ❌ Complex infrastructure

---

## 5️⃣ Model Ensemble (Hybrid)

### Cách Hoạt Động

```
Model A (0.75) ──┐
                 │
Model B (0.78) ──┼──→ Ensemble → Final Prediction (0.80)
                 │
Model C (0.72) ──┘
```

### Implementation

```python
"""
Model Ensemble - Combine multiple models
"""

class ModelEnsemble:
    def __init__(self, model_paths, weights=None):
        """
        Initialize ensemble.
        
        Args:
            model_paths: List of paths to model checkpoints
            weights: Optional weights for each model
        """
        self.models = []
        for path in model_paths:
            model, tokenizer = load_model(path)
            self.models.append(model)
        
        self.tokenizer = tokenizer
        self.weights = weights or [1.0 / len(self.models)] * len(self.models)
    
    def predict(self, text, threshold=0.5):
        """
        Predict using ensemble.
        
        Args:
            text: Input text
            threshold: Prediction threshold
            
        Returns:
            dict: Predictions with confidence scores
        """
        all_probs = []
        
        # Get predictions from each model
        for model in self.models:
            result = predict_emotions(text, model, self.tokenizer, 'cpu')
            probs = [result['scores'][label] for label in Config.EMOTION_LABELS]
            all_probs.append(probs)
        
        # Weighted average
        ensemble_probs = np.average(all_probs, axis=0, weights=self.weights)
        
        # Apply threshold
        predicted_emotions = [
            Config.EMOTION_LABELS[i]
            for i, prob in enumerate(ensemble_probs)
            if prob >= threshold
        ]
        
        scores = {
            label: float(prob)
            for label, prob in zip(Config.EMOTION_LABELS, ensemble_probs)
        }
        
        return {
            'emotions': predicted_emotions,
            'scores': scores
        }

# Usage
ensemble = ModelEnsemble([
    'experiments/person_a/exp001/',
    'experiments/person_b/exp001/',
    'experiments/person_c/exp001/'
])

result = ensemble.predict("I love this product!")
print(result['emotions'])
```

### Ưu Điểm
- ✅ Often better than single model
- ✅ Robust to individual model failures
- ✅ Can weight models by performance

### Nhược Điểm
- ❌ Slower inference (multiple models)
- ❌ More memory usage
- ❌ More complex deployment

---

## 📋 So Sánh Tổng Hợp

| Feature | Manual | Continuous | Federated | Online | Ensemble |
|---------|--------|------------|-----------|--------|----------|
| **Tự động** | ❌ | ✅ | ✅ | ✅ | ⚠️ |
| **Real-time** | ❌ | ⚠️ | ⚠️ | ✅ | ❌ |
| **Privacy** | ✅ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Độ phức tạp** | Thấp | Trung bình | Cao | Cao | Trung bình |
| **Infrastructure** | Minimal | Moderate | High | High | Moderate |
| **Accuracy** | Good | Good | Good | Variable | Best |
| **Setup time** | 1 giờ | 1 ngày | 1 tuần | 1 tuần | 1 ngày |

---

## 🎯 Khuyến Nghị

### Cho Team Nhỏ (2-5 người)
→ **Continuous Training Pipeline** (Option 2)
- Đủ tự động
- Không quá phức tạp
- Dễ maintain

### Cho Team Lớn (5+ người)
→ **Federated Learning** (Option 3)
- Scalable
- Privacy-preserving
- Professional

### Cho Production với High Traffic
→ **Online Learning** (Option 4)
- Real-time adaptation
- Always up-to-date

### Cho Maximum Accuracy
→ **Model Ensemble** (Option 5)
- Best performance
- Robust

---

## 🚀 Quick Start: Continuous Training

```bash
# 1. Tạo model registry
mkdir model_registry

# 2. Person A training
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001/ \
    --register-model

# 3. Person B training
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001/ \
    --lr 5e-5 \
    --register-model

# 4. Xem models
python model_registry.py list

# 5. Deploy best model
python model_registry.py deploy --model-id <best_model_id>

# 6. Enable auto-deploy
export AUTO_DEPLOY=true
```

---

## 📚 Tài Liệu Tham Khảo

- **Federated Learning**: https://flower.dev/
- **MLflow**: https://mlflow.org/
- **Weights & Biases**: https://wandb.ai/
- **Online Learning**: https://scikit-learn.org/stable/modules/computing.html#incremental-learning

---

**Created:** 2026-04-20  
**Version:** 1.0
