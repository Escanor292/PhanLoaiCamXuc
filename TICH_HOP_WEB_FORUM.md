# Tích Hợp Emotion Classification vào Web Forum

## 🎯 Mục tiêu

Hướng dẫn tích hợp AI phân tích cảm xúc vào web forum với:
- ✅ **Biểu đồ tròn sentiment** (Tích cực/Tiêu cực/Trung tính)
- ✅ **Biểu đồ cột emotions** (16 cảm xúc chi tiết)
- ✅ **Real-time analysis** khi user post comment
- ✅ **REST API** dễ tích hợp

---

## 🏗️ Kiến trúc hệ thống

```
Web Forum (Frontend)
├── User posts comment
├── JavaScript calls API
├── Display charts với Chart.js
└── Show results real-time

↕️ HTTP API

Emotion API Server (Backend)
├── Flask/FastAPI server
├── Load trained model
├── Process text input
└── Return JSON results
```

---

## 🚀 Bước 1: Setup API Server

### **1.1. Cài đặt dependencies**

```bash
# Cài thêm packages cho API
pip install -r requirements_api.txt
```

### **1.2. Chạy API server**

```bash
# Start API server
python api_server.py
```

**Kết quả:**
```
🚀 Starting Emotion Classification API Server...
✅ Model loaded successfully!
🌐 API Server starting on http://localhost:5000

Available endpoints:
  POST /predict - Analyze emotions
  GET /health - Health check
  GET /emotions/list - List supported emotions
```

### **1.3. Test API**

```bash
# Test health check
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sản phẩm này tuyệt vời!"}'
```

---

## 🎨 Bước 2: Demo Web Interface

### **2.1. Mở web demo**

```bash
# Mở file trong browser
open web_demo.html
# Hoặc
start web_demo.html
```

### **2.2. Test với sample comments**

1. Nhập comment: "Sản phẩm này tuyệt vời!"
2. Click "Phân tích cảm xúc"
3. Xem kết quả:
   - **Sentiment chart**: Tích cực 85%, Tiêu cực 10%, Trung tính 5%
   - **Emotion chart**: vui vẻ, tin tưởng, phấn khích
   - **Detected emotions**: Badges với confidence scores

---

## 🔌 Bước 3: Tích hợp vào Forum thực tế

### **3.1. PHP Integration (Ví dụ)**

```php
<?php
// forum_comment.php

function analyzeComment($text) {
    $api_url = 'http://localhost:5000/predict';
    
    $data = json_encode([
        'text' => $text,
        'threshold' => 0.4
    ]);
    
    $options = [
        'http' => [
            'header' => "Content-Type: application/json\r\n",
            'method' => 'POST',
            'content' => $data
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($api_url, false, $context);
    
    return json_decode($result, true);
}

// Khi user post comment
if ($_POST['comment']) {
    $comment_text = $_POST['comment'];
    
    // Lưu comment vào database
    $comment_id = saveComment($comment_text);
    
    // Phân tích cảm xúc
    $emotion_result = analyzeComment($comment_text);
    
    // Lưu kết quả phân tích
    saveEmotionAnalysis($comment_id, $emotion_result);
    
    // Hiển thị kết quả
    echo json_encode([
        'comment_id' => $comment_id,
        'emotions' => $emotion_result
    ]);
}
?>
```

### **3.2. JavaScript Integration**

```javascript
// forum.js - Tích hợp vào forum hiện có

class EmotionAnalyzer {
    constructor(apiUrl = 'http://localhost:5000') {
        this.apiUrl = apiUrl;
    }
    
    async analyzeComment(text, threshold = 0.4) {
        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text, threshold })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Emotion analysis failed:', error);
            return null;
        }
    }
    
    renderEmotionCharts(containerId, emotionData) {
        const container = document.getElementById(containerId);
        
        // Tạo sentiment chart
        this.createSentimentChart(container, emotionData.sentiment);
        
        // Tạo emotion badges
        this.createEmotionBadges(container, emotionData.detected_emotions);
    }
    
    createSentimentChart(container, sentiment) {
        const canvas = document.createElement('canvas');
        canvas.width = 200;
        canvas.height = 200;
        container.appendChild(canvas);
        
        new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['Tích cực', 'Tiêu cực', 'Trung tính'],
                datasets: [{
                    data: [
                        sentiment.positive * 100,
                        sentiment.negative * 100,
                        sentiment.neutral * 100
                    ],
                    backgroundColor: ['#28a745', '#dc3545', '#6c757d']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
    
    createEmotionBadges(container, emotions) {
        const badgeContainer = document.createElement('div');
        badgeContainer.className = 'emotion-badges mt-2';
        
        emotions.forEach(emotion => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-primary me-1';
            badge.textContent = `${emotion.vietnamese} (${(emotion.score * 100).toFixed(0)}%)`;
            badgeContainer.appendChild(badge);
        });
        
        container.appendChild(badgeContainer);
    }
}

// Sử dụng trong forum
const emotionAnalyzer = new EmotionAnalyzer();

// Khi user post comment
document.getElementById('postComment').addEventListener('click', async function() {
    const commentText = document.getElementById('commentInput').value;
    
    // Post comment như bình thường
    const commentId = await postComment(commentText);
    
    // Phân tích cảm xúc
    const emotionResult = await emotionAnalyzer.analyzeComment(commentText);
    
    if (emotionResult && emotionResult.success) {
        // Hiển thị charts trong comment
        const chartContainer = document.getElementById(`emotion-${commentId}`);
        emotionAnalyzer.renderEmotionCharts(chartContainer, emotionResult);
    }
});
```

### **3.3. React Integration**

```jsx
// EmotionAnalysis.jsx

import React, { useState, useEffect } from 'react';
import { Doughnut, Bar } from 'react-chartjs-2';

const EmotionAnalysis = ({ commentText, threshold = 0.4 }) => {
    const [emotionData, setEmotionData] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        if (commentText) {
            analyzeEmotion();
        }
    }, [commentText]);
    
    const analyzeEmotion = async () => {
        setLoading(true);
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: commentText, threshold })
            });
            
            const data = await response.json();
            if (data.success) {
                setEmotionData(data);
            }
        } catch (error) {
            console.error('Emotion analysis failed:', error);
        } finally {
            setLoading(false);
        }
    };
    
    if (loading) return <div>Đang phân tích cảm xúc...</div>;
    if (!emotionData) return null;
    
    const sentimentChartData = {
        labels: ['Tích cực', 'Tiêu cực', 'Trung tính'],
        datasets: [{
            data: [
                emotionData.sentiment.positive * 100,
                emotionData.sentiment.negative * 100,
                emotionData.sentiment.neutral * 100
            ],
            backgroundColor: ['#28a745', '#dc3545', '#6c757d']
        }]
    };
    
    return (
        <div className="emotion-analysis">
            <div className="row">
                <div className="col-md-6">
                    <h6>Sentiment Analysis</h6>
                    <Doughnut data={sentimentChartData} />
                </div>
                <div className="col-md-6">
                    <h6>Detected Emotions</h6>
                    <div className="emotion-badges">
                        {emotionData.detected_emotions.map((emotion, index) => (
                            <span key={index} className="badge bg-primary me-1">
                                {emotion.vietnamese} ({(emotion.score * 100).toFixed(0)}%)
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EmotionAnalysis;
```

---

## 📊 API Reference

### **POST /predict**

**Request:**
```json
{
    "text": "Sản phẩm này tuyệt vời!",
    "threshold": 0.4
}
```

**Response:**
```json
{
    "success": true,
    "text": "Sản phẩm này tuyệt vời!",
    "emotions": {
        "joy": {
            "score": 0.85,
            "vietnamese": "vui vẻ",
            "detected": true
        },
        "trust": {
            "score": 0.72,
            "vietnamese": "tin tưởng", 
            "detected": true
        }
        // ... 14 emotions khác
    },
    "detected_emotions": [
        {
            "emotion": "joy",
            "vietnamese": "vui vẻ",
            "score": 0.85
        }
    ],
    "sentiment": {
        "positive": 0.78,
        "negative": 0.12,
        "neutral": 0.10
    },
    "overall_sentiment": "positive",
    "threshold": 0.4
}
```

### **GET /health**

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
}
```

### **GET /emotions/list**

**Response:**
```json
{
    "emotions": [
        {
            "english": "joy",
            "vietnamese": "vui vẻ"
        }
        // ... 15 emotions khác
    ],
    "total": 16
}
```

---

## 🎨 UI Components

### **1. Sentiment Pie Chart**
```html
<div class="sentiment-chart">
    <canvas id="sentimentChart"></canvas>
    <div class="sentiment-summary">
        <span class="sentiment-positive">😊 Tích cực: 78%</span>
    </div>
</div>
```

### **2. Emotion Bar Chart**
```html
<div class="emotion-chart">
    <canvas id="emotionChart"></canvas>
</div>
```

### **3. Emotion Badges**
```html
<div class="emotion-badges">
    <span class="badge emotion-high">vui vẻ (85%)</span>
    <span class="badge emotion-medium">tin tưởng (72%)</span>
    <span class="badge emotion-low">phấn khích (45%)</span>
</div>
```

### **4. CSS Styles**
```css
.emotion-badge {
    display: inline-block;
    padding: 5px 10px;
    margin: 2px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
}

.emotion-high { background: #28a745; color: white; }
.emotion-medium { background: #ffc107; color: black; }
.emotion-low { background: #6c757d; color: white; }

.sentiment-positive { color: #28a745; }
.sentiment-negative { color: #dc3545; }
.sentiment-neutral { color: #6c757d; }
```

---

## 🚀 Production Deployment

### **1. Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements_api.txt ./
RUN pip install -r requirements.txt -r requirements_api.txt

# Copy model and code
COPY saved_model/ ./saved_model/
COPY *.py ./

# Expose port
EXPOSE 5000

# Run server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_server:app"]
```

```bash
# Build và run
docker build -t emotion-api .
docker run -p 5000:5000 emotion-api
```

### **2. Production Server**

```bash
# Sử dụng Gunicorn cho production
pip install gunicorn

# Run với multiple workers
gunicorn --bind 0.0.0.0:5000 --workers 4 api_server:app
```

### **3. Nginx Reverse Proxy**

```nginx
# /etc/nginx/sites-available/emotion-api
server {
    listen 80;
    server_name your-domain.com;
    
    location /api/emotion/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📈 Performance Optimization

### **1. Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_emotions_cached(text, threshold):
    # Cache predictions for identical texts
    return predict_emotions(text, threshold)
```

### **2. Batch Processing**
```python
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    texts = request.json['texts']
    results = []
    
    for text in texts:
        result = predict_emotions(text)
        results.append(result)
    
    return jsonify({'results': results})
```

### **3. Async Processing**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.route('/predict/async', methods=['POST'])
async def predict_async():
    text = request.json['text']
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, predict_emotions, text
    )
    
    return jsonify(result)
```

---

## 🔧 Troubleshooting

### **Problem 1: CORS Error**
```javascript
// Error: Access to fetch blocked by CORS policy

// Solution: API server đã có CORS enabled
// Nếu vẫn lỗi, check API server logs
```

### **Problem 2: Model Loading Error**
```bash
# Error: Model directory not found

# Solution: Đảm bảo saved_model/ folder tồn tại
ls saved_model/
# Should contain: pytorch_model.bin, tokenizer.json, etc.
```

### **Problem 3: Slow Predictions**
```python
# Problem: Predictions take too long

# Solutions:
# 1. Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Reduce max_length
MAX_LENGTH = 256  # Instead of 512

# 3. Use model quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

---

## 📋 Integration Checklist

### **Backend Setup:**
- [ ] API server running on port 5000
- [ ] Model loaded successfully
- [ ] Health check returns OK
- [ ] CORS enabled for frontend domain

### **Frontend Integration:**
- [ ] Chart.js library loaded
- [ ] API calls working
- [ ] Charts rendering correctly
- [ ] Error handling implemented

### **Production Ready:**
- [ ] Docker container built
- [ ] Nginx reverse proxy configured
- [ ] SSL certificate installed
- [ ] Monitoring and logging setup

---

## 🎯 Kết quả mong đợi

Sau khi tích hợp thành công:

### **User Experience:**
1. User viết comment trong forum
2. Hệ thống tự động phân tích cảm xúc
3. Hiển thị biểu đồ tròn sentiment (Tích cực/Tiêu cực)
4. Hiển thị biểu đồ cột emotions chi tiết
5. Show badges cho emotions được detect

### **Admin Dashboard:**
- Thống kê sentiment của forum theo thời gian
- Top emotions được detect nhiều nhất
- Phân tích xu hướng cảm xúc users

### **Business Value:**
- Hiểu rõ hơn về cảm xúc customers
- Phát hiện sớm feedback tiêu cực
- Cải thiện customer experience
- Data-driven decision making

---

**Created:** 2026-04-21  
**Version:** 1.0  
**API Server:** http://localhost:5000  
**Demo:** web_demo.html