# 🎭 Phân Loại Cảm Xúc Tiếng Việt (Vietnamese Emotion Classification)

Hệ thống phân loại cảm xúc đa nhãn cho tiếng Việt sử dụng PhoBERT + BiLSTM + Attention.

---

## 📚 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Bắt Đầu Nhanh](#-bắt-đầu-nhanh)
- [Quy Trình Đóng Góp](#-quy-trình-đóng-góp)
- [Lệnh Thường Dùng](#-lệnh-thường-dùng)
- [Kiến Trúc Kỹ Thuật](#-kiến-trúc-kỹ-thuật)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Giới Thiệu

### Tính Năng Chính
- ✅ **PhoBERT Hybrid**: Tối ưu cho tiếng Việt với BiLSTM + Self-Attention
- ✅ **Transfer Learning**: Học tiếp từ model tốt nhất, không học lại từ đầu
- ✅ **Auto-Sync Cloud**: Tự động đồng bộ model qua Hugging Face Hub
- ✅ **Multi-Label**: Hỗ trợ 16 cảm xúc, một câu có thể có nhiều cảm xúc
- ✅ **Smart Data Tracking**: Chỉ train dữ liệu mới, tránh lãng phí thời gian

### 16 Cảm Xúc Được Hỗ Trợ
`joy`, `trust`, `fear`, `surprise`, `sadness`, `disgust`, `anger`, `anticipation`, `love`, `optimism`, `pessimism`, `anxiety`, `empathy`, `sympathy`, `pride`, `shame`

---

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt Môi Trường

```bash
# Clone repository
git clone https://github.com/Escanor292/PhanLoaiCamXuc.git
cd PhanLoaiCamXuc

# Cài đặt dependencies
pip install -r requirements.txt
pip install huggingface_hub

# Đăng nhập Hugging Face (để sync model)
huggingface-cli login
# Nhập token có quyền WRITE
```

### 2. Tạo File Dữ Liệu Cá Nhân

```bash
# Windows
copy data\TEMPLATE_DONG_GOP_DATA.csv data\member_TenBan.csv

# Linux/Mac
cp data/TEMPLATE_DONG_GOP_DATA.csv data/member_TenBan.csv
```

### 3. Thêm Dữ Liệu

Mở `data/member_TenBan.csv` và thêm dữ liệu theo format:

| text | joy | trust | fear | surprise | sadness | ... |
|------|-----|-------|------|----------|---------|-----|
| "Tôi rất vui vì được thăng chức!" | 1 | 1 | 0 | 1 | 0 | ... |
| "Thất vọng quá, dự án bị hủy rồi" | 0 | 0 | 0 | 0 | 1 | ... |

**Quy tắc:**
- `1` = Có cảm xúc đó
- `0` = Không có cảm xúc đó
- Một câu có thể có nhiều cảm xúc (multi-label)

### 4. Training Model

```bash
# Chỉ cần 1 lệnh duy nhất!
python train_simple.py
```

**Hệ thống tự động:**
- ✅ Gộp tất cả dữ liệu từ `data/member_*.csv`
- ✅ Tải model tốt nhất từ Hugging Face (Transfer Learning)
- ✅ Training với dữ liệu mới
- ✅ Đẩy model lên cloud nếu tốt hơn model cũ

### 5. Push Kết Quả

```bash
git add data/member_TenBan.csv model_registry/registry.json
git commit -m "Training results from [Tên]: F1 Score 0.8xxx"
git push
```

---

## 🔄 Quy Trình Đóng Góp

### Workflow Hàng Ngày

```bash
# 1. Cập nhật code mới nhất
git pull

# 2. Thêm dữ liệu vào file của bạn
# Mở data/member_TenBan.csv và thêm dữ liệu

# 3. Training
python train_simple.py

# 4. Push kết quả
git add data/member_TenBan.csv model_registry/
git commit -m "Add 50 new samples, F1: 0.85"
git push
```

### Nguyên Tắc Làm Việc

1. **Isolation**: Mỗi member chỉ chỉnh sửa file CSV của mình
2. **Knowledge Sharing**: Kiến thức được chia sẻ qua Model, không qua CSV
3. **Auto-Sync**: Model tốt nhất tự động đồng bộ cho cả team
4. **No Duplication**: Hệ thống tự động loại bỏ dữ liệu trùng lặp

---

## ⚡ Lệnh Thường Dùng

### Training & Testing

```bash
# Training đơn giản (khuyên dùng)
python train_simple.py

# Training với tham số tùy chỉnh
python train_with_args.py --epochs 10 --lr 2e-5 --batch-size 32

# Demo dự đoán (interactive)
python demo_phobert.py --mode interactive

# Demo dự đoán (batch)
python demo_phobert.py --mode batch

# Test nhanh một câu
python my_test.py
```

### Quản Lý Model

```bash
# Liệt kê tất cả models và F1 scores
python model_registry.py list

# So sánh các experiments
python compare_experiments.py

# Sync model từ Hugging Face
python model_sharing.py sync
```

### Quản Lý Dữ Liệu

```bash
# Xem thống kê dữ liệu
python data_tracker.py stats

# Gộp dữ liệu thủ công
python merge_data.py

# Reset tracker (train lại từ đầu)
python data_tracker.py reset
```

### API Server

```bash
# Chạy API server (Production)
python api_server.py

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôi rất vui hôm nay!"}'
```

---

## 🧠 Kiến Trúc Kỹ Thuật

### Phương Pháp Học

**Deep Learning + Supervised Learning:**
- **Deep Learning**: PhoBERT với hàng triệu tham số, tự động trích xuất đặc trưng ngữ nghĩa
- **Supervised Learning**: Học từ dữ liệu đã gán nhãn, liên tục điều chỉnh weights qua các epochs

### Kiến Trúc Model (Hybrid PhoBERT)

```
Input Text (Tiếng Việt)
    ↓
PhoBERT Encoder (768-dim vectors)
    ↓
BiLSTM Layer (Hiểu ngữ cảnh 2 chiều)
    ↓
Self-Attention (Tập trung từ khóa quan trọng)
    ↓
Classification Layer (16 emotions)
    ↓
Output Probabilities
```

**So sánh với BERT Base:**

| Tính năng | PhoBERT Hybrid | BERT Base |
|-----------|----------------|-----------|
| Pre-trained | PhoBERT (VinAI - Tiếng Việt) | BERT (Google - Tiếng Anh) |
| Ngữ cảnh | BiLSTM 2 chiều | Không có |
| Attention | Self-Attention | Không có |
| Độ chính xác | Cao nhất (+30-50%) | Trung bình |

### Transfer Learning

**Lợi ích:**
- ⚡ **Nhanh hơn**: 3-5 epochs thay vì 15-20 epochs
- 🎯 **Chính xác hơn**: Kế thừa kiến thức từ model cũ
- 💾 **Tiết kiệm**: Cần ít dữ liệu mới hơn

**Cơ chế:** Model mới học tiếp từ model tốt nhất trong Registry, không học lại từ đầu.

### Model Registry

**Chính sách "Keep Only Best":**
- Mỗi model ~438MB
- Chỉ giữ 1 model tốt nhất trên đĩa
- Metadata của tất cả models được lưu trong `registry.json`

### Cloud Sharing (Hugging Face Hub)

**Repository:** `emotion-classification-vn/emotion-classification`

**Cơ chế:**
1. Member training xong → Model tốt nhất upload lên Hugging Face
2. Member khác training → Tự động download model về (nếu chưa có)
3. Xác thực qua Hugging Face Token

---

## 🐛 Troubleshooting

### Lỗi Thường Gặp

**1. "No CSV files found"**
```bash
# Đảm bảo đã tạo file dữ liệu
copy data\TEMPLATE_DONG_GOP_DATA.csv data\member_TenBan.csv
```

**2. Lỗi đồng bộ model**
```bash
# Đăng nhập lại Hugging Face
huggingface-cli login
# Kiểm tra kết nối Internet
```

**3. Git Conflict**
```bash
# Pull trước khi push
git pull
# Giải quyết conflict (nếu có)
git add .
git commit -m "Resolve conflict"
git push
```

**4. Lỗi DLL/Access Denied (Windows)**
- Liên hệ IT để whitelist:
  - Thư mục dự án `PhanLoaiCamXuc`
  - File `python.exe`
  - Thư viện PyTorch (`site-packages/torch/lib`)
- Chạy PowerShell/CMD với quyền Administrator

**5. Out of Memory**
```bash
# Giảm batch size
python train_with_args.py --batch-size 8

# Hoặc giảm max_length
python train_with_args.py --max-length 256
```

### Kiểm Tra Sức Khỏe Hệ Thống

```bash
# Kiểm tra môi trường Windows
python windows_doctor.py

# Kiểm tra trước khi push
python check_before_push.py
```

---

## 📊 Experiment Tracking

Xem chi tiết trong [experiments/README.md](experiments/README.md)

### Quick Start

```bash
# Tạo experiment mới
mkdir -p experiments/member_an/exp001_baseline

# Training với experiment tracking
python train_with_args.py \
    --output experiments/member_an/exp001_baseline/ \
    --experiment-name "An - Baseline Model"

# So sánh experiments
python compare_experiments.py \
    experiments/member_an/exp001_baseline/ \
    experiments/member_an/exp002_tuning/
```

---

## 📖 Tài Liệu Bổ Sung

- **[LICH_SU_CAP_NHAT.md](LICH_SU_CAP_NHAT.md)** - Lịch sử các phiên bản
- **[experiments/README.md](experiments/README.md)** - Hướng dẫn experiment tracking
- **[experiments/experiment_log.md](experiments/experiment_log.md)** - Log các experiments

---

## 🤝 Đóng Góp

### Tips Tạo Dữ Liệu Tốt

✅ **Nên:**
- Viết câu tự nhiên, đa dạng chủ đề
- Đánh nhãn chính xác
- Độ dài 10-50 từ
- Kiểm tra chính tả

❌ **Không nên:**
- Copy-paste từ Internet
- Câu quá ngắn (< 5 từ)
- Câu quá dài (> 50 từ)
- Đánh nhãn sai

### Quy Tắc Git

- Commit message rõ ràng: `"Add 50 samples, F1: 0.85"`
- Không commit file model (quá lớn)
- Pull trước khi push
- Giải quyết conflict ngay lập tức

---

## 📞 Liên Hệ & Hỗ Trợ

- **GitHub Issues**: [Report bugs](https://github.com/Escanor292/PhanLoaiCamXuc/issues)
- **Team Lead**: [Thông tin liên hệ]

---

## 📜 License

[Thêm license của dự án]

---

**Phiên bản:** 2.0  
**Cập nhật lần cuối:** 25/04/2026  
**Người duy trì:** Team Emotion Classification

---

*Chúc bạn training thành công! 🚀*
