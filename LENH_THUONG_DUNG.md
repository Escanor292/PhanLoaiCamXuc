# ⚡ Lệnh Thường Dùng (Common Commands)

Bản tra cứu nhanh các câu lệnh trong dự án.

---

## 🔝 1. Lệnh Quan Trọng Nhất (Top Priority)
Sử dụng hàng ngày cho các thành viên:
```bash
# 1. Cập nhật code và thông số mới nhất
git pull

# 2. Huấn luyện model (Tự động gộp data, tải model cũ, training và sync cloud)
python train_simple.py

# 3. Đẩy thông số kết quả lên GitHub
git add model_registry/registry.json
git commit -m "Kết quả training của [Tên]"
git push
```

---

## 📊 2. Kiểm Tra Trạng Thái (Status Check)
```bash
# Liệt kê tất cả model trong Registry và chỉ số F1
python model_registry.py list

# Xem thống kê dữ liệu (đã train bao nhiêu câu, còn bao nhiêu câu mới)
python data_tracker.py stats

# So sánh chỉ số giữa các phiên bản model
python compare_experiments.py
```

---

## 🧪 3. Thử Nghiệm & Dự Đoán (Inference)
```bash
# Chạy demo tương tác (gõ câu nào dự đoán câu đó)
python demo_phobert.py --mode interactive

# Dự đoán thử một vài ví dụ có sẵn
python demo_phobert.py --mode batch

# Test nhanh một câu duy nhất
python my_test.py
```

---

## ⚙️ 4. Quản Lý Dữ Liệu (Data Management)
```bash
# Tự động gộp các file CSV (thường đã được tích hợp trong train_simple)
python merge_data.py

# Reset tracker (để train lại từ đầu toàn bộ dữ liệu)
python data_tracker.py reset

# Kiểm tra xem có dữ liệu nào mới chưa được train không
python data_tracker.py check
```

---

## ☁️ 5. Đồng Bộ Cloud (Cloud Sharing)
```bash
# Tải model tốt nhất từ Hugging Face về máy local
python model_sharing.py sync

# Tải một model cụ thể theo ID
python model_sharing.py download --model-id [ID_MODEL]

# Cấu hình repository Hugging Face
python model_sharing.py config --repo [USERNAME/REPO_NAME]
```

---

## 🛠️ 6. Lệnh Nâng Cao (Advanced)
```bash
# Huấn luyện với các tham số tùy chỉnh
python train_with_args.py --model-type hybrid --epochs 10 --lr 2e-5

# Tự động kiểm tra và sửa lỗi môi trường Windows (DLL, Quyền truy cập)
python windows_doctor.py

# Chạy API Server (Production)
python api_server.py
```

---
*Ghi chú: Nếu gặp lỗi, hãy chạy `git pull` để đảm bảo bạn đang dùng code mới nhất.*
