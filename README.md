# 🎭 Phân Loại Cảm Xúc Đa Nhãn (Multi-label Emotion Classification)

Hệ thống phân tích cảm xúc tiếng Việt sử dụng mô hình **Hybrid PhoBERT** mạnh mẽ nhất hiện nay. Hệ thống có khả năng nhận diện đồng thời 16 cảm xúc khác nhau trong một câu bình luận.

---

## ⚡ 1. Bắt Đầu Nhanh (Quick Start)

Dành cho các thành viên đã có dữ liệu trong folder `data/`:

```bash
# 1. Cập nhật code và model mới nhất từ cả team
git pull

# 2. Huấn luyện model trên dữ liệu của bạn (Tự động 100%)
python train_simple.py

# 3. Đẩy kết quả thông số lên GitHub
git add model_registry/registry.json
git commit -m "Cập nhật kết quả training"
git push
```

---

## ✨ 2. Tính Năng Nổi Bật

*   **Hybrid PhoBERT**: Tối ưu hóa cho tiếng Việt với kiến trúc BiLSTM + Attention.
*   **Transfer Learning**: Model tự động học tiếp từ kiến thức của cả team, không cần học lại từ đầu.
*   **Auto-Sync Cloud**: Tự động đồng bộ file model nặng qua Hugging Face Hub.
*   **Dọn Dẹp Thông Minh**: Chỉ giữ lại model tốt nhất trên đĩa để tiết kiệm 85% dung lượng.
*   **Trực Quan Hóa**: Xem model tập trung vào từ khóa nào khi dự đoán cảm xúc.

---

## 📚 3. Tài Liệu Hướng Dẫn

Để dự án gọn gàng, hướng dẫn được chia thành các file chính sau:

1.  📖 **[HUONG_DAN_THANH_VIEN.md](HUONG_DAN_THANH_VIEN.md)**: Hướng dẫn đóng góp dữ liệu, cài đặt môi trường và quy trình làm việc hàng ngày.
2.  ⚙️ **[CHI_TIET_KY_THUAT.md](CHI_TIET_KY_THUAT.md)**: Giải thích sâu về kiến trúc model, cơ chế Transfer Learning và quản lý Registry.
3.  ⚡ **[LENH_THUONG_DUNG.md](LENH_THUONG_DUNG.md)**: Tra cứu nhanh các câu lệnh training, kiểm tra model và demo.
4.  📜 **[LICH_SU_CAP_NHAT.md](LICH_SU_CAP_NHAT.md)**: Theo dõi các phiên bản và cải tiến của hệ thống qua thời gian.

---

## 🛠️ 4. Cài Đặt (Installation)

### Yêu cầu:
*   Python 3.8 trở lên
*   Cài đặt thư viện: `pip install -r requirements.txt`
*   Đăng nhập Hugging Face: `hf auth login` (để tải/đẩy model)

---

## 🎮 5. Chạy Demo Tương Tác

Bạn muốn thử xem model dự đoán như thế nào? Chạy ngay:

```bash
python demo_phobert.py --mode interactive
```

Sau đó nhập một câu tiếng Việt bất kỳ, ví dụ: *"Sản phẩm này tuyệt vời quá, tôi rất hài lòng!"*

---
*Dự án được phát triển bởi đội ngũ đam mê NLP, tập trung vào việc hiểu sâu sắc cảm xúc người Việt trên mạng xã hội.*
