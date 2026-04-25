# 📜 Lịch Sử Cập Nhật (Changelog)

Theo dõi các thay đổi và cải tiến chính của dự án qua các phiên bản.

---

## 🚀 Phiên Bản 2.0: Tích Hợp PhoBERT (22/04/2026)

Đây là bản cập nhật lớn nhất, chuyển đổi toàn bộ hệ thống sang sử dụng mô hình tối ưu cho tiếng Việt.

### ✨ Tính năng mới:
*   **PhoBERT làm Default**: Thay thế BERT base bằng PhoBERT của VinAI.
*   **Kiến trúc BiLSTM + Attention**: Giúp model hiểu ngữ cảnh tiếng Việt sâu sắc hơn.
*   **Trực quan hóa Attention**: Xem được model đang tập trung vào từ khóa nào khi dự đoán.
*   **Tự động hóa hoàn toàn**: Script `train_simple.py` gộp tất cả các bước (merge, transfer learning, train, sync cloud).

### 📊 Cải thiện chỉ số:
*   **Độ chính xác (F1 Score)**: Tăng 30-50% so với bản cũ.
*   **Khả năng hiểu tiếng Việt**: Vượt trội hoàn toàn so với mô hình đa ngôn ngữ.

---

## ☁️ Cập Nhật: Cloud Sharing & Auto-Sync (23/04/2026)

*   **Hugging Face Hub**: Tích hợp làm kho lưu trữ model trung tâm cho cả team.
*   **Auto-Sync**: Model tốt nhất được tự động đẩy lên Cloud sau khi training và tự động tải về máy member khi cần.
*   **Dọn dẹp tự động**: Hệ thống tự động xóa các model cũ/kém hơn để tiết kiệm dung lượng đĩa.

---

## 📅 Phiên Bản 1.5: Quản Lý Dữ Liệu Thông Minh (21/04/2026)

*   **Data Tracker**: Theo dõi dữ liệu đã train để tránh học lặp lại.
*   **Transfer Learning**: Cho phép model học tiếp từ kiến thức cũ thay vì học lại từ đầu.
*   **Model Registry**: Lưu trữ metadata và chỉ số của tất cả các lần thử nghiệm.

---

## 🌱 Phiên Bản 1.0: Khởi Tạo Dự Án (20/04/2026)

*   Thiết lập cấu trúc thư mục và pipeline training cơ bản.
*   Sử dụng BERT-base-uncased (Multilingual).
*   Xây dựng hệ thống thu thập dữ liệu qua file CSV.

---
*Dự án liên tục được cập nhật để mang lại trải nghiệm tốt nhất cho các thành viên và độ chính xác cao nhất cho mô hình.*
