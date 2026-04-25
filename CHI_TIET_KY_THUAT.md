# ⚙️ Chi Tiết Kỹ Thuật (Technical Details)

Tài liệu này giải thích các thành phần kỹ thuật cốt lõi của hệ thống phân loại cảm xúc.

---

## 🧠 1. Kiến Trúc Model (Hybrid PhoBERT)

Hệ thống sử dụng mô hình **PhoBERT** kết hợp với các lớp học sâu bổ trợ để tối ưu cho tiếng Việt.

### So sánh kiến trúc:
| Tính năng | PhoBERT + BiLSTM + Attention (Hiện tại) | BERT Base (Cũ) |
| :--- | :--- | :--- |
| **Pre-trained** | PhoBERT (VinAI - Tiếng Việt) | BERT (Google - Tiếng Anh) |
| **Ngữ cảnh** | BiLSTM (2 chiều: trái->phải & phải->trái) | Không có |
| **Trọng tâm** | Self-Attention (Tập trung từ khóa) | Không có |
| **Độ chính xác** | Cao nhất (Tăng 30-50%) | Trung bình |

### Cơ chế hoạt động:
1.  **PhoBERT Encoder**: Chuyển đổi văn bản tiếng Việt thành các vector đặc trưng (768 chiều).
2.  **BiLSTM**: Hiểu ngữ cảnh của từ dựa trên các từ đứng trước và đứng sau nó.
3.  **Self-Attention**: Tự động xác định các từ quan trọng (ví dụ: "rất vui", "thất vọng") để tập trung phân tích.
4.  **Classification Layer**: Đưa ra xác suất cho 16 nhãn cảm xúc khác nhau.

---

## 📚 2. Học Chuyển Đổi (Transfer Learning)

Thay vì học từ đầu, model mới sẽ kế thừa "kiến thức" từ model tốt nhất trước đó.

*   **Lợi ích**:
    *   **Nhanh hơn**: Chỉ cần 3-5 epochs thay vì 15-20 epochs.
    *   **Chính xác hơn**: Không bị mất kiến thức từ các lần training cũ.
    *   **Tiết kiệm**: Cần ít dữ liệu mới hơn để đạt hiệu quả cao.
*   **Cơ chế**: Khi bạn chạy `train_simple.py`, hệ thống tự động tìm model có F1 Score cao nhất trong Registry để làm nền tảng.

---

## 💾 3. Quản Lý Model (Model Registry)

Hệ thống có cơ chế quản lý thông minh để tránh lãng phí bộ nhớ.

*   **Chính sách "Keep Only Best"**: 
    *   Mỗi model nặng ~438MB. Nếu lưu tất cả, dung lượng sẽ tăng rất nhanh.
    *   Hệ thống chỉ giữ lại **duy nhất 1 model tốt nhất** trên đĩa. Các model kém hơn sẽ bị xóa bỏ tự động sau khi so sánh.
*   **Registry Metadata**: File `model_registry/registry.json` lưu giữ lịch sử chỉ số (metrics) của tất cả các lần training để team tiện theo dõi.

---

## 🔄 4. Chia Sẻ Model Qua Cloud (Cloud Sharing)

Vì model quá nặng để đẩy lên GitHub, chúng ta sử dụng **Hugging Face Model Hub**.

*   **Repository**: `emotion-classification-vn/emotion-classification`.
*   **Cơ chế Sync**: 
    *   Khi member training xong, model tốt nhất được đẩy (upload) lên Hugging Face.
    *   Khi member khác chạy lệnh training, hệ thống tự động tải (download) model đó về nếu máy local chưa có.
*   **Bảo mật**: Sử dụng Hugging Face Token để xác thực quyền truy cập.

---

## 📦 5. Xử Lý Dữ Liệu (Data Processing)

*   **Auto-Merge**: Tự động gộp tất cả các file `member_*.csv` trong thư mục `data/` thành một bộ dữ liệu lớn.
*   **Deduplication**: Tự động loại bỏ các câu trùng lặp để tránh model bị "học vẹt".
*   **Data Tracker**: Theo dõi dữ liệu nào đã được dùng để training, đảm bảo không phí thời gian học lại dữ liệu cũ.

---
*Dự án sử dụng công nghệ tiên tiến nhất để mang lại hiệu quả phân loại cảm xúc tốt nhất cho tiếng Việt.*
