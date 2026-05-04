# ⚙️ Chi Tiết Kỹ Thuật (Technical Details)

Tài liệu này giải thích các thành phần kỹ thuật cốt lõi của hệ thống phân loại cảm xúc.

---

## 🤖 1. Phân Loại Phương Pháp Học (Learning Methodology)

Dự án này sử dụng phương pháp **Deep Learning (Học sâu)** kết hợp với **Supervised Learning (Học có giám sát)**:

- **Deep Learning (Học sâu)**: Mô hình của chúng ta (PhoBERT-Hybrid) sử dụng kiến trúc Transformer với hàng chục triệu tham số và nhiều lớp mạng nơ-ron (neural network) phức tạp. Các mạng này có khả năng tự động trích xuất các đặc trưng ngữ nghĩa từ văn bản (NLP) sâu sắc hơn rất nhiều so với các thuật toán Machine Learning truyền thống.
- **Học có giám sát (Supervised Learning)**: Quá trình huấn luyện mô hình (training) được thực hiện dựa trên các bộ dữ liệu **đã được gán nhãn sẵn** (ví dụ: các comment có đi kèm nhãn cảm xúc cụ thể như Tích cực, Tiêu cực,...). Mô hình học bằng cách đọc đầu vào, đưa ra dự đoán, so sánh với nhãn gốc (đáp án chuẩn) để tính toán sai số, và liên tục điều chỉnh các trọng số (weights) nhằm tăng độ chính xác qua các vòng lặp (epochs).

---

## 🧠 2. Kiến Trúc Model (Hybrid PhoBERT)

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

## 📚 3. Học Chuyển Đổi (Transfer Learning)

Thay vì học từ đầu, model mới sẽ kế thừa "kiến thức" từ model tốt nhất trước đó.

*   **Lợi ích**:
    *   **Nhanh hơn**: Chỉ cần 3-5 epochs thay vì 15-20 epochs.
    *   **Chính xác hơn**: Không bị mất kiến thức từ các lần training cũ.
    *   **Tiết kiệm**: Cần ít dữ liệu mới hơn để đạt hiệu quả cao.
*   **Cơ chế**: Khi bạn chạy `train_simple.py`, hệ thống tự động tìm model có F1 Score cao nhất trong Registry để làm nền tảng.

---

## 💾 4. Quản Lý Model (Model Registry)

Hệ thống có cơ chế quản lý thông minh để tránh lãng phí bộ nhớ.

*   **Chính sách "Keep Only Best"**: 
    *   Mỗi model nặng ~438MB. Nếu lưu tất cả, dung lượng sẽ tăng rất nhanh.
    *   Hệ thống chỉ giữ lại **duy nhất 1 model tốt nhất** trên đĩa. Các model kém hơn sẽ bị xóa bỏ tự động sau khi so sánh.
*   **Registry Metadata**: File `model_registry/registry.json` lưu giữ lịch sử chỉ số (metrics) của tất cả các lần training để team tiện theo dõi.

---

## 🔄 5. Chia Sẻ Model Qua Cloud (Cloud Sharing)

Vì model quá nặng để đẩy lên GitHub, chúng ta sử dụng **Hugging Face Model Hub**.

*   **Repository**: `emotion-classification-vn/emotion-classification`.
*   **Cơ chế Sync**: 
    *   Khi member training xong, model tốt nhất được đẩy (upload) lên Hugging Face.
    *   Khi member khác chạy lệnh training, hệ thống tự động tải (download) model đó về nếu máy local chưa có.
*   **Bảo mật**: Sử dụng Hugging Face Token để xác thực quyền truy cập.

---

## 📦 6. Xử Lý Dữ Liệu (Data Processing)

*   **Auto-Merge**: Tự động gộp tất cả các file `member_*.csv` trong thư mục `data/` thành một bộ dữ liệu lớn.
*   **Deduplication**: Tự động loại bỏ các câu trùng lặp để tránh model bị "học vẹt".
*   **Data Tracker**: Theo dõi dữ liệu nào đã được dùng để training, đảm bảo không phí thời gian học lại dữ liệu cũ.

---
*Dự án sử dụng công nghệ tiên tiến nhất để mang lại hiệu quả phân loại cảm xúc tốt nhất cho tiếng Việt.*
