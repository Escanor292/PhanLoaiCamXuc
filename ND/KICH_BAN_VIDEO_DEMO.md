# Kịch Bản & Hướng Dẫn Quay Video Demo Giải Thích Code
*(Dành cho Đồ án Phân loại đa nhãn cảm xúc tiếng Việt)*

---

## 🎬 TỔNG QUAN VIDEO (Thời lượng dự kiến: 3 - 5 phút)
*   **Mục tiêu:** Giải thích ngắn gọn, súc tích kiến trúc mã nguồn của dự án để giảng viên hoặc người xem hiểu ngay cách hoạt động.
*   **Phong cách:** Quay màn hình code (VS Code), vừa nói vừa chỉ chuột vào các đoạn code chính, giải thích bằng ngôn ngữ trực quan, tránh lý thuyết rườm rà.

---

## 🎥 PHẦN 1: GIỚI THIỆU CHUNG (0:00 - 0:45)
*(Hình ảnh: Hiển thị giao diện ứng dụng hoặc cấu trúc thư mục dự án trong VS Code)*

*   **Lời thoại gợi ý:**
    > *"Xin chào thầy cô và các bạn. Hôm nay mình sẽ demo và giải thích nhanh cấu trúc mã nguồn của **Hệ thống phân loại đa nhãn cảm xúc tiếng Việt**. Dự án của tụi mình sử dụng kiến trúc học sâu kết hợp giữa **PhoBERT (của VinAI)**, mạng **BiLSTM** và cơ chế **Attention** để nhận diện đồng thời nhiều cảm xúc trong một câu bình luận."*
*   **Các thành phần chính cần chỉ trên màn hình:**
    *   Thư mục `data/`: Chứa các file dữ liệu cảm xúc đã gán nhãn của các thành viên.
    *   File `dataset.py`: Đọc và tiền xử lý dữ liệu.
    *   File `model_phobert.py`: Định nghĩa bộ não AI (kiến trúc mạng).
    *   File `train_minimal.py`: Huấn luyện và đánh giá mô hình.
    *   File `api_server.py`: Cung cấp dịch vụ API Flask để tích hợp vào các hệ thống khác (web, chatbot).
    *   File `predict.py` & `Pro_Edition/gui_app.py`: Giao diện ứng dụng và chạy dự đoán.

---

## 🎥 PHẦN 2: TIỀN XỬ LÝ DỮ LIỆU (`dataset.py`) (0:45 - 1:30)
*(Hình ảnh: Mở file [dataset.py](file:///d:/PhanLoaiCamXuc/dataset.py))*

*   **Đoạn code cần focus:** Lớp `EmotionDataset` và hàm `__getitem__`.
*   **Giải thích code:**
    1.  **Đầu vào:** Nhận vào danh sách các câu `texts` và ma trận nhãn `labels` 16 cột (16 cảm xúc: joy, sadness, fear,...).
    2.  **Hàm `__getitem__`:** 
        *   Thay vì tokenize toàn bộ dữ liệu từ đầu gây tốn RAM, dự án sử dụng cơ chế **Tokenization on-the-fly** (tokenize trực tiếp khi nạp từng batch dữ liệu).
        *   Sử dụng PhoBERT Tokenizer để chuyển chữ tiếng Việt thành chuỗi số `input_ids` và tạo `attention_mask` (giúp mô hình biết từ nào là thật, từ nào là khoảng trắng padding).
        *   Trả về Tensor kiểu PyTorch sẵn sàng cho quá trình huấn luyện.

---

## 🎥 PHẦN 3: KIẾN TRÚC MÔ HÌNH HYBRID (`model_phobert.py`) (1:30 - 3:00)
*(Hình ảnh: Mở file [model_phobert.py](file:///d:/PhanLoaiCamXuc/model_phobert.py))*

*   **Đoạn code cần focus:** Class `HybridEmotionClassifier` và hàm `forward`.
*   **Giải thích code (Trọng tâm của video):**
    > *"Mô hình của tụi mình là một mô hình **Hybrid (Lai ghép)** kết hợp 2 luồng thông tin song song để tối đa hóa khả năng hiểu ngữ nghĩa:"*
    1.  **Nhánh 1 (Global Context):** Trích xuất vector đại diện cho cả câu bằng cách lấy token đầu tiên `[CLS]` từ đầu ra của PhoBERT.
    2.  **Nhánh 2 (Sequential & Focused Context):** 
        *   Đưa toàn bộ chuỗi từ qua mạng **BiLSTM** (Long Short-Term Memory 2 chiều). Mạng này giúp đọc câu từ trái sang phải và từ phải sang trái để hiểu mối quan hệ xa giữa các từ (ví dụ: từ phủ định đứng trước từ cảm xúc).
        *   Sau đó đi qua cơ chế **Self-Attention** (`AttentionLayer`) để tính toán trọng số. Từ nào mang nhiều cảm xúc (ví dụ: "vui", "tệ", "thất vọng") sẽ có trọng số attention cao hơn, giúp mô hình tập trung vào đúng trọng tâm.
    3.  **Hợp nhất thông tin (`torch.cat`):** Nối (concatenate) hai vector của Nhánh 1 và Nhánh 2 lại với nhau để tạo ra biểu diễn câu hoàn chỉnh nhất.
    4.  **Phân loại:** Đưa qua lớp Linear Classifier để dự đoán xác suất của 16 cảm xúc. Vì là bài toán đa nhãn, hàm kích hoạt cuối cùng là **Sigmoid** kết hợp với hàm loss **BCEWithLogitsLoss** thay vì Softmax.

---

## 🎥 PHẦN 4: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH (`train_minimal.py`) (3:00 - 4:00)
*(Hình ảnh: Mở file [train_minimal.py](file:///d:/PhanLoaiCamXuc/train_minimal.py))*

*   **Đoạn code cần focus:** Đoạn chia dữ liệu `train_test_split` và hàm `evaluate`.
*   **Giải thích code:**
    1.  **Chia dữ liệu:** Chia dữ liệu theo tỷ lệ **70% để huấn luyện (Train)**, **15% để thi thử kiểm định (Validation)** và **15% để thi thật (Test)**.
    2.  **Đánh giá:**
        *   Trong quá trình train, sau mỗi vòng (epoch), mô hình sẽ tự kiểm tra trên tập validation. Nếu điểm F1 tốt hơn, mô hình sẽ được lưu lại (tránh học vẹt).
        *   Sau khi train xong, mô hình được kiểm tra lần cuối trên tập **Test độc lập** (dữ liệu mô hình chưa từng được nhìn thấy).
        *   Sử dụng chỉ số **Macro F1-Score** để đánh giá độ chính xác trung bình trên 16 lớp cảm xúc và **Hamming Loss** để kiểm tra tỷ lệ đoán lệch nhãn.

---

## 🎥 PHẦN 5: TÍCH HỢP HỆ THỐNG QUA API (`api_server.py`) (4:00 - 4:45)
*(Hình ảnh: Mở file [api_server.py](file:///d:/PhanLoaiCamXuc/api_server.py))*

*   **Đoạn code cần focus:** Các hàm định nghĩa API và route của Flask như `predict_emotions`, `@app.route('/predict', ...)`.
*   **Giải thích code:**
    1.  **Framework Flask & CORS:** Sử dụng Flask để tạo web server gọn nhẹ và thư viện CORS để cho phép các trang web bên ngoài (cross-origin) gọi đến API này mà không bị chặn bảo mật.
    2.  **API chính `/predict` (HTTP POST):** 
        *   Nhận câu text cần phân tích từ yêu cầu gửi lên.
        *   Tokenize và đưa qua mô hình học sâu để dự đoán cảm xúc (như đã giải thích ở phần trước).
        *   Tự động phân nhóm và tính toán **Thái độ chung (Sentiment)**:
            *   *Tích cực (Positive):* Trung bình điểm của các nhãn vui vẻ, tin tưởng, yêu thương, tự hào, hào hứng, bình tĩnh...
            *   *Tiêu cực (Negative):* Trung bình điểm của buồn bã, giận dữ, thất vọng, lo lắng, ghê tởm, sợ hãi...
            *   *Trung lập (Neutral):* Điểm còn lại sau khi trừ đi điểm lớn nhất của Tích cực hoặc Tiêu cực.
        *   Trả về kết quả dưới định dạng JSON tiêu chuẩn, dễ dàng tích hợp vào diễn đàn (web forum), chatbot hoặc app di động.

---

## 🎥 PHẦN 6: CHẠY DEMO DỰ ĐOÁN & KẾT LUẬN (4:45 - End)
*(Hình ảnh: Chạy lệnh `python demo_prediction.py` hoặc mở app giao diện `gui_app.py` và gõ thử 1 vài câu. Hoặc mở giao diện web kết nối qua Flask API)*

*   **Cách demo:**
    1.  Gõ một câu có nhiều cảm xúc hỗn hợp (Ví dụ: *"Sản phẩm rất đẹp nhưng đóng gói hơi ẩu làm mình lo lắng"*).
    2.  Show kết quả AI dự đoán: Nó sẽ nhận diện ra cùng lúc các cảm xúc như `joy` (đẹp), `disappointed` (gói ẩu), `worried` (lo lắng) kèm theo độ tin cậy cụ thể.
*   **Lời kết:**
    > *"Như các bạn đã thấy, nhờ vào việc kết hợp PhoBERT hiểu sâu ngữ nghĩa tiếng Việt cùng với khả năng bắt ngữ cảnh của BiLSTM và cơ chế tập trung Attention, mô hình đã nhận diện rất chính xác các cảm xúc đa dạng trong câu. Bên cạnh đó, hệ thống cũng cung cấp sẵn các cổng API Flask giúp dễ dàng tích hợp dịch vụ phân tích cảm xúc này vào bất kỳ website hay ứng dụng nào khác. Cảm ơn thầy cô và các bạn đã lắng nghe!"*
