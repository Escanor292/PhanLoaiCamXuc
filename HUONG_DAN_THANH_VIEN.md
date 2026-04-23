# 👥 Hướng Dẫn Cho Thành Viên (Member Guide)

Tài liệu này tổng hợp tất cả các hướng dẫn cần thiết để bạn bắt đầu đóng góp dữ liệu và huấn luyện model trong dự án Phân Loại Cảm Xúc.

---

## 🎯 1. Quy Trình Tổng Quan (Workflow)

Hệ thống được thiết kế theo quy trình **Hybrid PhoBERT + Transfer Learning**, giúp việc đóng góp dữ liệu trở nên cực kỳ đơn giản:

1.  **Isolation (Cô lập)**: Mỗi member chỉ quản lý file CSV của riêng mình (`data/member_TenBan.csv`).
2.  **Knowledge Sharing**: Kiến thức được chia sẻ qua **Model**, không qua file CSV. Bạn không cần tải dữ liệu của người khác, chỉ cần tải Model đã "học" dữ liệu đó.
3.  **Vòng lặp hàng ngày**: `git pull` -> Thêm dữ liệu -> `python train_simple.py` -> `git push`.

---

## 🚀 2. Thiết Lập Lần Đầu (Setup)

### Bước 2.1: Cài đặt môi trường
1.  **Cài đặt Git**: Download từ [git-scm.com](https://git-scm.com/download/win).
2.  **Clone code**: 
    ```bash
    git clone https://github.com/Escanor292/PhanLoaiCamXuc.git
    cd PhanLoaiCamXuc
    ```
3.  **Cài đặt thư viện**:
    ```bash
    pip install -r requirements.txt
    pip install huggingface_hub
    ```
4.  **Đăng nhập Hugging Face** (để tự động đồng bộ model):
    ```bash
    hf auth login
    # Nhập token có quyền WRITE của bạn
    ```

### Bước 2.2: Tạo file dữ liệu cá nhân
```bash
copy data\TEMPLATE_DONG_GOP_DATA.csv data\member_TenCuaBan.csv
```

---

## 📊 3. Đóng Góp Dữ Liệu (Data Contribution)

Mở file `data/member_TenCuaBan.csv` và thêm các câu bình luận tiếng Việt theo định dạng sau:

| text | joy | trust | fear | surprise | ... (16 cảm xúc) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| "Tôi rất vui vì được thăng chức!" | 1 | 1 | 0 | 1 | ... |

*   **1**: Có cảm xúc đó.
*   **0**: Không có cảm xúc đó.
*   **Lưu ý**: Một câu có thể có nhiều nhãn cảm xúc cùng lúc.

### Tips tạo dữ liệu tốt:
*   ✅ Nên: Viết câu tự nhiên, đa dạng chủ đề, đánh nhãn chính xác.
*   ❌ Không nên: Copy-paste từ Internet, câu quá ngắn (< 5 từ) hoặc quá dài (> 50 từ).

---

## 🎓 4. Huấn Luyện Model (Training)

Chỉ cần chạy **1 câu lệnh duy nhất** để huấn luyện model trên dữ liệu mới của bạn:

```bash
python train_simple.py
```

**Hệ thống sẽ tự động:**
1.  Tự động gộp dữ liệu cá nhân của bạn.
2.  Tự động tải Model tốt nhất hiện tại từ Hugging Face về (Transfer Learning).
3.  Huấn luyện model tiếp tục dựa trên kiến thức cũ (nhanh và chính xác hơn).
4.  **Tự động đẩy (Sync) model lên Hugging Face** nếu kết quả tốt hơn model cũ.

---

## 🔄 5. Cập Nhật Kết Quả (Push Results)

Sau khi training thành công, bạn chỉ cần thực hiện các lệnh Git tiêu chuẩn để đồng bộ thông số (metrics) cho cả team:

```bash
# 1. Thêm kết quả vào registry
git add model_registry/registry.json

# 2. Commit và Push
git commit -m "Training results from [Tên Bạn]: F1 Score 0.8xxx"
git push
```

---

## 🐛 6. Giải Quyết Vấn Đề (Troubleshooting)

*   **Lỗi "No CSV files found"**: Đảm bảo bạn đã tạo file `member_TenCuaBan.csv` trong thư mục `data/`.
*   **Lỗi đồng bộ model**: Đảm bảo bạn đã chạy `hf auth login` và có kết nối Internet.
*   **Lỗi Conflict**: Nếu gặp conflict khi push, hãy chạy `git pull` trước, sau đó mới push lại.

---
*Chúc bạn đóng góp được nhiều dữ liệu chất lượng! 🚀*
