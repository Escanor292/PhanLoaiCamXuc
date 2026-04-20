# HƯỚNG DẪN ĐÓNG GÓP DỮ LIỆU TIẾNG VIỆT

## Tại sao cần dữ liệu tiếng Việt?

Model hiện tại được train trên tiếng Anh, nên độ chính xác với tiếng Việt rất thấp (40-50%).

**Khi có dữ liệu tiếng Việt:**
- Độ chính xác tăng lên 75-85%
- Model hiểu tiếng Việt tốt hơn
- Dự đoán chính xác hơn nhiều

**Mục tiêu:** Mỗi thành viên đóng góp 50-100 câu tiếng Việt

---

## Bước 1: Copy template

1. Copy file `data/TEMPLATE_DONG_GOP_DATA.csv`
2. Đổi tên thành `data/member_<tên_bạn>.csv`
   - Ví dụ: `data/member_nam.csv`, `data/member_linh.csv`

---

## Bước 2: Thêm dữ liệu của bạn

Mở file CSV và thêm các câu bình luận tiếng Việt.

### Format CSV:

```csv
comment,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,love,worried,disappointed,proud,embarrassed,jealous,calm,excited
"Câu bình luận của bạn",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

### 16 cảm xúc (Emotion Labels):

| Tiếng Việt | Tiếng Anh | Cột trong CSV |
|------------|-----------|---------------|
| vui vẻ | joy | joy |
| tin tưởng | trust | trust |
| sợ hãi | fear | fear |
| ngạc nhiên | surprise | surprise |
| buồn bã | sadness | sadness |
| ghê tởm | disgust | disgust |
| tức giận | anger | anger |
| mong đợi | anticipation | anticipation |
| yêu thương | love | love |
| lo lắng | worried | worried |
| thất vọng | disappointed | disappointed |
| tự hào | proud | proud |
| xấu hổ | embarrassed | embarrassed |
| ghen tị | jealous | jealous |
| bình tĩnh | calm | calm |
| phấn khích | excited | excited |

### Cách đánh nhãn:

- **1** = Câu này CÓ cảm xúc đó
- **0** = Câu này KHÔNG CÓ cảm xúc đó

**Lưu ý:** Một câu có thể có NHIỀU cảm xúc (multi-label)

---

## Ví dụ cụ thể:

### Ví dụ 1: "Tôi rất vui vì được thăng chức!"

Phân tích:
- ✅ vui vẻ (joy) = 1
- ✅ tin tưởng (trust) = 1
- ✅ ngạc nhiên (surprise) = 1
- ✅ tự hào (proud) = 1
- ✅ phấn khích (excited) = 1
- ❌ Các cảm xúc khác = 0

```csv
"Tôi rất vui vì được thăng chức!",1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1
```

### Ví dụ 2: "Tôi lo lắng về kỳ thi ngày mai"

Phân tích:
- ✅ sợ hãi (fear) = 1
- ✅ lo lắng (worried) = 1
- ❌ Các cảm xúc khác = 0

```csv
"Tôi lo lắng về kỳ thi ngày mai",0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0
```

### Ví dụ 3: "Tôi tức giận vì bị lừa dối"

Phân tích:
- ✅ ghê tởm (disgust) = 1
- ✅ tức giận (anger) = 1
- ✅ thất vọng (disappointed) = 1
- ❌ Các cảm xúc khác = 0

```csv
"Tôi tức giận vì bị lừa dối",0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0
```

---

## Bước 3: Lưu file

1. Lưu file CSV của bạn vào thư mục `data/`
2. Tên file: `data/member_<tên_bạn>.csv`
3. Commit và push lên GitHub:

```bash
git add data/member_<tên_bạn>.csv
git commit -m "Add data from <tên_bạn>"
git push
```

---

## Bước 4: Merge tất cả dữ liệu (Leader làm)

Khi tất cả thành viên đã đóng góp xong:

```bash
# Pull tất cả dữ liệu mới
git pull

# Merge tất cả file
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv

# Kiểm tra kết quả
# File master_dataset_vi.csv sẽ chứa tất cả dữ liệu đã merge
```

---

## Bước 5: Train model mới (Leader làm)

```bash
# Train với dữ liệu tiếng Việt
python train_with_args.py --data data/master_dataset_vi.csv --epochs 5 --register-model

# So sánh với model cũ
python compare_experiments.py
```

---

## Tips để tạo dữ liệu tốt:

### ✅ NÊN:
- Viết câu tự nhiên, như bình luận thật
- Đa dạng chủ đề: sản phẩm, dịch vụ, sự kiện, cảm xúc cá nhân
- Đa dạng độ dài: ngắn (5-10 từ) và dài (20-30 từ)
- Một câu có thể có nhiều cảm xúc
- Đánh nhãn chính xác

### ❌ KHÔNG NÊN:
- Copy paste từ internet (có thể vi phạm bản quyền)
- Câu quá ngắn: "Tốt", "Hay" (không đủ ngữ cảnh)
- Câu quá dài: > 50 từ (khó phân tích)
- Đánh nhãn sai hoặc thiếu

---

## Mục tiêu đề xuất:

| Số thành viên | Câu/người | Tổng câu | Độ chính xác dự kiến |
|---------------|-----------|----------|----------------------|
| 3 người | 50 câu | 150 câu | 65-75% |
| 5 người | 100 câu | 500 câu | 75-85% |
| 10 người | 100 câu | 1000 câu | 85-90% |

**Khuyến nghị:** Mỗi người nên đóng góp ít nhất 50 câu để có kết quả tốt.

---

## Câu hỏi thường gặp:

### Q1: Làm sao biết câu có cảm xúc nào?

**A:** Đọc câu và tự hỏi:
- Người viết đang cảm thấy gì?
- Nếu tôi là người viết, tôi sẽ cảm thấy thế nào?
- Có nhiều cảm xúc cùng lúc không?

### Q2: Một câu có thể có bao nhiêu cảm xúc?

**A:** Không giới hạn! Một câu có thể có 1-5 cảm xúc.

Ví dụ: "Tôi vui nhưng cũng lo lắng về tương lai"
- vui vẻ = 1
- lo lắng = 1
- mong đợi = 1

### Q3: Nếu không chắc chắn về cảm xúc?

**A:** 
- Nếu không chắc → đánh 0
- Chỉ đánh 1 khi BẠN CHẮC CHẮN câu đó có cảm xúc đó

### Q4: Có cần viết câu dài không?

**A:** Không nhất thiết. Câu ngắn (10-20 từ) cũng tốt, miễn là rõ ràng.

Ví dụ tốt:
- "Sản phẩm này tuyệt vời!" (ngắn, rõ ràng)
- "Tôi rất hài lòng với dịch vụ chăm sóc khách hàng" (dài, chi tiết)

---

## Liên hệ:

Nếu có thắc mắc, hỏi leader hoặc xem file `HUONG_DAN_CHO_THANH_VIEN.md`
