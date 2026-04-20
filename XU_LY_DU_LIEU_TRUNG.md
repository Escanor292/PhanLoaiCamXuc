# XỬ LÝ DỮ LIỆU TRÙNG VÀ CONFLICT

## Vấn đề

### Vấn đề 1: Dữ liệu trùng hoàn toàn

**Ví dụ:**
```csv
# File member_nam.csv
"Tôi rất vui!",1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

# File member_linh.csv
"Tôi rất vui!",1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

**Kết quả:** Cùng câu, cùng nhãn → Trùng lặp

---

### Vấn đề 2: Conflict (Câu trùng, nhãn khác) ⚠️

**Ví dụ:**
```csv
# File member_nam.csv
"Tôi rất vui!",1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0  ← joy=1

# File member_linh.csv
"Tôi rất vui!",1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0  ← joy=1, sadness=1
```

**Kết quả:** Cùng câu, KHÁC nhãn → Conflict!

**Tại sao có conflict?**
- Người 1 nghĩ: "Tôi rất vui!" → chỉ có vui vẻ
- Người 2 nghĩ: "Tôi rất vui!" → vui vẻ + buồn bã (???)
- Có thể do: hiểu sai, nhầm lẫn, hoặc quan điểm khác nhau

---

## Giải pháp của merge_data.py

### Tính năng mới: Phát hiện và xử lý conflict

Script `merge_data.py` giờ có thể:

1. ✅ **Phát hiện conflict** tự động
2. ✅ **Báo cáo conflict** chi tiết
3. ✅ **Cho phép chọn cách xử lý** conflict

---

## Các chiến lược xử lý conflict

### 1. `report` (Mặc định) - Báo cáo và giữ đầu tiên

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

**Kết quả:**
```
⚠ WARNING: Found 2 conflicts (same text, different labels)!
======================================================================

Conflict 1: "Tôi rất vui!"
  Found 2 times with different labels:
    Version 1: vui vẻ
    Version 2: vui vẻ, buồn bã

======================================================================
Conflict strategy: report
⚠ Keeping first occurrence of each conflict (default)
  To change: use --conflict-strategy [first|last|merge|skip]
======================================================================
```

**Hành động:** Giữ version 1, xóa version 2

---

### 2. `first` - Giữ đầu tiên

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy first
```

**Kết quả:** Giống `report`, nhưng không báo cáo chi tiết

---

### 3. `last` - Giữ cuối cùng

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy last
```

**Kết quả:** Giữ version cuối cùng, xóa các version trước

---

### 4. `merge` - Merge nhãn (OR logic) ✅ KHUYẾN NGHỊ

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy merge
```

**Logic:**
- Nếu **BẤT KỲ** version nào có nhãn = 1 → Kết quả = 1
- Chỉ khi **TẤT CẢ** version có nhãn = 0 → Kết quả = 0

**Ví dụ:**
```
Version 1: joy=1, sadness=0
Version 2: joy=1, sadness=1
─────────────────────────────
Kết quả:   joy=1, sadness=1  ← Merge (OR logic)
```

**Ưu điểm:**
- Không mất thông tin
- Bao gồm tất cả quan điểm
- Tốt cho multi-label classification

**Nhược điểm:**
- Có thể thêm nhãn sai nếu 1 người đánh nhầm

---

### 5. `skip` - Bỏ qua tất cả conflict

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy skip
```

**Kết quả:** Xóa TẤT CẢ các version của câu conflict

**Ưu điểm:**
- Đảm bảo data sạch (không có conflict)
- Tránh nhãn sai

**Nhược điểm:**
- Mất data (có thể mất nhiều câu)

---

## So sánh các chiến lược

| Chiến lược | Khi nào dùng | Ưu điểm | Nhược điểm |
|------------|--------------|---------|------------|
| `report` (mặc định) | Muốn biết có conflict | Báo cáo chi tiết | Mất thông tin (giữ 1) |
| `first` | Tin người đầu tiên | Đơn giản | Mất thông tin |
| `last` | Tin người cuối cùng | Đơn giản | Mất thông tin |
| `merge` ✅ | Muốn giữ tất cả thông tin | Không mất data | Có thể thêm nhãn sai |
| `skip` | Muốn data sạch 100% | Data chắc chắn đúng | Mất nhiều data |

---

## Ví dụ thực tế

### Tình huống: 3 người đánh nhãn cùng 1 câu

```csv
# member_nam.csv
"Sản phẩm này tốt!",1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0  ← joy, trust

# member_linh.csv
"Sản phẩm này tốt!",1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0  ← joy, trust, surprise

# member_hoa.csv
"Sản phẩm này tốt!",1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0  ← joy, trust, love
```

### Kết quả với từng chiến lược:

**`report` / `first`:**
```
Kết quả: joy=1, trust=1  ← Chỉ giữ version 1
```

**`last`:**
```
Kết quả: joy=1, trust=1, love=1  ← Chỉ giữ version 3
```

**`merge` ✅:**
```
Kết quả: joy=1, trust=1, surprise=1, love=1  ← Merge tất cả
```

**`skip`:**
```
Kết quả: (Xóa câu này)
```

---

## Khuyến nghị

### Cho team nhỏ (3-5 người):

✅ **Dùng `merge`**
```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy merge
```

**Lý do:**
- Giữ được tất cả thông tin
- Nhiều quan điểm → Model học tốt hơn
- Ít conflict (team nhỏ, dễ thống nhất)

---

### Cho team lớn (10+ người):

1. **Bước 1:** Dùng `report` để xem conflict
```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

2. **Bước 2:** Review conflict, quyết định:
   - Nếu conflict ít (< 5%) → Dùng `merge`
   - Nếu conflict nhiều (> 10%) → Dùng `skip` hoặc review thủ công

---

## Cách review conflict thủ công

### Bước 1: Chạy với `report`

```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

### Bước 2: Xem conflict

```
⚠ WARNING: Found 5 conflicts!

Conflict 1: "Tôi rất vui!"
  Version 1: vui vẻ
  Version 2: vui vẻ, buồn bã

Conflict 2: "Sản phẩm tốt!"
  Version 1: vui vẻ, tin tưởng
  Version 2: vui vẻ, tin tưởng, ngạc nhiên
```

### Bước 3: Quyết định

- **Conflict 1:** Sai rõ ràng (vui + buồn?) → Dùng `first` hoặc `skip`
- **Conflict 2:** Hợp lý (có thể ngạc nhiên) → Dùng `merge`

### Bước 4: Thảo luận với team

- Họp team, review conflict
- Thống nhất cách đánh nhãn
- Sửa lại file gốc nếu cần

---

## Tips tránh conflict

### 1. Thống nhất quy tắc đánh nhãn

**Tạo file `QUYY_TAC_DANH_NHAN.md`:**
```markdown
# Quy tắc đánh nhãn

## Câu "Sản phẩm này tốt!"
- ✅ Đánh: joy, trust
- ❌ Không đánh: surprise (trừ khi có "rất ngạc nhiên")

## Câu "Tôi rất vui!"
- ✅ Đánh: joy, excited
- ❌ Không đánh: sadness (mâu thuẫn)
```

### 2. Review chéo

- Người A tạo data → Người B review
- Phát hiện sai sớm

### 3. Dùng tool validation

```bash
# Validate trước khi merge
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv --conflict-strategy report
```

---

## Tóm tắt

| Vấn đề | Giải pháp | Command |
|--------|-----------|---------|
| Dữ liệu trùng hoàn toàn | Tự động xóa | `python merge_data.py` (mặc định) |
| Conflict (câu trùng, nhãn khác) | Phát hiện và báo cáo | `--conflict-strategy report` |
| Muốn giữ tất cả thông tin | Merge nhãn | `--conflict-strategy merge` |
| Muốn data sạch 100% | Bỏ qua conflict | `--conflict-strategy skip` |

**Khuyến nghị:** Dùng `merge` cho team nhỏ, `report` + review thủ công cho team lớn.
