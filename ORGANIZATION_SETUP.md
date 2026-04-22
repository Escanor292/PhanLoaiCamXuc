# 🏢 Organization Setup - Hướng Dẫn Tạo Team Repository

## 🎯 Mục Tiêu
Tạo organization repository để team cùng quản lý model: `emotion-classification-vn/emotion-classification`

## 📋 Các Bước Setup

### Bước 1: Tạo Organization (Bạn làm thủ công)
1. **Vào:** https://huggingface.co/organizations/new
2. **Organization name:** `emotion-classification-vn`
3. **Display name:** `Emotion Classification Vietnam`
4. **Description:** `Vietnamese Emotion Classification Team`
5. **Chọn:** Public organization
6. **Create organization**

### Bước 2: Tạo Model Repository
1. **Vào:** https://huggingface.co/new
2. **Owner:** Chọn `emotion-classification-vn` (organization)
3. **Repository name:** `emotion-classification`
4. **Repository type:** Model
5. **Visibility:** Public
6. **Create repository**

### Bước 3: Thêm Team Members
1. **Vào organization:** https://huggingface.co/emotion-classification-vn
2. **Settings** → **Members**
3. **Invite members** với email hoặc username
4. **Role:** Member hoặc Admin

### Bước 4: Cấp Quyền Token
Mỗi thành viên cần:
1. **Token với quyền write** cho organization repos
2. **Fine-grained token** → Select organization → Write access

## 🔄 Chuyển Sang Organization Repo

Sau khi tạo xong organization repo, chạy:

```bash
python switch_repo.py
# Chọn option 2: emotion-classification-vn/emotion-classification
```

Hoặc thủ công:

```bash
python model_sharing.py config --method huggingface --repo "emotion-classification-vn/emotion-classification"
python model_sharing.py sync
```

## 🚀 Workflow Sau Khi Setup

### Admin (Bạn):
```bash
# Upload model lên organization repo
python model_sharing.py sync
git push  # Share metadata
```

### Team Members:
```bash
# Login với token có quyền organization
hf auth login  # hoặc python setup_hf_login.py

# Training tự động download từ organization repo
python train_simple.py
```

## 📊 Lợi Ích Organization Repo

| Personal Repo | Organization Repo |
|---------------|-------------------|
| Chỉ 1 người quản lý | Team cùng quản lý |
| Khó chia sẻ quyền | Dễ chia sẻ quyền |
| Không professional | Professional hơn |
| Khó mở rộng team | Dễ mở rộng team |

## 🔧 Backup Plan

Nếu chưa setup được organization, team vẫn có thể dùng:

```bash
python switch_repo.py
# Chọn option 1: Escanor292/emotion-classification (đã hoạt động)
```

## ✅ Checklist

- [ ] Tạo organization `emotion-classification-vn`
- [ ] Tạo repo `emotion-classification-vn/emotion-classification`
- [ ] Thêm team members vào organization
- [ ] Mỗi member có token với quyền write organization
- [ ] Test upload: `python model_sharing.py sync`
- [ ] Test download: `python train_simple.py`

## 🆘 Troubleshooting

### Lỗi: "403 Forbidden"
- Token không có quyền write organization
- Tạo token mới với quyền organization

### Lỗi: "Repository not found"
- Repository chưa được tạo
- Kiểm tra tên organization và repo

### Lỗi: "Not a member"
- Chưa được thêm vào organization
- Admin cần invite member

---

**Sau khi setup xong, team sẽ có repository chuyên nghiệp để chia sẻ model!** 🎯