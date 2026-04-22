#!/usr/bin/env python3
"""
Script kiểm tra trước khi push để đảm bảo không xóa file
"""

import subprocess
import sys

def check_deleted_files():
    """Kiểm tra xem có file nào bị xóa không"""
    try:
        # Lấy danh sách file bị xóa so với origin/main
        result = subprocess.run([
            'git', 'diff', '--name-only', '--diff-filter=D', 'origin/main..HEAD'
        ], capture_output=True, text=True, check=True)
        
        deleted_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if deleted_files and deleted_files != ['']:
            print("❌ CẢNH BÁO: Phát hiện file bị xóa!")
            print("Các file bị xóa:")
            for file in deleted_files:
                print(f"  - {file}")
            print("\n🚫 QUY TẮC: Chỉ được phép thêm hoặc sửa file, không được xóa!")
            print("💡 Hướng dẫn:")
            print("  1. Khôi phục file bị xóa: git checkout HEAD~1 -- <tên_file>")
            print("  2. Hoặc hủy commit xóa file: git reset --soft HEAD~1")
            return False
        
        print("✅ Kiểm tra OK: Không có file nào bị xóa")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi kiểm tra: {e}")
        return False

def main():
    print("🔍 Kiểm tra trước khi push...")
    
    if not check_deleted_files():
        print("\n❌ Push bị từ chối!")
        sys.exit(1)
    
    print("✅ Có thể push an toàn!")

if __name__ == "__main__":
    main()