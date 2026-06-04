"""
Remove specific file from data tracker
Allows you to retrain specific files without resetting everything
"""

import json
import sys

def remove_file_from_tracker(filename):
    """Remove a specific file from data tracker."""
    tracker_file = 'model_registry/data_tracker.json'
    
    try:
        with open(tracker_file, 'r') as f:
            tracker = json.load(f)
        
        if filename in tracker['files']:
            samples = tracker['files'][filename]['samples']
            del tracker['files'][filename]
            
            # Update total
            tracker['total_trained_samples'] = sum(
                f['samples'] for f in tracker['files'].values()
            )
            
            with open(tracker_file, 'w') as f:
                json.dump(tracker, f, indent=2)
            
            print(f"✅ Đã xóa {filename} khỏi tracker")
            print(f"   • Số mẫu đã xóa: {samples}")
            print(f"   • Bạn có thể train lại file này ngay bây giờ!")
        else:
            print(f"⚠️ File {filename} không có trong tracker")
            print(f"\nCác file hiện có:")
            for f in tracker['files'].keys():
                print(f"   • {f}")
    
    except FileNotFoundError:
        print("❌ Không tìm thấy data_tracker.json")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python remove_file_from_tracker.py <filename>")
        print("\nVí dụ:")
        print("  python remove_file_from_tracker.py member_an.csv")
        print("  python remove_file_from_tracker.py sample_comments.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    remove_file_from_tracker(filename)
