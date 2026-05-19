import os
import sys

# Thêm thư mục gốc của dự án vào sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(project_root), "config.py")):
    # Nếu file này nằm trong thư mục con (ví dụ Pro_Edition)
    project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import threading
import queue
import time
import glob
import shutil
import json
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

# Import local modules
from config import Config
from gui_logger import setup_gui_logging
from model_registry import ModelRegistry
from huggingface_hub import login, whoami

# Matplotlib for charts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")



# --- CẤU HÌNH GIAO DIỆN PREMIUM ---
# Màu sắc chủ đạo (Theme slate/blue hiện đại)
BG_COLOR = "#0f172a"          # Nền chính
SIDEBAR_COLOR = "#1e293b"     # Nền sidebar
CARD_COLOR = "#1e293b"        # Nền card
ACCENT_COLOR = "#3b82f6"      # Màu nhấn (Xanh dương)
ACCENT_HOVER = "#2563eb"      # Xanh dương đậm khi hover
SUCCESS_COLOR = "#10b981"     # Xanh lá
WARNING_COLOR = "#f59e0b"     # Vàng/Cam
TEXT_PRIMARY = "#f8fafc"      # Trắng sáng
TEXT_SECONDARY = "#94a3b8"    # Xám nhạt

FONT_FAMILY = "Segoe UI"

ctk.set_appearance_mode("dark")

class PremiumEmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Emotion Classifier - Pro Edition")
        self.geometry("1200x750")
        self.configure(fg_color=BG_COLOR)
        
        # Cấu hình grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Biến trạng thái
        self.log_queue = queue.Queue()
        self.is_training = False
        self.hf_user = None
        self.gui_config_file = "gui_config.json"
        self.editor_path = self.load_gui_config().get("editor_path", "")
        self.registry = ModelRegistry()  # Thêm registry để quản lý model
        
        # Setup logging
        setup_gui_logging(self.log_queue)

        # Model cache for testing
        self.current_model = None
        self.current_tokenizer = None
        self.loaded_model_id = None


        # ==========================================
        # SIDEBAR
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=SIDEBAR_COLOR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1) # Đẩy phần dưới xuống

        # Logo / Branding
        self.logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.logo_frame.grid(row=0, column=0, padx=20, pady=(30, 30))
        
        self.logo_icon = ctk.CTkLabel(self.logo_frame, text="🧠", font=ctk.CTkFont(family=FONT_FAMILY, size=40))
        self.logo_icon.pack()
        
        self.logo_text = ctk.CTkLabel(self.logo_frame, text="AI EMOTION", font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"), text_color=ACCENT_COLOR)
        self.logo_text.pack(pady=(5, 0))
        self.logo_sub = ctk.CTkLabel(self.logo_frame, text="PRO EDITION", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=TEXT_SECONDARY)
        self.logo_sub.pack()

        # Menu Buttons (Style hiện đại)
        self.nav_buttons = {}
        self.create_nav_button("login", "🔑 Tài Khoản HF", 1, self.show_login)
        self.create_nav_button("training", "🚀 Huấn Luyện AI", 2, self.show_training)
        self.create_nav_button("test", "🔍 Thử Nghiệm", 3, self.show_test)
        self.create_nav_button("data", "📊 Quản Lý Dữ Liệu", 4, self.show_data)


        # Nút Doctor (Nổi bật)
        self.btn_doctor = ctk.CTkButton(
            self.sidebar, text="🛠️ Sửa Lỗi Môi Trường", 
            command=self.run_doctor_async, 
            fg_color="#b91c1c", hover_color="#991b1b",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            height=45, corner_radius=8
        )
        self.btn_doctor.grid(row=5, column=0, padx=20, pady=20)

        # Status Footer
        self.status_label = ctk.CTkLabel(self.sidebar, text="🔴 Offline", font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TEXT_SECONDARY)
        self.status_label.grid(row=7, column=0, padx=20, pady=(10, 20), sticky="w")

        # ==========================================
        # MAIN CONTENT AREA
        # ==========================================
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color=BG_COLOR)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1) # View chiếm chỗ
        self.main_container.grid_rowconfigure(2, weight=0) # Log area cố định

        # Header Title
        self.header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        self.tab_title = ctk.CTkLabel(self.header_frame, text="Tổng Quan", font=ctk.CTkFont(family=FONT_FAMILY, size=28, weight="bold"), text_color=TEXT_PRIMARY)
        self.tab_title.pack(side="left")
        
        self.tab_subtitle = ctk.CTkLabel(self.header_frame, text="Bảng điều khiển hệ thống", font=ctk.CTkFont(family=FONT_FAMILY, size=14), text_color=TEXT_SECONDARY)
        self.tab_subtitle.pack(side="left", padx=15, pady=(8,0))

        # View Container (Chứa các màn hình khác nhau)
        self.view_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.view_container.grid(row=1, column=0, sticky="nsew")
        self.view_container.grid_columnconfigure(0, weight=1)
        self.view_container.grid_rowconfigure(0, weight=1)

        # Khởi tạo các View
        self.init_training_view()
        self.init_test_view()
        self.init_login_view()
        self.init_data_view()


        # ==========================================
        # LOG TERMINAL AREA
        # ==========================================
        self.log_frame = ctk.CTkFrame(self.main_container, height=180, fg_color="#000000", corner_radius=10, border_width=1, border_color="#334155")
        self.log_frame.grid(row=2, column=0, sticky="ew", pady=(20, 0))
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(1, weight=1)
        
        self.log_header = ctk.CTkFrame(self.log_frame, height=30, fg_color="#1e293b", corner_radius=10)
        self.log_header.grid(row=0, column=0, sticky="ew")
        
        # Terminal dots
        ctk.CTkLabel(self.log_header, text="🔴 🟡 🟢", font=("Arial", 10)).pack(side="left", padx=10)
        ctk.CTkLabel(self.log_header, text="Terminal Output", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"), text_color=TEXT_SECONDARY).pack(side="left", padx=5)
        
        self.log_text = ctk.CTkTextbox(self.log_frame, height=150, font=("Consolas", 13), fg_color="#000000", text_color="#10b981")
        self.log_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Khởi động tính năng
        self.after(100, self.poll_log_queue)
        self.show_login()
        self.check_hf_status()

    # ==========================================
    # UI COMPONENTS BUILDERS
    # ==========================================
    def create_nav_button(self, key, text, row, command):
        btn = ctk.CTkButton(
            self.sidebar, text=text, command=lambda k=key, c=command: self.handle_nav_click(k, c),
            fg_color="transparent", hover_color=CARD_COLOR, text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
            anchor="w", height=45, corner_radius=8
        )
        btn.grid(row=row, column=0, padx=15, pady=5, sticky="ew")
        self.nav_buttons[key] = btn

    def handle_nav_click(self, key, command):
        # Reset colors
        for k, btn in self.nav_buttons.items():
            btn.configure(fg_color="transparent", text_color=TEXT_SECONDARY)
        # Highlight active
        self.nav_buttons[key].configure(fg_color=ACCENT_COLOR, text_color=TEXT_PRIMARY, hover_color=ACCENT_HOVER)
        command()

    # ==========================================
    # VIEWS INITIALIZATION
    # ==========================================
    def init_training_view(self):
        self.training_view = ctk.CTkScrollableFrame(self.view_container, fg_color="transparent")
        
        # Dashboard Grid
        self.training_view.grid_columnconfigure(0, weight=1)
        self.training_view.grid_columnconfigure(1, weight=1)

        
        # CARD 1: Thông tin mô hình
        self.model_card = ctk.CTkFrame(self.training_view, fg_color=CARD_COLOR, corner_radius=15)
        self.model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 20))
        
        ctk.CTkLabel(self.model_card, text="🏆 Mô Hình Tốt Nhất", font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))
        self.best_model_info = ctk.CTkLabel(self.model_card, text="Đang tải...", justify="left", font=ctk.CTkFont(family=FONT_FAMILY, size=14))
        self.best_model_info.pack(anchor="w", padx=20, pady=(0, 20))
        
        # CARD 2: Cấu hình hiện tại
        self.config_card = ctk.CTkFrame(self.training_view, fg_color=CARD_COLOR, corner_radius=15)
        self.config_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 20))
        
        ctk.CTkLabel(self.config_card, text="⚙️ Cấu Hình Hybrid PhoBERT", font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))
        config_text = f"• Base: {Config.MODEL_NAME}\n• Epochs tối đa: {Config.NUM_EPOCHS}\n• Learning Rate: {Config.LEARNING_RATE}\n• Batch Size: {Config.BATCH_SIZE}\n• Tự động Transfer Learning: Bật"
        ctk.CTkLabel(self.config_card, text=config_text, justify="left", font=ctk.CTkFont(family=FONT_FAMILY, size=14)).pack(anchor="w", padx=20, pady=(0, 20))

        # CARD 3: Chọn dữ liệu (NEW)
        self.data_select_card = ctk.CTkFrame(self.training_view, fg_color=CARD_COLOR, corner_radius=15)
        self.data_select_card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 20))
        
        data_header = ctk.CTkFrame(self.data_select_card, fg_color="transparent")
        data_header.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(data_header, text="📂 Chọn dữ liệu huấn luyện", font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(data_header, text="🔄 Làm mới danh sách", width=140, height=28, command=self.refresh_training_data_list, fg_color="#334155").pack(side="right")
        
        self.train_data_scroll = ctk.CTkScrollableFrame(self.data_select_card, height=150, fg_color="#1e293b", corner_radius=10)
        self.train_data_scroll.pack(fill="x", padx=20, pady=(0, 15))
        self.train_data_checkboxes = []

        # ACTION AREA (Dưới cùng)
        self.action_frame = ctk.CTkFrame(self.training_view, fg_color="transparent")
        self.action_frame.grid(row=2, column=0, columnspan=2, pady=(5, 20))

        # Container cho các nút
        self.buttons_container = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        self.buttons_container.pack()
        
        # Nút huấn luyện to khổng lồ
        self.btn_start_train = ctk.CTkButton(
            self.buttons_container, text="🚀 KHỞI ĐỘNG HUẤN LUYỆN", 
            command=self.start_training_thread,
            fg_color=SUCCESS_COLOR, hover_color="#059669", text_color="#ffffff",
            font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
            height=70, width=400, corner_radius=35
        )
        self.btn_start_train.pack(pady=(0, 10))
        
        # Nút đẩy model lên cloud (nhỏ hơn, nằm dưới)
        self.btn_upload_model = ctk.CTkButton(
            self.buttons_container, text="☁️ Đẩy Model Tốt Nhất Lên Cloud", 
            command=self.upload_best_model_to_cloud,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER, text_color="#ffffff",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            height=45, width=400, corner_radius=10
        )
        self.btn_upload_model.pack()
        
        self.progress_bar = ctk.CTkProgressBar(self.action_frame, width=400, height=10, progress_color=ACCENT_COLOR)
        self.progress_bar.pack(pady=20)
        self.progress_bar.set(0)

    def init_data_view(self):
        self.data_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        
        # Header controls
        self.data_header = ctk.CTkFrame(self.data_view, fg_color="transparent")
        self.data_header.pack(fill="x", pady=(0, 15))
        
        btn_style = {"font": ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"), "height": 35, "corner_radius": 6}
        
        ctk.CTkButton(self.data_header, text="➕ Thêm CSV", command=self.add_csv_file, fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER, **btn_style).pack(side="left")
        ctk.CTkButton(self.data_header, text="🔄 Làm mới", command=self.refresh_csv_list, fg_color=CARD_COLOR, hover_color="#334155", **btn_style).pack(side="left", padx=10)
        ctk.CTkButton(self.data_header, text="☁️ Đẩy lên Git", command=self.push_data_to_git, fg_color=SUCCESS_COLOR, hover_color="#059669", **btn_style).pack(side="right")
        
        # List Container
        self.scroll_frame = ctk.CTkScrollableFrame(self.data_view, fg_color=CARD_COLOR, corner_radius=15, scrollbar_button_color="#334155")
        self.scroll_frame.pack(fill="both", expand=True)
        
        # Header row cho bảng
        header_row = ctk.CTkFrame(self.scroll_frame, fg_color="transparent", height=30)
        header_row.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header_row, text="Sử dụng", font=ctk.CTkFont(weight="bold"), width=60).pack(side="left")
        ctk.CTkLabel(header_row, text="Tên File", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header_row, text="Hành động", font=ctk.CTkFont(weight="bold")).pack(side="right", padx=10)
        
        # Dòng kẻ ngang
        ctk.CTkFrame(self.scroll_frame, height=2, fg_color="#334155").pack(fill="x", padx=10, pady=5)
        
        self.csv_checkboxes = []

    def init_login_view(self):
        self.login_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        
        # Thẻ đăng nhập trung tâm
        self.login_card = ctk.CTkFrame(self.login_view, fg_color=CARD_COLOR, corner_radius=20, width=500, height=450)
        self.login_card.place(relx=0.5, rely=0.5, anchor="center")
        self.login_card.pack_propagate(False)

        ctk.CTkLabel(self.login_card, text="🤗 Hugging Face Connect", font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"), text_color=ACCENT_COLOR).pack(pady=(40, 10))
        
        self.hf_status_label = ctk.CTkLabel(self.login_card, text="Đang kiểm tra trạng thái...", font=ctk.CTkFont(family=FONT_FAMILY, size=14))
        self.hf_status_label.pack(pady=(0, 20))

        ctk.CTkLabel(self.login_card, text="Access Token:", font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=50, pady=(10, 5))
        
        self.token_entry = ctk.CTkEntry(self.login_card, width=400, height=45, placeholder_text="hf_...", show="•", corner_radius=10, border_color="#334155")
        self.token_entry.pack(pady=(0, 20))

        self.btn_do_login = ctk.CTkButton(
            self.login_card, text="Kết Nối / Đăng Nhập", 
            command=self.do_hf_login,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            height=50, width=400, corner_radius=12
        )
        self.btn_do_login.pack(pady=10)

        guide = "Hướng dẫn:\n1. Vào huggingface.co/settings/tokens\n2. Tạo token mới\n3. Cấp quyền WRITE (để đẩy model)"
        ctk.CTkLabel(self.login_card, text=guide, justify="left", font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TEXT_SECONDARY).pack(pady=(20, 0))


    # ==========================================
    # NAVIGATION METHODS
    # ==========================================
    def hide_all_views(self):
        self.training_view.grid_forget()
        self.test_view.grid_forget()
        self.data_view.grid_forget()
        self.login_view.grid_forget()

    def show_training(self):
        self.tab_title.configure(text="Phòng Huấn Luyện")
        self.tab_subtitle.configure(text="Khởi động và theo dõi AI học tập")
        self.hide_all_views()
        self.training_view.grid(row=0, column=0, sticky="nsew")
        self.update_best_model_info()
        self.refresh_training_data_list()

    def show_test(self):
        self.tab_title.configure(text="Thử Nghiệm Mô Hình")
        self.tab_subtitle.configure(text="Dự đoán cảm xúc thời gian thực với biểu đồ")
        self.hide_all_views()
        self.test_view.grid(row=0, column=0, sticky="nsew")
        self.update_test_model_info()


    def show_data(self):
        self.tab_title.configure(text="Kho Dữ Liệu")
        self.tab_subtitle.configure(text="Quản lý dữ liệu train/test")
        self.hide_all_views()
        self.data_view.grid(row=0, column=0, sticky="nsew")
        self.refresh_csv_list()

    def show_login(self):
        self.tab_title.configure(text="Đám Mây (Cloud)")
        self.tab_subtitle.configure(text="Kết nối hệ sinh thái Hugging Face")
        self.hide_all_views()
        self.login_view.grid(row=0, column=0, sticky="nsew")

    # ==========================================
    # CORE LOGIC
    # ==========================================
    def poll_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.insert("end", msg)
            self.log_text.see("end")
        self.after(100, self.poll_log_queue)

    def load_gui_config(self):
        if os.path.exists(self.gui_config_file):
            try:
                with open(self.gui_config_file, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def save_gui_config(self):
        with open(self.gui_config_file, 'w') as f:
            json.dump({"editor_path": self.editor_path}, f)

    def update_best_model_info(self):
        try:
            registry = ModelRegistry()
            best = registry.get_best_model()
            if best:
                f1_score = best['metrics']['macro_f1'] * 100
                reg_date = best['registered_at'].split('T')[0]
                y, m, d = reg_date.split('-')
                formatted_date = f"{d}/{m}/{y}"
                
                info = f"🆔 ID: {best['model_id']}\n" \
                       f"🎯 Độ chính xác (F1): {f1_score:.2f}%\n" \
                       f"📉 Mức độ lỗi (Loss): {best['metrics']['test_loss']:.4f}\n" \
                       f"👤 Người tạo: {best['metadata'].get('person', 'N/A')}\n" \
                       f"📅 Ngày cập nhật: {formatted_date}"
                self.best_model_info.configure(text=info, text_color=SUCCESS_COLOR)

            else:
                self.best_model_info.configure(text="Chưa có mô hình nào.\nHãy huấn luyện lần đầu!", text_color=WARNING_COLOR)
        except Exception:
            self.best_model_info.configure(text="Lỗi đọc registry.", text_color="#ef4444")

    # --- DATA LOGIC ---
    def refresh_csv_list(self):
        # Giữ lại header row và separator
        children = self.scroll_frame.winfo_children()
        for child in children[2:]: # Bỏ qua 2 row đầu (header và dòng kẻ)
            child.destroy()
            
        self.csv_checkboxes = []
        csv_files = glob.glob(os.path.join(Config.DATA_DIR, "*.csv"))
        csv_files = [f for f in csv_files if "TEMPLATE" not in f.upper() and "merged_temp" not in f.lower()]
        
        if not csv_files:
            ctk.CTkLabel(self.scroll_frame, text="Thư mục trống. Hãy thêm file CSV.", text_color=TEXT_SECONDARY).pack(pady=20)
            return

        for i, f in enumerate(sorted(csv_files)):
            name = os.path.basename(f)
            bg_color = CARD_COLOR if i % 2 == 0 else "#253347" # Alternating row color
            
            row = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, corner_radius=6, height=40)
            row.pack(fill="x", padx=10, pady=2)
            row.pack_propagate(False) # Cố định chiều cao
            
            # Checkbox
            cb = ctk.CTkCheckBox(row, text="", width=20, border_width=2, fg_color=ACCENT_COLOR)
            cb.pack(side="left", padx=(10, 0), pady=10)
            cb.select()
            cb.file_path = f
            self.csv_checkboxes.append(cb)
            
            # Tên file
            ctk.CTkLabel(row, text=name, font=ctk.CTkFont(size=14)).pack(side="left", padx=10)
            
            # Nút sửa
            btn_edit = ctk.CTkButton(
                row, text="✏️ Sửa", width=70, height=28, 
                fg_color="#475569", hover_color="#334155", 
                command=lambda p=f: self.edit_csv_file(p)
            )
            btn_edit.pack(side="right", padx=10)

    def add_csv_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            filename = os.path.basename(file_path)
            dest = os.path.join(Config.DATA_DIR, filename)
            if os.path.exists(dest):
                if not messagebox.askyesno("Trùng tên", f"File {filename} đã tồn tại. Ghi đè?"): return
            shutil.copy(file_path, dest)
            print(f"✅ Đã thêm file mới: {filename}")
            self.refresh_csv_list()

    def edit_csv_file(self, file_path):
        try:
            if self.editor_path and os.path.exists(self.editor_path):
                import subprocess
                subprocess.Popen([self.editor_path, file_path])
                print(f"📝 Đang mở bằng {os.path.basename(self.editor_path)}...")
            else:
                if messagebox.askyesno("Chọn App", "Bạn chưa cài đặt phần mềm mặc định để mở CSV.\nBạn có muốn chọn 1 phần mềm (vd: Excel, Notepad) ngay bây giờ không?"):
                    path = filedialog.askopenfilename(filetypes=[("Executable", "*.exe")])
                    if path:
                        self.editor_path = path
                        self.save_gui_config()
                        import subprocess
                        subprocess.Popen([self.editor_path, file_path])
                        return
                
                # Mở mặc định của Windows
                if os.name == 'nt': os.startfile(file_path)
                else: import subprocess; subprocess.call(['open', file_path])
                print(f"📝 Đang mở file bằng trình mặc định hệ thống.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở file: {e}")

    def push_data_to_git(self):
        if not messagebox.askyesno("Đẩy lên Git", "Đẩy tất cả dữ liệu CSV lên kho lưu trữ chung (Git)?"): return
        def run():
            try:
                print("\n" + "="*40 + "\n📤 ĐANG ĐẨY DỮ LIỆU LÊN GIT...\n" + "="*40)
                import subprocess
                if not os.path.exists(".git"):
                    print("❌ Thư mục chưa được khởi tạo Git!")
                    return

                for cmd in [["git", "add", "data/*.csv"], ["git", "commit", "-m", f"Cập nhật data - {datetime.now().strftime('%H:%M %d/%m')}"], ["git", "push"]]:
                    print(f"> {' '.join(cmd)}")
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.stdout: print(res.stdout.strip())
                    if res.stderr: print(res.stderr.strip())
                    
                print("✅ Hoàn tất đồng bộ Git!")
                self.after(0, lambda: messagebox.showinfo("Thành công", "Dữ liệu đã nằm an toàn trên Git!"))
            except Exception as e:
                print(f"❌ Lỗi Git: {e}")
        threading.Thread(target=run, daemon=True).start()

    # --- HF LOGIC ---
    def check_hf_status(self):
        try:
            user = whoami()
            self.hf_user = user['name']
            self.hf_status_label.configure(text=f"🟢 Đã kết nối: {self.hf_user}", text_color=SUCCESS_COLOR)
            self.status_label.configure(text=f"🟢 Online ({self.hf_user})", text_color=SUCCESS_COLOR)
            return True
        except:
            self.hf_status_label.configure(text="🔴 Chưa kết nối", text_color=WARNING_COLOR)
            self.status_label.configure(text="🔴 Offline", text_color=WARNING_COLOR)
            return False

    def do_hf_login(self):
        token = self.token_entry.get().strip()
        if not token.startswith("hf_"):
            messagebox.showerror("Lỗi", "Token phải bắt đầu bằng 'hf_'")
            return
        
        self.btn_do_login.configure(state="disabled", text="Đang kết nối...")
        def run():
            try:
                login(token=token)
                self.after(0, self.check_hf_status)
                print("✅ Kết nối Hugging Face thành công!")
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                self.after(0, lambda: messagebox.showerror("Lỗi", f"Kết nối thất bại: {e}"))
            finally:
                self.after(0, lambda: self.btn_do_login.configure(state="normal", text="Kết Nối / Đăng Nhập"))
        threading.Thread(target=run, daemon=True).start()

    # --- TRAINING LOGIC ---
    def run_doctor_async(self):
        from windows_doctor import run_doctor
        print("\n" + "🛠️ "*15 + "\nĐANG CHẠY CÔNG CỤ CHẨN ĐOÁN...\n" + "🛠️ "*15)
        threading.Thread(target=run_doctor, daemon=True).start()

    def start_training_thread(self):
        if self.is_training: return
        
        selected = [cb.file_path for cb in self.train_data_checkboxes if cb.get()]
        if not selected:
            messagebox.showwarning("Thiếu dữ liệu", "Phải chọn ít nhất 1 file CSV để AI có thể học!")
            return


        if not self.check_hf_status() and not messagebox.askyesno("Chưa kết nối Cloud", "Bạn chưa đăng nhập HF. Kết quả sẽ KHÔNG được lưu lên mây. Vẫn tiếp tục chạy?"):
            self.show_login()
            return

        self.is_training = True
        self.btn_start_train.configure(state="disabled", text="⚡ ĐANG HUẤN LUYỆN...", fg_color=WARNING_COLOR)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        threading.Thread(target=self.run_training_process, args=(selected,), daemon=True).start()

    def run_training_process(self, csv_files):
        try:
            from train_simple import merge_all_data, Config
            from train_with_args import main as train_main
            from transfer_learning import load_base_model_for_transfer, should_use_transfer_learning, get_transfer_learning_settings
            from data_tracker import DataTracker
            from model_sharing import ModelSharing
            import pandas as pd

            print("\n" + "🌟 "*15 + "\nBẮT ĐẦU QUY TRÌNH HUẤN LUYỆN TỰ ĐỘNG\n" + "🌟 "*15)

            merged_file, new_samples, stats = merge_all_data(csv_files)
            if not merged_file or new_samples == 0:
                print("⏭️ Dữ liệu này AI đã học rồi, không cần học lại!")
                self.finish_training(False)
                return

            use_transfer = should_use_transfer_learning()
            epochs, lr = 5, Config.LEARNING_RATE
            base_info = None

            if use_transfer:
                device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
                _, _, base_info = load_base_model_for_transfer('hybrid', device)
                settings = get_transfer_learning_settings(base_info)
                epochs, lr = settings['epochs'], settings['learning_rate']

            person = self.hf_user if self.hf_user else os.getenv('USERNAME', 'team_member')
            
            old_argv = sys.argv
            sys.argv = ['train.py', '--model-type', 'hybrid', '--data', merged_file, '--epochs', str(epochs), '--lr', str(lr), '--experiment-name', f'PRO Training by {person}', '--register-model']
            if base_info: sys.argv.extend(['--transfer-from', base_info['model_id']])
            
            train_main()
            sys.argv = old_argv

            tracker = DataTracker()
            tracker.mark_as_trained(csv_files, pd.read_csv(merged_file))

            if self.hf_user:
                print("\n☁️ ĐANG ĐỒNG BỘ LÊN ĐÁM MÂY...")
                if ModelSharing().sync_best_model(): print("✅ Đồng bộ thành công!")
                else: print("⚠️ Đồng bộ thất bại.")

            self.finish_training(True)

        except Exception as e:
            print(f"❌ LỖI NGHIÊM TRỌNG: {e}")
            import traceback; traceback.print_exc()
            self.finish_training(False)

    def refresh_training_data_list(self):
        # Clear existing
        for cb in self.train_data_checkboxes: cb.destroy()
        self.train_data_checkboxes = []
        
        csv_files = glob.glob(os.path.join(Config.DATA_DIR, "*.csv"))
        csv_files = [f for f in csv_files if "TEMPLATE" not in f.upper() and "merged_temp" not in f.lower()]
        
        for f in sorted(csv_files):
            name = os.path.basename(f)
            cb = ctk.CTkCheckBox(self.train_data_scroll, text=name, font=ctk.CTkFont(size=13))
            cb.pack(anchor="w", padx=20, pady=5)
            cb.select()
            cb.file_path = f
            self.train_data_checkboxes.append(cb)

    def init_test_view(self):
        self.test_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.test_view.grid_columnconfigure(0, weight=1)
        
        # Top Card: Input
        self.test_input_card = ctk.CTkFrame(self.test_view, fg_color=CARD_COLOR, corner_radius=15)
        self.test_input_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        ctk.CTkLabel(self.test_input_card, text="📝 Nhập bình luận để kiểm tra", font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        self.test_input = ctk.CTkEntry(self.test_input_card, placeholder_text="Nhập nội dung cần phân loại cảm xúc...", height=45, font=ctk.CTkFont(size=14))
        self.test_input.pack(fill="x", padx=20, pady=10)
        self.test_input.bind("<Return>", lambda e: self.run_prediction())
        
        self.btn_predict = ctk.CTkButton(
            self.test_input_card, text="🔍 PHÂN TÍCH CẢM XÚC", 
            command=self.run_prediction,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            height=45
        )
        self.btn_predict.pack(pady=(0, 20), padx=20, fill="x")

        # Bottom Area: Result & Chart
        self.test_result_frame = ctk.CTkFrame(self.test_view, fg_color="transparent")
        self.test_result_frame.grid(row=1, column=0, sticky="nsew")
        self.test_result_frame.grid_columnconfigure(0, weight=1) # Text result
        self.test_result_frame.grid_columnconfigure(1, weight=2) # Chart

        # Text Result Card (Converted to ScrollableFrame)
        self.res_text_card = ctk.CTkScrollableFrame(self.test_result_frame, fg_color=CARD_COLOR, corner_radius=15)
        self.res_text_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(self.res_text_card, text="📊 Kết quả dự đoán", font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold")).pack(anchor="w", padx=10, pady=15)
        self.res_label = ctk.CTkLabel(self.res_text_card, text="Chưa có dữ liệu.\nNhấn nút Phân tích để xem kết quả.", justify="left", font=ctk.CTkFont(size=15), text_color=TEXT_SECONDARY)
        self.res_label.pack(anchor="nw", padx=10, pady=10)


        # Chart Card (Converted to ScrollableFrame)
        self.chart_card = ctk.CTkScrollableFrame(self.test_result_frame, fg_color=CARD_COLOR, corner_radius=15, orientation="vertical")
        self.chart_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Create 2 subplots: Bar chart and Pie chart
        # Tăng kích thước Figure để cuộn xem cho sướng
        self.fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.fig.patch.set_facecolor(CARD_COLOR)

        
        # Bar Chart Ax
        self.ax_bar = self.fig.add_subplot(121) # Left: Bar
        self.ax_pie = self.fig.add_subplot(122) # Right: Pie
        
        self.setup_ax_style(self.ax_bar)
        self.setup_ax_style(self.ax_pie)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def setup_ax_style(self, ax):
        ax.set_facecolor(CARD_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#334155')


    def update_test_model_info(self):
        if not self.loaded_model_id:
            try:
                registry = ModelRegistry()
                best = registry.get_best_model()
                if best:
                    self.tab_subtitle.configure(text=f"Đang sử dụng mô hình tốt nhất: {best['model_id']} (F1: {best['metrics']['macro_f1']*100:.1f}%)")
                else:
                    self.tab_subtitle.configure(text="CẢNH BÁO: Chưa có mô hình nào được huấn luyện!", text_color=WARNING_COLOR)
            except: pass

    def run_prediction(self):
        text = self.test_input.get().strip()
        if not text: return

        def task():
            try:
                from predict import predict_emotions
                from utils import load_model
                import torch

                # Load model if not cached
                registry = ModelRegistry()
                best = registry.get_best_model()
                if not best:
                    self.after(0, lambda: messagebox.showerror("Lỗi", "Không tìm thấy mô hình nào để dự đoán!"))
                    return

                if self.loaded_model_id != best['model_id']:
                    print(f"🔄 Đang tải mô hình mới: {best['model_id']}...")
                    self.after(0, lambda: self.btn_predict.configure(text="⏳ Đang tải mô hình...", state="disabled"))
                    
                    device = Config.DEVICE
                    model_path = os.path.join(Config.MODEL_REGISTRY_DIR, "models", best['model_id'])
                    self.current_model, self.current_tokenizer = load_model(model_path, device)
                    self.loaded_model_id = best['model_id']

                self.after(0, lambda: self.btn_predict.configure(text="⏳ Đang phân tích...", state="disabled"))
                
                device = Config.DEVICE
                result = predict_emotions(text, self.current_model, self.current_tokenizer, device)
                
                # Update UI
                self.after(0, lambda r=result: self.update_prediction_ui(r))
                
            except Exception as e:
                print(f"❌ Lỗi dự đoán: {e}")
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Lỗi", f"Không thể dự đoán: {msg}"))

            finally:
                self.after(0, lambda: self.btn_predict.configure(text="🔍 PHÂN TÍCH CẢM XÚC", state="normal"))

        threading.Thread(target=task, daemon=True).start()

    def update_prediction_ui(self, result):
        # Định nghĩa các nhóm cảm xúc
        pos_labels = ['joy', 'trust', 'love', 'proud', 'excited', 'calm']
        neg_labels = ['fear', 'sadness', 'disgust', 'anger', 'worried', 'disappointed', 'embarrassed', 'jealous']
        neu_labels = ['surprise', 'anticipation']

        # Update text results
        emotions_vi = [Config.EMOTION_LABELS_VI.get(e, e) for e in result['emotions']]
        res_text = "🎯 Cảm xúc phát hiện:\n"
        if emotions_vi:
            res_text += "\n".join([f" • {e}" for e in emotions_vi])
        else:
            res_text += " (Không rõ ràng)"
        
        res_text += "\n\n📈 Top 5 tin cậy nhất:\n"
        sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
        for emo, score in sorted_scores[:5]:
            emo_vi = Config.EMOTION_LABELS_VI.get(emo, emo)
            res_text += f" • {emo_vi:15s}: {score*100:>5.1f}%\n"
            
        self.res_label.configure(text=res_text, text_color=TEXT_PRIMARY)

        # Update Charts
        self.ax_bar.clear()
        self.ax_pie.clear()
        
        # 1. Bar Chart (Top 10 labels for clarity)
        top_10 = sorted_scores[:10][::-1]
        labels = [Config.EMOTION_LABELS_VI.get(emo, emo) for emo, _ in top_10]
        values = [score for _, score in top_10]
        
        bars = self.ax_bar.barh(labels, values, color=ACCENT_COLOR)
        for i, (emo, _) in enumerate(top_10):
            if emo in result['emotions']:
                bars[i].set_color(SUCCESS_COLOR)

        self.ax_bar.set_xlim(0, 1.0)
        self.ax_bar.set_title("Top 10 Cảm xúc", color=TEXT_PRIMARY, fontsize=10, weight="bold")
        self.setup_ax_style(self.ax_bar)

        # 2. Pie Chart (Positive vs Negative vs Neutral)
        pos_score = sum(result['scores'][l] for l in pos_labels)
        neg_score = sum(result['scores'][l] for l in neg_labels)
        neu_score = sum(result['scores'][l] for l in neu_labels)
        
        total = pos_score + neg_score + neu_score
        if total > 0:
            sizes = [pos_score/total, neg_score/total, neu_score/total]
            pie_labels = ['Tích cực', 'Tiêu cực', 'Trung tính']
            colors = [SUCCESS_COLOR, '#ef4444', TEXT_SECONDARY]
            
            # Chỉ vẽ những phần có giá trị > 0.01
            final_sizes = []
            final_labels = []
            final_colors = []
            for s, l, c in zip(sizes, pie_labels, colors):
                if s > 0.01:
                    final_sizes.append(s)
                    final_labels.append(l)
                    final_colors.append(c)

            self.ax_pie.pie(final_sizes, labels=final_labels, autopct='%1.1f%%', 
                           colors=final_colors, textprops={'color': TEXT_PRIMARY, 'fontsize': 8},
                           startangle=90, pctdistance=0.85)
            
            # Vẽ hình tròn giữa để tạo biểu đồ donut
            centre_circle = plt.Circle((0,0), 0.70, fc=CARD_COLOR)
            self.ax_pie.add_artist(centre_circle)
        
        self.ax_pie.set_title("Tỉ lệ cảm xúc", color=TEXT_PRIMARY, fontsize=10, weight="bold")
        
        self.fig.tight_layout()
        self.canvas.draw()


    def upload_best_model_to_cloud(self):
        """Đẩy model tốt nhất lên Hugging Face."""
        if not self.check_hf_status():
            messagebox.showwarning("Chưa kết nối", "Bạn cần đăng nhập Hugging Face trước!\n\nVào tab 'Tài Khoản HF' để đăng nhập.")
            self.show_login()
            return
        
        # Kiểm tra có model tốt nhất không
        best_model = self.registry.get_best_model()
        if not best_model:
            messagebox.showwarning("Không có model", "Chưa có model nào được huấn luyện!\n\nHãy huấn luyện model trước.")
            return
        
        model_id = best_model['model_id']
        f1_score = best_model['metrics']['macro_f1'] * 100
        
        if not messagebox.askyesno(
            "Xác nhận đẩy lên Cloud", 
            f"Bạn muốn đẩy model tốt nhất lên Hugging Face?\n\n"
            f"🆔 Model ID: {model_id}\n"
            f"🎯 F1 Score: {f1_score:.2f}%\n\n"
            f"Model sẽ được upload lên repository của bạn."
        ):
            return
        
        # Disable nút và hiển thị trạng thái
        self.btn_upload_model.configure(state="disabled", text="☁️ Đang upload...")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        def upload_thread():
            try:
                from model_sharing import ModelSharing
                
                print("\n" + "☁️ "*15 + "\nĐANG ĐẨY MODEL LÊN HUGGING FACE...\n" + "☁️ "*15)
                print(f"📦 Model ID: {model_id}")
                print(f"🎯 F1 Score: {f1_score:.2f}%")
                
                sharing = ModelSharing()
                success = sharing.sync_best_model()
                
                if success:
                    print("✅ Upload thành công!")
                    self.after(0, lambda: messagebox.showinfo(
                        "Thành công", 
                        f"✅ Model đã được đẩy lên Hugging Face!\n\n"
                        f"Model ID: {model_id}\n"
                        f"F1 Score: {f1_score:.2f}%\n\n"
                        f"Các thành viên khác có thể tải về và sử dụng."
                    ))
                else:
                    print("❌ Upload thất bại!")
                    self.after(0, lambda: messagebox.showerror(
                        "Lỗi", 
                        "❌ Không thể upload model lên Hugging Face!\n\n"
                        "Kiểm tra:\n"
                        "• Kết nối Internet\n"
                        "• Token có quyền WRITE\n"
                        "• Repository tồn tại"
                    ))
                    
            except Exception as e:
                print(f"❌ Lỗi upload: {e}")
                import traceback
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi upload: {e}"))
            finally:
                # Reset nút và progress bar
                self.after(0, lambda: self.btn_upload_model.configure(state="normal", text="☁️ Đẩy Model Tốt Nhất Lên Cloud"))
                self.after(0, lambda: self.progress_bar.stop())
                self.after(0, lambda: self.progress_bar.set(0))
        
        threading.Thread(target=upload_thread, daemon=True).start()

    def finish_training(self, success):
        self.is_training = False
        def update():
            self.btn_start_train.configure(state="normal", text="🚀 KHỞI ĐỘNG HUẤN LUYỆN", fg_color=SUCCESS_COLOR)
            self.progress_bar.stop()
            self.progress_bar.set(0)
            self.update_best_model_info()
            if success: messagebox.showinfo("Hoàn tất", "🎉 AI đã học xong kiến thức mới!")
            else: messagebox.showerror("Thất bại", "⚠️ Quá trình học bị gián đoạn. Xem log để biết chi tiết.")
        self.after(0, update)

if __name__ == "__main__":
    app = PremiumEmotionApp()
    app.mainloop()
