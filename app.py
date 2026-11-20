import os
import time
import tkinter as tk
from threading import Thread
from tkinter import filedialog
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from processor import classify_video

# Configure CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class DeepFakeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Main window configuration
        self.title("DeepFake Detection System | AI Video Forensics")
        self.geometry("1150x720")
        self.resizable(True, True)
        self.minsize(1020, 640)

        # Modern Cyber-Slate & Emerald/Crimson color palette
        self.colors = {
            "bg": "#0B0F19",              # Deep Midnight Background
            "panel": "#111827",           # Rich Slate Sidebar & Cards
            "card_inner": "#1E293B",      # Elevated Surface
            "border": "#334155",          # Subtle Slate Border
            "primary": "#2563EB",         # Vibrant Royal Blue
            "primary_hover": "#1D4ED8",
            "cyan": "#38BDF8",            # Electric Cyan Accent
            "accent": "#10B981",          # Emerald Green (REAL)
            "accent_bg": "#064E3B",
            "danger": "#F43F5E",          # Crimson Rose (FAKE)
            "danger_bg": "#4C0519",
            "text": "#F8FAFC",            # Crisp White Text
            "text_dim": "#94A3B8",        # Muted Slate Text
        }

        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.after_id = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_pos = 0
        self.play_start_time = 0.0
        self.paused_elapsed = 0.0
        self.photo_image = None

        self.configure(fg_color=self.colors["bg"])
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Responsive Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_area()

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(
            self,
            width=290,
            corner_radius=0,
            fg_color=self.colors["panel"],
            border_width=1,
            border_color="#1E293B",
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # App Branding
        title_label = ctk.CTkLabel(
            sidebar,
            text="🛡️ DF DETECTOR",
            font=("Segoe UI", 26, "bold"),
            text_color=self.colors["cyan"],
        )
        title_label.pack(pady=(38, 4))

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Spatiotemporal AI Video Forensics",
            font=("Segoe UI", 12),
            text_color=self.colors["text_dim"],
        )
        subtitle.pack(pady=(0, 35))

        # Upload Video Button
        self.btn_select = ctk.CTkButton(
            sidebar,
            text="📂   SELECT VIDEO",
            font=("Segoe UI", 14, "bold"),
            height=48,
            corner_radius=10,
            fg_color=self.colors["card_inner"],
            hover_color="#334155",
            border_width=1,
            border_color=self.colors["border"],
            text_color=self.colors["text"],
            command=self.select_video,
        )
        self.btn_select.pack(padx=22, pady=(0, 14), fill="x")

        # Start Analysis Button
        self.btn_analyze = ctk.CTkButton(
            sidebar,
            text="⚡   START ANALYSIS",
            font=("Segoe UI", 14, "bold"),
            height=48,
            corner_radius=10,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            text_color="#FFFFFF",
            text_color_disabled="#94A3B8",
            state="disabled",
            command=self.start_analysis,
        )
        self.btn_analyze.pack(padx=22, pady=(0, 14), fill="x")

        # Model Pipeline Specification Card
        info_frame = ctk.CTkFrame(
            sidebar,
            fg_color=self.colors["bg"],
            corner_radius=12,
            border_width=1,
            border_color="#1E293B",
        )
        info_frame.pack(padx=22, fill="x", side="bottom", pady=24)

        ctk.CTkLabel(
            info_frame,
            text="PIPELINE ARCHITECTURE",
            font=("Segoe UI", 11, "bold"),
            text_color=self.colors["cyan"],
        ).pack(anchor="w", padx=15, pady=(12, 6))

        specs = [
            "• Face Alignment: MTCNN",
            "• Spatial Features: ResNet50",
            "• Temporal Sequence: GRU",
        ]
        for spec in specs:
            ctk.CTkLabel(
                info_frame,
                text=spec,
                font=("Consolas", 11),
                text_color=self.colors["text_dim"],
            ).pack(anchor="w", padx=15, pady=1)

        self.status_badge = ctk.CTkLabel(
            info_frame,
            text="● System Status: Ready",
            font=("Consolas", 11, "bold"),
            text_color=self.colors["accent"],
        )
        self.status_badge.pack(anchor="w", padx=15, pady=(10, 12))

    def create_main_area(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)

        # Top File Info & Playback Control Header
        top_bar = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 12))

        self.file_label = ctk.CTkLabel(
            top_bar,
            text="No video selected — Upload an MP4, AVI, or MOV file to begin",
            font=("Segoe UI", 14),
            text_color=self.colors["text_dim"],
            anchor="w",
        )
        self.file_label.pack(side="left", fill="x", expand=True)

        self.btn_toggle_play = ctk.CTkButton(
            top_bar,
            text="⏸ Pause",
            width=95,
            height=32,
            corner_radius=8,
            font=("Segoe UI", 12, "bold"),
            fg_color=self.colors["card_inner"],
            hover_color="#334155",
            text_color=self.colors["text"],
            state="disabled",
            command=self.toggle_playback,
        )
        self.btn_toggle_play.pack(side="right")

        # Video Preview Frame
        self.preview_frame = ctk.CTkFrame(
            main_frame,
            fg_color="#050811",
            corner_radius=16,
            border_width=2,
            border_color="#1E293B",
        )
        self.preview_frame.pack(fill="both", expand=True)
        self.preview_frame.pack_propagate(False)

        # Native 1:1 Pixel Canvas (prevents Windows DPI zoom-in)
        self.video_canvas = tk.Canvas(
            self.preview_frame,
            bg="#050811",
            highlightthickness=0,
            bd=0,
        )

        # Standby placeholders
        self.preview_display = ctk.CTkLabel(
            self.preview_frame,
            text="🎬",
            font=("Segoe UI", 64),
            text_color="#334155",
        )
        self.preview_display.place(relx=0.5, rely=0.44, anchor="center")

        self.preview_subtext = ctk.CTkLabel(
            self.preview_frame,
            text="SYSTEM STANDBY — AWAITING VIDEO INPUT",
            font=("Segoe UI", 13, "bold"),
            text_color="#475569",
        )
        self.preview_subtext.place(relx=0.5, rely=0.58, anchor="center")

        # Progress Bar Section
        progress_header = ctk.CTkFrame(main_frame, fg_color="transparent")
        progress_header.pack(fill="x", pady=(18, 6))

        self.progress_label = ctk.CTkLabel(
            progress_header,
            text="Inference Progress",
            font=("Segoe UI", 12),
            text_color=self.colors["text_dim"],
        )
        self.progress_label.pack(side="left")

        self.progress_percent = ctk.CTkLabel(
            progress_header,
            text="0%",
            font=("Consolas", 12, "bold"),
            text_color=self.colors["cyan"],
        )
        self.progress_percent.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            height=10,
            corner_radius=5,
            fg_color="#1E293B",
            progress_color=self.colors["cyan"],
        )
        self.progress_bar.pack(fill="x", pady=(0, 14))
        self.progress_bar.set(0)

        # Result Banner Card
        self.result_frame = ctk.CTkFrame(
            main_frame,
            height=95,
            fg_color=self.colors["panel"],
            corner_radius=14,
            border_width=1,
            border_color="#1E293B",
        )
        self.result_frame.pack(fill="x")
        self.result_frame.pack_propagate(False)

        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="AWAITING ANALYSIS",
            font=("Segoe UI", 26, "bold"),
            text_color="#475569",
        )
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

    def stop_video_stream(self):
        self.is_playing = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not file_path:
            return

        self.stop_video_stream()
        self.video_path = file_path
        filename = os.path.basename(file_path)

        # Open video stream for real-time 1.0x embedded playback
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not fps or fps <= 0 or fps > 120:
                fps = 30.0
            self.fps = fps

            self.file_label.configure(
                text=f"📄 {filename}   |   {width}×{height} • {fps:.0f} FPS",
                text_color=self.colors["text"],
            )

            # Switch from standby labels to native video canvas
            self.preview_display.place_forget()
            self.preview_subtext.place_forget()
            self.video_canvas.pack(fill="both", expand=True, padx=6, pady=6)
            self.update_idletasks()

            self.current_frame_pos = 0
            self.play_start_time = time.perf_counter()
            self.paused_elapsed = 0.0
            self.is_playing = True

            self.btn_toggle_play.configure(state="normal", text="⏸ Pause")
            self.update_video_frame()
        else:
            self.file_label.configure(
                text=f"⚠️ Failed to open video: {filename}",
                text_color=self.colors["danger"],
            )
            return

        # Reset UI states for new video
        self.btn_analyze.configure(state="normal")
        self.preview_frame.configure(border_color=self.colors["cyan"])
        self.result_frame.configure(
            fg_color=self.colors["panel"],
            border_width=1,
            border_color="#1E293B",
        )
        self.result_label.configure(
            text="READY FOR ANALYSIS",
            text_color=self.colors["cyan"],
        )
        self.status_badge.configure(
            text="● System Status: Video Loaded",
            text_color=self.colors["cyan"],
        )
        self.update_progress(0.0)

    def update_video_frame(self):
        if not self.is_playing or self.cap is None:
            return

        # Sync frame position with real-world wall clock for exact 1.0x speed
        elapsed = time.perf_counter() - self.play_start_time
        target_frame = int(elapsed * self.fps)

        if self.total_frames > 0 and target_frame >= self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_start_time = time.perf_counter()
            self.current_frame_pos = 0
            target_frame = 0

        # Skip intermediate frames if rendering took slightly longer
        while self.current_frame_pos < target_frame:
            if not self.cap.grab():
                break
            self.current_frame_pos += 1

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_start_time = time.perf_counter()
            self.current_frame_pos = 0
            ret, frame = self.cap.read()

        if ret:
            self.current_frame_pos += 1
            cw = self.video_canvas.winfo_width()
            ch = self.video_canvas.winfo_height()

            if cw > 10 and ch > 10:
                h, w = frame.shape[:2]
                scale = min(cw / w, ch / h)
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)

                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self.photo_image = ImageTk.PhotoImage(image=img)

                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    cw // 2,
                    ch // 2,
                    image=self.photo_image,
                    anchor="center",
                )

        self.after_id = self.after(15, self.update_video_frame)

    def toggle_playback(self):
        if self.cap is None:
            return
        if self.is_playing:
            self.is_playing = False
            self.paused_elapsed = time.perf_counter() - self.play_start_time
            if self.after_id is not None:
                self.after_cancel(self.after_id)
                self.after_id = None
            self.btn_toggle_play.configure(text="▶️ Play")
        else:
            self.is_playing = True
            self.play_start_time = time.perf_counter() - self.paused_elapsed
            self.btn_toggle_play.configure(text="⏸ Pause")
            self.update_video_frame()

    def update_progress(self, value):
        self.progress_bar.set(value)
        self.progress_percent.configure(text=f"{int(value * 100)}%")

    def start_analysis(self):
        if not self.video_path:
            return

        # Pause preview playback during inference to allocate full CPU to TensorFlow
        if self.is_playing:
            self.toggle_playback()

        self.btn_analyze.configure(state="disabled", text="⏳   ANALYZING...")
        self.btn_select.configure(state="disabled")
        self.btn_toggle_play.configure(state="disabled")

        self.status_badge.configure(
            text="● System Status: Running AI...",
            text_color=self.colors["cyan"],
        )
        self.result_label.configure(
            text="EXTRACTING FACES & ANALYZING FRAMES...",
            font=("Segoe UI", 20, "bold"),
            text_color=self.colors["cyan"],
        )

        Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            for i in range(1, 4):
                time.sleep(0.25)
                val = i / 10
                self.after(0, lambda v=val: self.update_progress(v))

            label, confidence = classify_video(self.video_path)

            for i in range(4, 11):
                time.sleep(0.08)
                val = i / 10
                self.after(0, lambda v=val: self.update_progress(v))

            self.after(0, lambda: self.show_result(label, confidence))

        except Exception as e:
            print(f"Analysis error: {e}")
            self.after(
                0,
                lambda: self.file_label.configure(
                    text="⚠️ Error occurred during video analysis!",
                    text_color=self.colors["danger"],
                ),
            )
            self.after(0, self.reset_buttons)

    def show_result(self, label, confidence):
        percentage = f"{confidence * 100:.1f}%"

        if label == "real":
            text = f"✅  AUTHENTIC (REAL)   •   Confidence: {percentage}"
            color = self.colors["accent"]
            bg_color = self.colors["accent_bg"]
        else:
            text = f"🚨  DEEPFAKE DETECTED (FAKE)   •   Confidence: {percentage}"
            color = self.colors["danger"]
            bg_color = self.colors["danger_bg"]

        self.result_label.configure(
            text=text,
            font=("Segoe UI", 26, "bold"),
            text_color=color,
        )
        self.result_frame.configure(
            fg_color=bg_color,
            border_width=2,
            border_color=color,
        )
        self.preview_frame.configure(border_color=color)
        self.status_badge.configure(
            text="● System Status: Analysis Complete",
            text_color=color,
        )

        self.reset_buttons()

        # Resume embedded video playback automatically
        if self.cap is not None and not self.is_playing:
            self.toggle_playback()

    def reset_buttons(self):
        self.btn_analyze.configure(state="normal", text="⚡   START ANALYSIS")
        self.btn_select.configure(state="normal")
        self.btn_toggle_play.configure(state="normal")

    def on_closing(self):
        self.stop_video_stream()
        self.destroy()


if __name__ == "__main__":
    app = DeepFakeApp()
    app.mainloop()