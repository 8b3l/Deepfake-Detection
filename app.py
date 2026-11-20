import logging
import math
import time
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import filedialog, messagebox
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from processor import AnalysisError, SEQUENCE_LENGTH, VideoAnalysis, classify_video


LOGGER = logging.getLogger(__name__)

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
        self.playback_job = None
        self.analysis_job = None
        self.fps = 30.0
        self.current_frame_pos = 0
        self.play_start_time = 0.0
        self.paused_elapsed = 0.0
        self.preview_image = None
        self.photo_image = None
        self.resume_after_analysis = False
        self.analysis_events: Queue[tuple[str, str | VideoAnalysis]] = Queue()

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
            "• Spatial features: ResNet50",
            "• Temporal sequence: GRU",
            "• Local video analysis",
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
            text="Select an MP4, AVI, MOV, or MKV video to begin",
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

        # Use a native canvas so preview dimensions are measured in pixels.
        self.video_canvas = tk.Canvas(
            self.preview_frame,
            bg="#050811",
            highlightthickness=0,
            bd=0,
        )

        self.preview_item = self.video_canvas.create_image(0, 0, anchor="center")
        self.video_canvas.bind("<Configure>", self.render_preview)

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
            text="Ready to analyze",
            font=("Segoe UI", 12),
            text_color=self.colors["text_dim"],
        )
        self.progress_label.pack(side="left")

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
        self.result_label.place(relx=0.5, rely=0.40, anchor="center")

        self.result_detail = ctk.CTkLabel(
            self.result_frame,
            text="",
            font=("Segoe UI", 12),
            text_color=self.colors["text"],
            wraplength=650,
        )
        self.result_detail.place(relx=0.5, rely=0.73, anchor="center")

    def stop_video_stream(self):
        self.is_playing = False
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
            self.playback_job = None
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

        capture = None
        try:
            capture = cv2.VideoCapture(file_path)
            ok, frame = capture.read()
        except cv2.error:
            LOGGER.exception("Could not open the video preview")
            ok = False
        if not ok:
            if capture is not None:
                capture.release()
            messagebox.showerror(
                "Could not open video",
                "The selected file contains no readable frames. Check the file and its codec.",
                parent=self,
            )
            return

        self.stop_video_stream()
        self.video_path = file_path
        self.cap = capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 30.0
        height, width = frame.shape[:2]
        self.file_label.configure(
            text=f"📄 {Path(file_path).name}   |   {width}×{height} • {self.fps:.0f} FPS",
            text_color=self.colors["text"],
        )

        self.preview_display.place_forget()
        self.preview_subtext.place_forget()
        self.video_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.preview_image = frame
        self.render_preview()

        self.current_frame_pos = 0
        self.paused_elapsed = 0.0
        self.toggle_playback()

        self.btn_analyze.configure(state="normal")
        self.btn_toggle_play.configure(state="normal")
        self.update_result("READY FOR ANALYSIS", "", "cyan", "panel")
        self.status_badge.configure(
            text="● System Status: Video Loaded",
            text_color=self.colors["cyan"],
        )
        self.progress_label.configure(text="Ready to analyze")
        self.progress_bar.set(0)

    def render_preview(self, _event=None):
        if self.preview_image is None:
            return
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        height, width = self.preview_image.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        size = (max(int(width * scale), 1), max(int(height * scale), 1))
        resized = cv2.resize(self.preview_image, size)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.video_canvas.coords(self.preview_item, canvas_width // 2, canvas_height // 2)
        self.video_canvas.itemconfigure(self.preview_item, image=self.photo_image)

    def update_video_frame(self):
        self.playback_job = None
        if not self.is_playing or self.cap is None:
            return

        elapsed = time.perf_counter() - self.play_start_time
        target_frame = int(elapsed * self.fps)
        if target_frame > self.current_frame_pos:
            while self.current_frame_pos + 1 < target_frame:
                if not self.cap.grab():
                    break
                self.current_frame_pos += 1

            ok, frame = self.cap.read()
            if ok:
                self.current_frame_pos += 1
            else:
                # Use decoding to detect the actual end, even if metadata is inaccurate.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    self.stop_video_stream()
                    self.btn_toggle_play.configure(state="disabled", text="▶️ Play")
                    return
                self.current_frame_pos = 0
                self.play_start_time = time.perf_counter()

            self.preview_image = frame
            self.render_preview()

        self.playback_job = self.after(15, self.update_video_frame)

    def toggle_playback(self):
        if self.cap is None:
            return
        if self.is_playing:
            self.is_playing = False
            self.paused_elapsed = time.perf_counter() - self.play_start_time
            if self.playback_job is not None:
                self.after_cancel(self.playback_job)
                self.playback_job = None
            self.btn_toggle_play.configure(text="▶️ Play")
        else:
            self.is_playing = True
            self.play_start_time = time.perf_counter() - self.paused_elapsed
            self.btn_toggle_play.configure(text="⏸ Pause")
            self.update_video_frame()

    def start_analysis(self):
        if not self.video_path or self.analysis_job is not None:
            return

        self.resume_after_analysis = self.is_playing
        if self.is_playing:
            self.toggle_playback()
        self.btn_analyze.configure(state="disabled", text="⏳   ANALYZING...")
        self.btn_select.configure(state="disabled")
        self.btn_toggle_play.configure(state="disabled")
        self.status_badge.configure(text="● System Status: Analyzing", text_color=self.colors["cyan"])
        self.update_result("ANALYZING VIDEO", "", "cyan", "panel")
        self.progress_label.configure(text="Starting analysis")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

        Thread(target=self.run_logic, args=(self.video_path,), daemon=True).start()
        self.analysis_job = self.after(50, self.poll_analysis)

    def run_logic(self, video_path):
        try:
            result = classify_video(
                video_path,
                on_stage=lambda stage: self.analysis_events.put(("stage", stage)),
            )
        except AnalysisError as exc:
            LOGGER.warning("Analysis failed: %s", exc, exc_info=exc.__cause__ is not None)
            self.analysis_events.put(("error", str(exc)))
        except Exception:
            LOGGER.exception("Unexpected analysis error")
            self.analysis_events.put(("error", "An unexpected error occurred during analysis."))
        else:
            self.analysis_events.put(("result", result))

    def poll_analysis(self):
        self.analysis_job = None
        try:
            while True:
                event, value = self.analysis_events.get_nowait()
                if event == "stage":
                    self.progress_label.configure(text=value)
                else:
                    self.finish_analysis(event, value)
                    return
        except Empty:
            self.analysis_job = self.after(50, self.poll_analysis)

    def update_result(self, title, detail, color, background):
        self.result_label.configure(text=title, text_color=self.colors[color])
        self.result_detail.configure(text=detail)
        self.result_frame.configure(
            fg_color=self.colors[background],
            border_color=self.colors[color],
        )
        self.preview_frame.configure(border_color=self.colors[color])

    def finish_analysis(self, event, value):
        success = event == "result"
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1 if success else 0)
        self.progress_label.configure(text="Analysis complete" if success else "Analysis failed")

        if success:
            detail = (
                f"Model fake score {value.fake_score:.4f}"
                f"  •  {value.sampled_frames}/{SEQUENCE_LENGTH} frames sampled"
            )
            if value.label == "fake":
                self.update_result("LIKELY FAKE", detail, "danger", "danger_bg")
            else:
                self.update_result("LIKELY REAL", detail, "accent", "accent_bg")
        else:
            self.update_result("COULD NOT ANALYZE", value, "danger", "panel")

        self.status_badge.configure(
            text="● System Status: Complete" if success else "● System Status: Error",
            text_color=self.colors["cyan"] if success else self.colors["danger"],
        )
        self.btn_analyze.configure(state="normal", text="⚡   START ANALYSIS")
        self.btn_select.configure(state="normal")
        if self.cap is not None:
            self.btn_toggle_play.configure(state="normal")
            if self.resume_after_analysis:
                self.toggle_playback()

    def on_closing(self):
        if self.analysis_job is not None:
            self.after_cancel(self.analysis_job)
            self.analysis_job = None
            self.progress_bar.stop()
        self.stop_video_stream()
        self.destroy()


if __name__ == "__main__":
    app = DeepFakeApp()
    app.mainloop()
