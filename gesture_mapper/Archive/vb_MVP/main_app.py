import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import pickle
import yaml
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np

# Import our modules
import gesture_recorder
import live_gesture_listener
import gesture_matcher
from landmark_visualizer import create_visualizer
from temporal_smoother import create_smoother, GestureStateTracker

TEMPLATES_DIR = "templates"
CONFIG_PATH = "gesture_config.yaml"

class ModernGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Control System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')  # Dark theme
        
        # Data
        self.gesture_names = self.load_gesture_names()
        self.mappings = self.load_mappings()
        
        # UI State
        self.current_page = "home"  # home, recording, library, settings
        self.selected_gesture = None
        self.listener_active = False
        self.listener_thread = None
        self.stop_listener_event = threading.Event()
        
        # Camera and visualization
        self.camera_feed = None
        self.landmark_feed = None
        self.cap = None
        self.visualizer = None
        self.smoother = create_smoother("exponential", alpha=0.2)
        self.gesture_tracker = GestureStateTracker(min_duration=0.3, confidence_threshold=0.6)
        
        # Status variables
        self.status_vars = {
            'recognition': tk.StringVar(value="Inactive"),
            'best_match': tk.StringVar(value="None"),
            'confidence': tk.StringVar(value="0%"),
            'gesture_count': tk.StringVar(value="0"),
            'camera_status': tk.StringVar(value="Disconnected"),
            'processing_status': tk.StringVar(value="Idle")
        }
        
        self.create_widgets()
        self.setup_camera()
        self.update_gesture_count()
        
    def create_widgets(self):
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top status bar
        self.create_status_bar()
        
        # Main content area
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # Initialize home page
        self.show_home_page()
        
    def create_status_bar(self):
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill="x", pady=(0, 10))
        
        # Status indicators
        indicators = [
            ("Recognition:", self.status_vars['recognition']),
            ("Best Match:", self.status_vars['best_match']),
            ("Confidence:", self.status_vars['confidence']),
            ("Gestures:", self.status_vars['gesture_count']),
            ("Camera:", self.status_vars['camera_status']),
            ("Processing:", self.status_vars['processing_status'])
        ]
        
        for i, (label, var) in enumerate(indicators):
            ttk.Label(status_frame, text=label, font=('Arial', 9)).grid(row=0, column=i*2, padx=(10, 5), pady=5)
            ttk.Label(status_frame, textvariable=var, font=('Arial', 9, 'bold')).grid(row=0, column=i*2+1, padx=(0, 20), pady=5)
            
    def show_home_page(self):
        self.current_page = "home"
        self.clear_content()
        
        # Main content area with camera feed
        camera_frame = ttk.Frame(self.content_frame)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Camera feed label
        self.camera_label = ttk.Label(camera_frame, text="Camera Feed", font=('Arial', 12, 'bold'))
        self.camera_label.pack(pady=(0, 10))
        
        # Camera display
        self.camera_display = tk.Label(camera_frame, text="Initializing camera...", 
                                       relief="solid", bg="black", fg="white")
        self.camera_display.pack(fill="both", expand=True)
        
        # Landmark visualization frame
        landmark_frame = ttk.Frame(self.content_frame)
        landmark_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        
        # Landmark feed label
        self.landmark_label = ttk.Label(landmark_frame, text="3D Landmark Visualization", 
                                       font=('Arial', 12, 'bold'))
        self.landmark_label.pack(pady=(0, 10))
        
        # Landmark display
        self.visualizer = create_visualizer(landmark_frame, use_matplotlib=True)
        self.visualizer.start_animation()
        
        # Side panel
        self.create_side_panel()
        
        # Start camera update loop
        self.update_camera_feed()
        
    def create_side_panel(self):
        side_panel = ttk.Frame(self.content_frame)
        side_panel.pack(side="right", fill="y", padx=(10, 0))
        
        # Title
        ttk.Label(side_panel, text="Controls", font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # Main action buttons
        self.start_stop_btn = ttk.Button(side_panel, text="Start Recognition", 
                                        command=self.toggle_recognition, width=20)
        self.start_stop_btn.pack(pady=5)
        
        ttk.Button(side_panel, text="Record New Gesture", 
                  command=self.show_recording_page, width=20).pack(pady=5)
        
        ttk.Button(side_panel, text="Gesture Library", 
                  command=self.show_library_page, width=20).pack(pady=5)
        
        ttk.Button(side_panel, text="Action Mapper", 
                  command=self.show_mapper_page, width=20).pack(pady=5)
        
        ttk.Button(side_panel, text="Settings", 
                  command=self.show_settings_page, width=20).pack(pady=5)
        
        # Quick actions section
        ttk.Separator(side_panel, orient="horizontal").pack(fill="x", pady=20)
        ttk.Label(side_panel, text="Quick Actions", font=('Arial', 12, 'bold')).pack()
        
        ttk.Button(side_panel, text="Test Camera", 
                  command=self.test_camera, width=20).pack(pady=5)
        
        ttk.Button(side_panel, text="Refresh Gestures", 
                  command=self.refresh_gestures, width=20).pack(pady=5)
        
    def show_recording_page(self):
        self.current_page = "recording"
        self.clear_content()
        
        # Split view layout
        left_frame = ttk.Frame(self.content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(self.content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Left side - Camera feed
        ttk.Label(left_frame, text="Camera Feed", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        self.recording_camera_display = tk.Label(left_frame, text="Camera feed...", 
                                                 relief="solid", bg="black", fg="white")
        self.recording_camera_display.pack(fill="both", expand=True)
        
        # Recording controls
        recording_controls = ttk.Frame(left_frame)
        recording_controls.pack(pady=20)
        
        self.record_btn = ttk.Button(recording_controls, text="Start Recording", 
                                    command=self.start_recording)
        self.record_btn.pack(side="left", padx=5)
        
        ttk.Button(recording_controls, text="Back to Home", 
                  command=self.show_home_page).pack(side="left", padx=5)
        
        # Right side - 3D visualization and template management
        ttk.Label(right_frame, text="3D Landmark Visualization", 
                 font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Gesture selection for visualization
        selection_frame = ttk.Frame(right_frame)
        selection_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(selection_frame, text="Visualize Gesture:").pack(side="left")
        self.gesture_var = tk.StringVar()
        self.gesture_combo = ttk.Combobox(selection_frame, textvariable=self.gesture_var, 
                                         values=["None"] + self.gesture_names)
        self.gesture_combo.pack(side="left", padx=5)
        self.gesture_combo.bind("<<ComboboxSelected>>", self.on_gesture_selected)
        
        # 3D visualization area
        self.recording_visualizer = create_visualizer(right_frame, use_matplotlib=True)
        self.recording_visualizer.start_animation()
        
        # Template management
        template_frame = ttk.LabelFrame(right_frame, text="Template Management")
        template_frame.pack(fill="x", pady=10)
        
        self.template_listbox = tk.Listbox(template_frame, height=6)
        self.template_listbox.pack(fill="x", padx=5, pady=5)
        
        template_buttons = ttk.Frame(template_frame)
        template_buttons.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(template_buttons, text="Delete", 
                  command=self.delete_template).pack(side="left", padx=2)
        ttk.Button(template_buttons, text="Rename", 
                  command=self.rename_template).pack(side="left", padx=2)
        
    def show_library_page(self):
        self.current_page = "library"
        self.clear_content()
        
        # Header
        header_frame = ttk.Frame(self.content_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(header_frame, text="Gesture Library", 
                 font=('Arial', 16, 'bold')).pack(side="left")
        
        ttk.Button(header_frame, text="Back to Home", 
                  command=self.show_home_page).pack(side="right")
        
        # Gesture grid
        self.create_gesture_grid()
        
    def show_mapper_page(self):
        self.current_page = "mapper"
        self.clear_content()
        
        # Header
        header_frame = ttk.Frame(self.content_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(header_frame, text="Action Mapper", 
                 font=('Arial', 16, 'bold')).pack(side="left")
        
        ttk.Button(header_frame, text="Back to Home", 
                  command=self.show_home_page).pack(side="right")
        
        # Mapper content (simplified for now)
        content_frame = ttk.Frame(self.content_frame)
        content_frame.pack(fill="both", expand=True)
        
        ttk.Label(content_frame, text="Action mapping interface will be implemented here", 
                 font=('Arial', 12)).pack(pady=50)
        
    def show_settings_page(self):
        self.current_page = "settings"
        self.clear_content()
        
        # Header
        header_frame = ttk.Frame(self.content_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(header_frame, text="Settings", 
                 font=('Arial', 16, 'bold')).pack(side="left")
        
        ttk.Button(header_frame, text="Back to Home", 
                  command=self.show_home_page).pack(side="right")
        
        # Settings content
        settings_frame = ttk.Frame(self.content_frame)
        settings_frame.pack(fill="both", expand=True)
        
        # Camera settings
        camera_group = ttk.LabelFrame(settings_frame, text="Camera Settings")
        camera_group.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(camera_group, text="Camera Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        camera_var = tk.StringVar(value="0")
        ttk.Entry(camera_group, textvariable=camera_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Recognition settings
        recognition_group = ttk.LabelFrame(settings_frame, text="Recognition Settings")
        recognition_group.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(recognition_group, text="Confidence Threshold:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        threshold_var = tk.StringVar(value="0.6")
        ttk.Entry(recognition_group, textvariable=threshold_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
    def clear_content(self):
        # Clear existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def setup_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.status_vars['camera_status'].set("Connected")
                print("Camera initialized successfully")
            else:
                self.status_vars['camera_status'].set("Failed")
                print("Failed to open camera")
        except Exception as e:
            self.status_vars['camera_status'].set("Error")
            print(f"Camera setup error: {e}")
            
    def update_camera_feed(self):
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Get display size based on current page
                    if self.current_page == "home":
                        display_size = (640, 480)  # Larger for home page
                    else:
                        display_size = (480, 360)  # Smaller for recording page
                    
                    # Resize frame for display
                    frame_resized = cv2.resize(frame, display_size)
                    
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update appropriate display
                    if self.current_page == "home" and hasattr(self, 'camera_display'):
                        self.camera_display.configure(image=photo, text="")
                        self.camera_display.image = photo
                    elif self.current_page == "recording" and hasattr(self, 'recording_camera_display'):
                        self.recording_camera_display.configure(image=photo, text="")
                        self.recording_camera_display.image = photo
                        
                    # Process landmarks for 3D visualization (only if not frozen)
                    if not getattr(self, '_processing_frozen', False):
                        self.process_landmarks_for_visualization(frame)
                    
                    self.status_vars['processing_status'].set("Active")
                else:
                    self.status_vars['processing_status'].set("Frame Error")
            else:
                self.status_vars['processing_status'].set("Camera Error")
        except Exception as e:
            print(f"Camera update error: {e}")
            self.status_vars['processing_status'].set("Error")
            
        # Schedule next update
        self.root.after(33, self.update_camera_feed)  # ~30 FPS
        
    def toggle_recognition(self):
        if not self.listener_active:
            self.start_recognition()
        else:
            self.stop_recognition()
            
    def start_recognition(self):
        if self.listener_thread and self.listener_thread.is_alive():
            return
            
        self.stop_listener_event.clear()
        self.listener_thread = threading.Thread(
            target=live_gesture_listener.start_listener,
            args=(self.stop_listener_event, self.mappings, self.update_status, None),
            daemon=True
        )
        self.listener_thread.start()
        
        self.listener_active = True
        self.start_stop_btn.config(text="Stop Recognition")
        self.status_vars['recognition'].set("Active")
        
    def stop_recognition(self):
        if self.listener_thread and self.listener_thread.is_alive():
            self.stop_listener_event.set()
            self.listener_thread.join(timeout=1.0)
            
        self.listener_active = False
        self.start_stop_btn.config(text="Start Recognition")
        self.status_vars['recognition'].set("Inactive")
        self.status_vars['best_match'].set("None")
        self.status_vars['confidence'].set("0%")
        
    def update_status(self, gesture_name, confidence):
        if gesture_name and confidence is not None:
            self.status_vars['best_match'].set(gesture_name)
            self.status_vars['confidence'].set(f"{confidence:.1%}")
        else:
            self.status_vars['best_match'].set("None")
            self.status_vars['confidence'].set("0%")
            
    def start_recording(self):
        """Start gesture recording process"""
        try:
            # Freeze processing to prevent conflicts
            self._processing_frozen = True
            
            # Get gesture name from user
            gesture_name = simpledialog.askstring("Record Gesture", "Enter gesture name:")
            if gesture_name:
                # Start recording in a separate thread to prevent UI freezing
                recording_thread = threading.Thread(
                    target=self._record_gesture_thread,
                    args=(gesture_name,),
                    daemon=True
                )
                recording_thread.start()
            else:
                # Unfreeze if user cancels
                self._processing_frozen = False
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            self._processing_frozen = False
            
    def _record_gesture_thread(self, gesture_name):
        """Record gesture in separate thread"""
        try:
            # Import and use gesture_recorder
            import gesture_recorder
            success = gesture_recorder.interactive_gesture_editor(
                self.root, gesture_name, True
            )
            
            if success:
                # Refresh gesture list
                self.root.after(0, self.refresh_gestures)
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Gesture '{gesture_name}' recorded successfully!"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to record gesture '{gesture_name}'"))
                
        except Exception as e:
            print(f"Recording error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {e}"))
        finally:
            # Unfreeze processing
            self.root.after(0, lambda: setattr(self, '_processing_frozen', False))
        
    def on_gesture_selected(self, event):
        selected = self.gesture_var.get()
        if selected and selected != "None":
            # Update 3D visualization
            self.update_3d_visualization(selected)
            
    def update_3d_visualization(self, gesture_name):
        # Load and display template landmarks
        if gesture_name and gesture_name in self.gesture_names:
            try:
                templates = gesture_recorder.load_templates()
                if gesture_name in templates and templates[gesture_name]:
                    # Use the first variation for visualization
                    template_landmarks = templates[gesture_name][0]
                    self.recording_visualizer.set_template_landmarks(template_landmarks)
            except Exception as e:
                print(f"Error loading template for visualization: {e}")
        
    def create_gesture_grid(self):
        # Create a scrollable frame for gestures
        canvas = tk.Canvas(self.content_frame)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create gesture grid
        row = 0
        col = 0
        max_cols = 4
        
        for gesture_name in self.gesture_names:
            gesture_frame = ttk.LabelFrame(scrollable_frame, text=gesture_name)
            gesture_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Placeholder for gesture thumbnail
            tk.Label(gesture_frame, text="[Gesture Preview]", 
                    width=20, height=8, relief="solid", bg="lightgray").pack(pady=10)
            
            # Action buttons
            btn_frame = ttk.Frame(gesture_frame)
            btn_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(btn_frame, text="Edit", 
                      command=lambda g=gesture_name: self.edit_gesture(g)).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Delete", 
                      command=lambda g=gesture_name: self.delete_gesture(g)).pack(side="left", padx=2)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def edit_gesture(self, gesture_name):
        messagebox.showinfo("Edit", f"Edit functionality for {gesture_name} will be implemented")
        
    def delete_gesture(self, gesture_name):
        if messagebox.askyesno("Delete", f"Delete gesture '{gesture_name}'?"):
            # Implementation here
            self.refresh_gestures()
            
    def delete_template(self):
        selection = self.template_listbox.curselection()
        if selection:
            # Implementation here
            pass
            
    def rename_template(self):
        selection = self.template_listbox.curselection()
        if selection:
            # Implementation here
            pass
            
    def test_camera(self):
        if self.cap and self.cap.isOpened():
            messagebox.showinfo("Camera Test", "Camera is working correctly!")
        else:
            messagebox.showerror("Camera Test", "Camera is not available!")
            
    def refresh_gestures(self):
        self.gesture_names = self.load_gesture_names()
        self.update_gesture_count()
        if self.current_page == "library":
            self.show_library_page()
            
    def update_gesture_count(self):
        self.status_vars['gesture_count'].set(str(len(self.gesture_names)))
        
    def process_landmarks_for_visualization(self, frame):
        """Process frame to extract landmarks and update 3D visualization"""
        try:
            import mediapipe as mp
            
            # Initialize MediaPipe Hands if not already done
            if not hasattr(self, 'mp_hands'):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=1
                )
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract all 21 landmarks
                    raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    
                    # Normalize landmarks
                    normalized_landmarks = gesture_recorder.normalize_landmarks(raw_coords)
                    
                    if normalized_landmarks:
                        # Select subset of 12 landmarks
                        landmarks_subset = gesture_recorder.select_landmarks_subset(normalized_landmarks)
                        
                        if landmarks_subset and self.visualizer:
                            # Convert landmarks to proper format for visualization
                            landmarks_for_viz = [(float(x), float(y), float(z)) for x, y, z in landmarks_subset]
                            
                            # Update 3D visualization
                            self.visualizer.update_live_landmarks(landmarks_for_viz)
                            
                            # Also update recording page visualizer if on that page
                            if hasattr(self, 'recording_visualizer') and self.recording_visualizer:
                                self.recording_visualizer.update_live_landmarks(landmarks_for_viz)
                            
                            # Apply temporal smoothing
                            smoothed_landmarks = self.smoother.smooth_landmarks(landmarks_for_viz)
                            
                            # Update status with gesture recognition
                            if self.listener_active:
                                gesture_name, confidence = self.gesture_tracker.update(
                                    self.detect_gesture(smoothed_landmarks), 0.8
                                )
                                if gesture_name:
                                    self.status_vars['best_match'].set(gesture_name)
                                    self.status_vars['confidence'].set(f"{confidence:.1%}")
                            
                            break  # Process only first hand
            else:
                # No hands detected, clear visualization
                if self.visualizer:
                    self.visualizer.update_live_landmarks(None)
                if hasattr(self, 'recording_visualizer') and self.recording_visualizer:
                    self.recording_visualizer.update_live_landmarks(None)
                    
        except Exception as e:
            print(f"Error processing landmarks: {e}")
            
    def detect_gesture(self, landmarks):
        """Simple gesture detection for status updates"""
        try:
            templates = gesture_recorder.load_templates()
            if templates:
                gesture_name, confidence = gesture_matcher.match_gesture(landmarks, templates)
                return gesture_name if confidence >= 0.6 else None
        except Exception as e:
            print(f"Error detecting gesture: {e}")
        return None
        
    def load_gesture_names(self):
        names = []
        if not os.path.exists(TEMPLATES_DIR):
            os.makedirs(TEMPLATES_DIR)
        for file in os.listdir(TEMPLATES_DIR):
            if file.endswith(".pkl"):
                names.append(file[:-4])
        return sorted(names)
        
    def load_mappings(self):
        try:
            with open(CONFIG_PATH, "r") as f:
                mappings = yaml.safe_load(f)
                return mappings if mappings else {}
        except FileNotFoundError:
            return {}
            
    def cleanup(self):
        """Clean up resources when closing"""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernGestureApp(root)
    
    # Set up cleanup on window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 