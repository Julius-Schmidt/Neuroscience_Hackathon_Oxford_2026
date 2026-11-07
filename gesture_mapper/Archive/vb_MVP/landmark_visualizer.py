# landmark_visualizer.py
# Module for 3D visualization of hand landmarks

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class LandmarkVisualizer:
    def __init__(self, parent_frame, width=400, height=300):
        self.parent_frame = parent_frame
        self.width = width
        self.height = height
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Data storage
        self.live_landmarks = None
        self.template_landmarks = None
        self.animation_running = False
        
        # Setup the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the 3D plot with proper settings"""
        # Swap X and Z labels to reflect display swap (plotting Z on X axis)
        self.ax.set_xlabel('Z')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('X')
        self.ax.set_title('Hand Landmarks')
        
        # Set consistent view limits - extended to 3 units
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        
        # Set initial view angle
        self.ax.view_init(elev=30, azim=135)
        
        # Set aspect ratio to be equal
        self.ax.set_box_aspect([1, 1, 1])



    def update_live_landmarks(self, landmarks):
        """Update the live landmark data"""
        if landmarks and len(landmarks) == 12:
            # Use landmarks as-is; visualization controls handle perspective
            self.live_landmarks = np.array(landmarks)
            self.redraw()
            
    
            
    def set_template_landmarks(self, landmarks):
        """Set template landmarks for comparison"""
        if landmarks and len(landmarks) == 12:
            # Use landmarks as-is; visualization controls handle perspective
            self.template_landmarks = np.array(landmarks)
            self.redraw()
            
    def clear_template(self):
        """Clear template landmarks"""
        self.template_landmarks = None
        self.redraw()
        
    def redraw(self):
        """Redraw the plot with current data"""
        self.ax.clear()
        self.setup_plot()
        
        def swap_xz(points):
            if points is None:
                return None
            # Swap X and Z for display and mirror original Z: (x, y, z) -> (-z, y, x)
            return np.stack([-points[:,2], points[:,1], points[:,0]], axis=1)

        # Plot live landmarks
        if self.live_landmarks is not None:
            live_display = swap_xz(self.live_landmarks)
            x, y, z = live_display.T
            self.ax.scatter(x, y, z, c='blue', s=80, label='Live', alpha=0.8, edgecolors='white', linewidth=1)
            
            # Connect landmarks with lines (hand structure)
            self.draw_hand_connections(live_display, 'blue')
            
            # Add landmark numbers for live data
            for i, (x_pos, y_pos, z_pos) in enumerate(live_display):
                self.ax.text(x_pos, y_pos, z_pos, str(i), color='blue', fontsize=8, ha='center', va='center')
            
        # Plot template landmarks
        if self.template_landmarks is not None:
            template_display = swap_xz(self.template_landmarks)
            x, y, z = template_display.T
            self.ax.scatter(x, y, z, c='red', s=80, label='Template', alpha=0.8, edgecolors='white', linewidth=1)
            
            # Connect landmarks with lines
            self.draw_hand_connections(template_display, 'red')
            
            # Add landmark numbers for template data
            for i, (x_pos, y_pos, z_pos) in enumerate(template_display):
                self.ax.text(x_pos, y_pos, z_pos, str(i), color='red', fontsize=8, ha='center', va='center')
            
        # Add legend if both are present
        if self.live_landmarks is not None and self.template_landmarks is not None:
            self.ax.legend(loc='upper right')
            
        # Add grid for better orientation
        self.ax.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
        
    def draw_hand_connections(self, landmarks, color):
        """Draw lines connecting landmarks to show hand structure"""
        # Define connections for our 12-landmark subset order:
        # 0: Wrist
        # 1-3: Thumb (CMC, MCP, TIP)
        # 4-5: Index (MCP, TIP)
        # 6-7: Middle (MCP, TIP)
        # 8-9: Ring (MCP, TIP)
        # 10-11: Pinky (MCP, TIP)
        connections = [
            (0, 1),  # Wrist → Thumb CMC
            (1, 2),  # Thumb CMC → MCP
            (2, 3),  # Thumb MCP → TIP
            (0, 4),  # Wrist → Index MCP
            (4, 5),  # Index MCP → TIP
            (0, 6),  # Wrist → Middle MCP
            (6, 7),  # Middle MCP → TIP
            (0, 8),  # Wrist → Ring MCP
            (8, 9),  # Ring MCP → TIP
            (0, 10), # Wrist → Pinky MCP
            (10, 11) # Pinky MCP → TIP
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                self.ax.plot([start[0], end[0]], 
                           [start[1], end[1]], 
                           [start[2], end[2]], 
                           color=color, alpha=0.5, linewidth=1)
                           
    def start_animation(self):
        """Start real-time animation"""
        self.animation_running = True
        self.animate()
        
    def stop_animation(self):
        """Stop real-time animation"""
        self.animation_running = False
        
    def animate(self):
        """Animation loop"""
        if self.animation_running:
            self.redraw()
            self.parent_frame.after(100, self.animate)  # 10 FPS
            
    def get_screenshot(self):
        """Get current plot as PIL Image"""
        # Save to temporary buffer
        self.fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        
        # Convert to PIL Image
        return Image.fromarray(buf)
        
    def set_view_angle(self, elev=None, azim=None):
        """Set the view angle of the 3D plot"""
        if elev is not None:
            self.ax.view_init(elev=elev, azim=self.ax.azim)
        if azim is not None:
            self.ax.view_init(elev=self.ax.elev, azim=azim)
        self.redraw()
        
    def rotate_view(self, delta_azim=10):
        """Rotate the view by a small amount"""
        current_azim = self.ax.azim
        self.ax.view_init(elev=self.ax.elev, azim=current_azim + delta_azim)
        self.redraw()
        
    def set_axis_limits(self, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), z_range=(-1.5, 1.5)):
        """Set custom axis limits"""
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.set_zlim(z_range)
        self.redraw()
        
    def reset_view(self):
        """Reset to default view"""
        self.ax.view_init(elev=30, azim=165)
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_box_aspect([1, 1, 1])
        # Keep mirroring solely in display mapping
        self.redraw()

class SimpleLandmarkDisplay:
    """Simplified landmark display for when matplotlib is not available"""
    
    def __init__(self, parent_frame, width=400, height=300):
        self.parent_frame = parent_frame
        self.width = width
        self.height = height
        
        # Create canvas
        self.canvas = tk.Canvas(parent_frame, width=width, height=height, 
                               bg='black', relief="solid", bd=1)
        self.canvas.pack(fill="both", expand=True)
        
        # Data
        self.live_landmarks = None
        self.template_landmarks = None
        
    def update_live_landmarks(self, landmarks):
        """Update live landmarks"""
        if landmarks and len(landmarks) == 12:
            self.live_landmarks = landmarks
            self.redraw()
            
    def set_template_landmarks(self, landmarks):
        """Set template landmarks"""
        if landmarks and len(landmarks) == 12:
            self.template_landmarks = landmarks
            self.redraw()
            
    def redraw(self):
        """Redraw the display"""
        self.canvas.delete("all")
        
        # Draw title
        self.canvas.create_text(self.width//2, 20, text="Landmark Visualization", 
                               fill="white", font=("Arial", 12, "bold"))
        
        if self.live_landmarks:
            self.draw_landmarks_2d(self.live_landmarks, "blue", "Live")
            
        if self.template_landmarks:
            self.draw_landmarks_2d(self.template_landmarks, "red", "Template")
            
    def draw_landmarks_2d(self, landmarks, color, label):
        """Draw landmarks in 2D projection"""
        # Simple 2D projection (X-Y plane)
        scale = min(self.width, self.height) * 0.3
        center_x, center_y = self.width//2, self.height//2
        
        # Draw label
        y_offset = 40 if label == "Live" else 60
        self.canvas.create_text(50, y_offset, text=f"{label}:", 
                               fill=color, font=("Arial", 10, "bold"))
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks):
            screen_x = center_x + x * scale
            screen_y = center_y + y * scale
            
            # Draw point
            self.canvas.create_oval(screen_x-3, screen_y-3, screen_x+3, screen_y+3, 
                                   fill=color, outline="white")
            
            # Draw landmark number
            self.canvas.create_text(screen_x+8, screen_y, text=str(i), 
                                   fill="white", font=("Arial", 8))
                                   
    def clear_template(self):
        """Clear template landmarks"""
        self.template_landmarks = None
        self.redraw()

def create_visualizer(parent_frame, use_matplotlib=True):
    """Factory function to create appropriate visualizer"""
    try:
        if use_matplotlib:
            return LandmarkVisualizer(parent_frame)
        else:
            return SimpleLandmarkDisplay(parent_frame)
    except ImportError:
        print("Matplotlib not available, using simple display")
        return SimpleLandmarkDisplay(parent_frame)
