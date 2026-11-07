# temporal_smoother.py
# Module for temporal smoothing of landmark data

import numpy as np
import collections
from typing import List, Tuple, Optional

class TemporalSmoother:
    """
    Temporal smoothing for hand landmarks to reduce jitter and improve recognition stability.
    
    Uses exponential moving average and buffer-based smoothing techniques.
    """
    
    def __init__(self, alpha: float = 0.2, buffer_size: int = 5, 
                 min_confidence: float = 0.1, max_velocity: float = 0.5):
        """
        Initialize the temporal smoother.
        
        Args:
            alpha: Smoothing factor (0.0 to 1.0). Lower = more smoothing
            buffer_size: Number of frames to keep in buffer
            min_confidence: Minimum confidence for landmark to be considered valid
            max_velocity: Maximum allowed velocity between frames (for outlier rejection)
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.buffer_size = max(1, buffer_size)
        self.min_confidence = min_confidence
        self.max_velocity = max_velocity
        
        # Buffers for smoothing
        self.landmark_buffer = collections.deque(maxlen=buffer_size)
        self.confidence_buffer = collections.deque(maxlen=buffer_size)
        
        # Previous smoothed values
        self.previous_smoothed = None
        self.previous_confidence = None
        
        # Statistics
        self.frame_count = 0
        self.smoothing_applied = 0
        
    def smooth_landmarks(self, landmarks: List[Tuple[float, float, float]], 
                        confidences: Optional[List[float]] = None) -> List[Tuple[float, float, float]]:
        """
        Apply temporal smoothing to landmark data.
        
        Args:
            landmarks: List of (x, y, z) landmark coordinates
            confidences: Optional list of confidence values for each landmark
            
        Returns:
            Smoothed landmark coordinates
        """
        if not landmarks or len(landmarks) != 12:
            return landmarks
            
        # Convert to numpy array for easier manipulation
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Handle confidences
        if confidences is None:
            confidences = [1.0] * len(landmarks)
        confidences_array = np.array(confidences, dtype=np.float32)
        
        # Add to buffer
        self.landmark_buffer.append(landmarks_array.copy())
        self.confidence_buffer.append(confidences_array.copy())
        
        # Apply smoothing based on buffer size
        if len(self.landmark_buffer) < 2:
            # Not enough data yet, return as-is
            self.previous_smoothed = landmarks_array.copy()
            self.previous_confidence = confidences_array.copy()
            return landmarks
            
        # Apply different smoothing strategies based on buffer size
        if len(self.landmark_buffer) == 2:
            # Simple exponential smoothing
            smoothed = self._exponential_smoothing(landmarks_array, confidences_array)
        else:
            # Weighted average with outlier rejection
            smoothed = self._weighted_average_smoothing()
            
        # Apply velocity-based outlier rejection
        smoothed = self._velocity_outlier_rejection(smoothed)
        
        # Update previous values
        self.previous_smoothed = smoothed.copy()
        self.previous_confidence = confidences_array.copy()
        
        self.frame_count += 1
        self.smoothing_applied += 1
        
        return smoothed.tolist()
        
    def _exponential_smoothing(self, current: np.ndarray, 
                             confidence: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        if self.previous_smoothed is None:
            return current
            
        # Weight by confidence
        weight = np.clip(confidence, 0.0, 1.0)
        alpha_weighted = self.alpha * weight
        
        # Apply smoothing
        smoothed = (1 - alpha_weighted) * self.previous_smoothed + alpha_weighted * current
        
        return smoothed
        
    def _weighted_average_smoothing(self) -> np.ndarray:
        """Apply weighted average smoothing using buffer."""
        landmarks_array = np.array(list(self.landmark_buffer))
        confidences_array = np.array(list(self.confidence_buffer))
        
        # Calculate weights (more recent frames have higher weight)
        weights = np.linspace(0.1, 1.0, len(self.landmark_buffer))
        weights = weights / np.sum(weights)  # Normalize
        
        # Apply confidence weighting
        confidence_weights = np.clip(confidences_array, self.min_confidence, 1.0)
        final_weights = weights[:, np.newaxis] * confidence_weights
        
        # Normalize weights for each landmark
        weight_sums = np.sum(final_weights, axis=0)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)  # Avoid division by zero
        final_weights = final_weights / weight_sums
        
        # Calculate weighted average
        smoothed = np.sum(landmarks_array * final_weights[:, :, np.newaxis], axis=0)
        
        return smoothed
        
    def _velocity_outlier_rejection(self, current: np.ndarray) -> np.ndarray:
        """Reject outliers based on velocity between frames."""
        if self.previous_smoothed is None:
            return current
            
        # Calculate velocity (distance between current and previous)
        velocity = np.linalg.norm(current - self.previous_smoothed, axis=1)
        
        # Find outliers
        outlier_mask = velocity > self.max_velocity
        
        if np.any(outlier_mask):
            # Replace outliers with previous values
            current[outlier_mask] = self.previous_smoothed[outlier_mask]
            
        return current
        
    def reset(self):
        """Reset the smoother state."""
        self.landmark_buffer.clear()
        self.confidence_buffer.clear()
        self.previous_smoothed = None
        self.previous_confidence = None
        self.frame_count = 0
        self.smoothing_applied = 0
        
    def get_statistics(self) -> dict:
        """Get smoothing statistics."""
        return {
            'frame_count': self.frame_count,
            'smoothing_applied': self.smoothing_applied,
            'buffer_size': len(self.landmark_buffer),
            'smoothing_ratio': self.smoothing_applied / max(1, self.frame_count)
        }
        
    def set_parameters(self, alpha: Optional[float] = None, 
                      buffer_size: Optional[int] = None,
                      min_confidence: Optional[float] = None,
                      max_velocity: Optional[float] = None):
        """Update smoothing parameters."""
        if alpha is not None:
            self.alpha = np.clip(alpha, 0.0, 1.0)
        if buffer_size is not None:
            self.buffer_size = max(1, buffer_size)
            # Adjust buffer if needed
            while len(self.landmark_buffer) > self.buffer_size:
                self.landmark_buffer.popleft()
                self.confidence_buffer.popleft()
        if min_confidence is not None:
            self.min_confidence = min_confidence
        if max_velocity is not None:
            self.max_velocity = max_velocity

class GestureStateTracker:
    """
    Track gesture state over time to prevent rapid switching and improve stability.
    """
    
    def __init__(self, min_duration: float = 0.3, confidence_threshold: float = 0.6):
        """
        Initialize gesture state tracker.
        
        Args:
            min_duration: Minimum time (seconds) a gesture must be detected before switching
            confidence_threshold: Minimum confidence for gesture to be considered valid
        """
        self.min_duration = min_duration
        self.confidence_threshold = confidence_threshold
        
        # Current state
        self.current_gesture = None
        self.current_confidence = 0.0
        self.start_time = None
        
        # History
        self.gesture_history = []
        
    def update(self, gesture_name: Optional[str], confidence: float, 
               timestamp: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Update gesture state and return stable gesture.
        
        Args:
            gesture_name: Detected gesture name (None if no match)
            confidence: Confidence score
            timestamp: Current timestamp (uses time.time() if None)
            
        Returns:
            Tuple of (stable_gesture_name, confidence)
        """
        import time
        if timestamp is None:
            timestamp = time.time()
            
        # Check if we have a valid gesture
        valid_gesture = (gesture_name is not None and 
                        confidence >= self.confidence_threshold)
        
        # Update state
        if valid_gesture and gesture_name == self.current_gesture:
            # Same gesture, update confidence
            self.current_confidence = max(self.current_confidence, confidence)
        elif valid_gesture and gesture_name != self.current_gesture:
            # New gesture detected
            if self.start_time is None or (timestamp - self.start_time) >= self.min_duration:
                # Enough time has passed, switch to new gesture
                self._switch_gesture(gesture_name, confidence, timestamp)
            # Otherwise, keep current gesture
        elif not valid_gesture:
            # No valid gesture, reset if enough time has passed
            if (self.start_time is not None and 
                (timestamp - self.start_time) >= self.min_duration * 2):
                self._reset_state()
                
        return self.current_gesture, self.current_confidence
        
    def _switch_gesture(self, gesture_name: str, confidence: float, timestamp: float):
        """Switch to a new gesture."""
        # Record previous gesture in history
        if self.current_gesture is not None:
            self.gesture_history.append({
                'gesture': self.current_gesture,
                'confidence': self.current_confidence,
                'duration': timestamp - self.start_time,
                'start_time': self.start_time,
                'end_time': timestamp
            })
            
        # Update current state
        self.current_gesture = gesture_name
        self.current_confidence = confidence
        self.start_time = timestamp
        
    def _reset_state(self):
        """Reset the current state."""
        self.current_gesture = None
        self.current_confidence = 0.0
        self.start_time = None
        
    def get_current_gesture(self) -> Tuple[Optional[str], float]:
        """Get current gesture and confidence."""
        return self.current_gesture, self.current_confidence
        
    def get_gesture_history(self) -> List[dict]:
        """Get gesture history."""
        return self.gesture_history.copy()
        
    def reset(self):
        """Reset the tracker."""
        self._reset_state()
        self.gesture_history.clear()

# Factory function for creating appropriate smoother
def create_smoother(smoothing_type: str = "exponential", **kwargs) -> TemporalSmoother:
    """
    Create a temporal smoother with specified type.
    
    Args:
        smoothing_type: Type of smoothing ("exponential", "weighted", "adaptive")
        **kwargs: Additional parameters for the smoother
        
    Returns:
        Configured TemporalSmoother instance
    """
    if smoothing_type == "exponential":
        return TemporalSmoother(alpha=kwargs.get('alpha', 0.2))
    elif smoothing_type == "weighted":
        return TemporalSmoother(alpha=0.3, buffer_size=kwargs.get('buffer_size', 5))
    elif smoothing_type == "adaptive":
        # Adaptive smoothing based on motion
        return TemporalSmoother(alpha=0.1, buffer_size=7, max_velocity=0.3)
    else:
        raise ValueError(f"Unknown smoothing type: {smoothing_type}")
