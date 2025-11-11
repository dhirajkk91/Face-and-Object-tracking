"""
UI Renderer Module - Handles all visual rendering.
"""
import cv2


class UIRenderer:
    """Handles drawing UI elements on frames."""
    
    @staticmethod
    def draw_face_box(frame, x1, y1, x2, y2, label, color):
        """Draw a face bounding box with label."""
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 6, y2 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    @staticmethod
    def draw_input_box(frame, input_text):
        """Draw input box for entering name."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, h - 150), (w - 50, h - 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        cv2.putText(frame, "New face detected!", (70, h - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Enter name: {input_text}", (70, h - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press ESC to cancel", (70, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    @staticmethod
    def draw_stats(frame, known_count):
        """Draw statistics on frame."""
        cv2.putText(frame, f'Known: {known_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    @staticmethod
    def draw_object_box(frame, x1, y1, x2, y2, label, color):
        """Draw an object bounding box with label."""
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    @staticmethod
    def draw_unified_stats(frame, face_count, object_count):
        """Draw unified statistics on frame."""
        cv2.putText(frame, f'Faces: {face_count} | Objects: {object_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
