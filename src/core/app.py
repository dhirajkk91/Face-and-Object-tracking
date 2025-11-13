"""
Face Recognition Application - Main entry point.
Unified app that brings all modules together.
"""
import cv2
from detection import FaceDetector
from recognition import FaceEmbedder, FaceTracker
from storage import FaceDatabase
from ui import UIRenderer, InputHandler


class FaceRecognitionApp:
    """Main application class for face recognition system."""
    
    def __init__(self):
        """Initialize the application."""
        print("=" * 60)
        print("Face Recognition System")
        print("=" * 60)
        
        # Initialize modules
        self.detector = FaceDetector(confidence_threshold=0.5)
        self.embedder = FaceEmbedder()
        self.database = FaceDatabase()
        self.tracker = FaceTracker(samples_needed=20)
        self.ui = UIRenderer()
        self.input_handler = InputHandler()
        
        print("=" * 60)
        print("System ready!")
        print("=" * 60)
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Processed frame with annotations
        """
        # Detect faces
        faces = self.detector.detect(frame)
        
        results = []
        
        for x1, y1, x2, y2, confidence in faces:
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # Extract embedding
            embedding = self.embedder.extract(face_img)
            
            # Try to recognize
            name, distance = self.database.find_match(embedding)
            
            if name:
                # Known face
                results.append({
                    'box': (x1, y1, x2, y2),
                    'name': name,
                    'status': 'known',
                    'confidence': confidence
                })
            else:
                # Unknown face - track it
                face_id, sample_count, is_ready = self.tracker.track(embedding)
                
                results.append({
                    'box': (x1, y1, x2, y2),
                    'name': None,
                    'status': 'ready' if is_ready else 'collecting',
                    'face_id': face_id,
                    'sample_count': sample_count,
                    'confidence': confidence
                })
        
        # Draw results
        output = self._draw_results(frame, results)
        
        return output, results
    
    def _draw_results(self, frame, results):
        """Draw detection and recognition results on frame."""
        output = frame.copy()
        
        # Assign numbers to ready faces
        ready_face_count = 0
        for result in results:
            x1, y1, x2, y2 = result['box']
            
            # Determine color and label
            if result['status'] == 'known':
                color = (0, 255, 0)
                label = result['name']
            elif result['status'] == 'ready':
                ready_face_count += 1
                color = (0, 255, 255)
                label = f"[{ready_face_count}] Ready - Press {ready_face_count}"
                # Store selection number for input handler
                result['selection_number'] = ready_face_count
            else:
                color = (0, 0, 255)
                sample_count = result['sample_count']
                label = f"Collecting {sample_count}/{self.tracker.samples_needed}"
            
            # Draw face box
            self.ui.draw_face_box(output, x1, y1, x2, y2, label, color)
        
        # Draw input box if in input mode
        if self.input_handler.is_in_input_mode():
            selected_num = self.input_handler.get_selected_number()
            self.ui.draw_input_box(output, self.input_handler.get_input_text(), selected_num)
        
        # Draw stats
        self.ui.draw_stats(output, self.database.count())
        
        return output
    
    def run(self):
        """Run the main application loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        window_name = 'Face Recognition - Press Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nStarting face recognition...")
        if self.database.count() > 0:
            print(f"Known faces: {', '.join(self.database.get_known_names())}")
        else:
            print("No known faces - system will learn as you use it")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                output, results = self.process_frame(frame)
                
                # Show frame
                cv2.imshow(window_name, output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.input_handler.handle_key(key, results, self.database, self.tracker):
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nFace recognition stopped")
            print(f"Total known faces: {self.database.count()}")
