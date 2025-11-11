"""
Unified Application - Face Recognition + Object Detection with Person ReID.
"""
import cv2
from detection import FaceDetector, ObjectDetector
from recognition import FaceEmbedder, FaceTracker
from storage import FaceDatabase
from ui import UIRenderer, InputHandler
from tracking import ObjectTracker, PersonTracker


class UnifiedApp:
    """Unified application with face recognition and object detection."""
    
    def __init__(self, enable_face_recognition=True, enable_object_detection=True):
        """
        Initialize the unified application.
        
        Args:
            enable_face_recognition: Enable face recognition
            enable_object_detection: Enable object detection
        """
        print("=" * 60)
        print("Unified Detection System")
        print("=" * 60)
        
        self.enable_face_recognition = enable_face_recognition
        self.enable_object_detection = enable_object_detection
        
        # Initialize face recognition modules
        if enable_face_recognition:
            self.face_detector = FaceDetector(confidence_threshold=0.5)
            self.embedder = FaceEmbedder()
            self.database = FaceDatabase()
            self.tracker = FaceTracker(samples_needed=200)
            self.input_handler = InputHandler()
        
        # Initialize object detection
        if enable_object_detection:
            self.object_detector = ObjectDetector(confidence_threshold=0.4)
            self.object_tracker = ObjectTracker(max_disappeared=20)
        
        # Initialize person tracker (ReID)
        if enable_face_recognition:
            self.person_tracker = PersonTracker(max_disappeared=50)
        
        self.ui = UIRenderer()
        
        print("=" * 60)
        print("System ready!")
        print("=" * 60)
    
    def process_frame(self, frame):
        """
        Process a single frame with Person ReID.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Tuple of (processed frame, face results, object results)
        """
        face_results = []
        object_results = []
        person_bodies = []
        
        # Step 1: Detect person bodies first (for ReID)
        if self.enable_face_recognition and self.enable_object_detection:
            all_objects = self.object_detector.detect(frame)
            
            # Extract person detections
            for x1, y1, x2, y2, class_name, confidence in all_objects:
                if class_name == 'person':
                    person_bodies.append((x1, y1, x2, y2))
            
            # Update person tracker
            tracked_persons = self.person_tracker.update(person_bodies)
        
        # Step 2: Face recognition with ReID
        if self.enable_face_recognition:
            faces = self.face_detector.detect(frame)
            
            for x1, y1, x2, y2, confidence in faces:
                face_img = frame[y1:y2, x1:x2]
                face_box = (x1, y1, x2, y2)
                
                if face_img.size == 0:
                    continue
                
                embedding = self.embedder.extract(face_img)
                name, distance = self.database.find_match(embedding)
                
                # Find which person body this face belongs to
                person_id = None
                if self.enable_object_detection and person_bodies:
                    for pid, (centroid, person_box) in tracked_persons.items():
                        if self.person_tracker.is_face_inside_person(face_box, person_box):
                            person_id = pid
                            break
                
                if name:
                    # Known face - associate with person
                    if person_id is not None:
                        self.person_tracker.associate_face_to_person(name, person_id)
                    
                    face_results.append({
                        'box': (x1, y1, x2, y2),
                        'name': name,
                        'status': 'known',
                        'type': 'face',
                        'person_id': person_id
                    })
                else:
                    # Unknown face - check if person has known identity
                    if person_id is not None:
                        known_face = self.person_tracker.get_face_for_person(person_id)
                        if known_face:
                            # Use person's known identity even though face not directly recognized
                            face_results.append({
                                'box': (x1, y1, x2, y2),
                                'name': known_face,
                                'status': 'known_via_reid',
                                'type': 'face',
                                'person_id': person_id
                            })
                            continue
                    
                    # Truly unknown - collect samples
                    face_id, sample_count, is_ready = self.tracker.track(embedding)
                    face_results.append({
                        'box': (x1, y1, x2, y2),
                        'name': None,
                        'status': 'ready' if is_ready else 'collecting',
                        'face_id': face_id,
                        'sample_count': sample_count,
                        'type': 'face',
                        'person_id': person_id
                    })
        
        # Object detection with tracking
        if self.enable_object_detection:
            objects = self.object_detector.detect(frame)
            
            # Filter out person class if face recognition is enabled
            filtered_objects = []
            for x1, y1, x2, y2, class_name, confidence in objects:
                if class_name == 'person' and self.enable_face_recognition:
                    continue
                filtered_objects.append((x1, y1, x2, y2, class_name, confidence))
            
            # Update tracker with detections
            tracked_objects = self.object_tracker.update(filtered_objects)
            
            # Build results with tracking IDs
            for object_id, (centroid, class_name, confidence) in tracked_objects.items():
                # Find corresponding box
                for x1, y1, x2, y2, obj_class, obj_conf in filtered_objects:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Match by centroid proximity
                    if abs(cx - centroid[0]) < 20 and abs(cy - centroid[1]) < 20:
                        object_results.append({
                            'box': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': confidence,
                            'type': 'object',
                            'id': object_id
                        })
                        break
        
        # Draw results
        output = self._draw_results(frame, face_results, object_results)
        
        return output, face_results, object_results
    
    def _draw_results(self, frame, face_results, object_results):
        """Draw all detection results on frame."""
        output = frame.copy()
        
        # Assign numbers to ready faces and draw
        ready_face_count = 0
        for idx, result in enumerate(face_results):
            x1, y1, x2, y2 = result['box']
            
            if result['status'] == 'known':
                color = (0, 255, 0)
                person_id = result.get('person_id')
                label = f"{result['name']}" + (f" (P{person_id})" if person_id is not None else "")
            elif result['status'] == 'known_via_reid':
                color = (0, 200, 200)  # Cyan for ReID
                person_id = result.get('person_id')
                label = f"{result['name']} [ReID]"
            elif result['status'] == 'ready':
                ready_face_count += 1
                color = (0, 255, 255)
                label = f"[{ready_face_count}] Ready - Press {ready_face_count}"
                # Store selection number in result for input handler
                result['selection_number'] = ready_face_count
            else:
                color = (0, 0, 255)
                sample_count = result['sample_count']
                label = f"Collecting {sample_count}/{self.tracker.samples_needed}"
            
            self.ui.draw_face_box(output, x1, y1, x2, y2, label, color)
        
        # Draw objects with tracking IDs
        for result in object_results:
            x1, y1, x2, y2 = result['box']
            class_name = result['class']
            confidence = result['confidence']
            object_id = result.get('id', -1)
            
            color = self.object_detector.get_color(class_name)
            label = f"ID:{object_id} {class_name}: {confidence:.2f}"
            
            self.ui.draw_object_box(output, x1, y1, x2, y2, label, color)
        
        # Draw input box if in input mode
        if self.enable_face_recognition and self.input_handler.is_in_input_mode():
            selected_num = self.input_handler.get_selected_number()
            self.ui.draw_input_box(output, self.input_handler.get_input_text(), selected_num)
        
        # Draw stats
        face_count = self.database.count() if self.enable_face_recognition else 0
        object_count = len(object_results)
        self.ui.draw_unified_stats(output, face_count, object_count)
        
        return output
    
    def run(self):
        """Run the unified application loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        window_name = 'Unified Detection - Press Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nStarting unified detection...")
        if self.enable_face_recognition and self.database.count() > 0:
            print(f"Known faces: {', '.join(self.database.get_known_names())}")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                output, face_results, object_results = self.process_frame(frame)
                
                # Show frame
                cv2.imshow(window_name, output)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if self.enable_face_recognition:
                    if not self.input_handler.handle_key(key, face_results, self.database, self.tracker):
                        break
                else:
                    if key == ord('q'):
                        break
                
                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nUnified detection stopped")
            if self.enable_face_recognition:
                print(f"Total known faces: {self.database.count()}")
