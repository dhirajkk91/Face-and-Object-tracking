"""
Person Tracker Module - Tracks people using body + face association.
Implements Person Re-identification (ReID).
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class PersonTracker:
    """
    Tracks people by associating face identities with body detections.
    Maintains identity even when face is not visible.
    """
    
    def __init__(self, max_disappeared=50):
        """
        Initialize person tracker.
        
        Args:
            max_disappeared: Maximum frames a person can disappear
        """
        self.next_person_id = 0
        self.persons = OrderedDict()  # {person_id: body_centroid}
        self.face_to_person = {}  # {face_name: person_id}
        self.person_to_face = {}  # {person_id: face_name}
        self.disappeared = OrderedDict()  # {person_id: frames_disappeared}
        self.person_boxes = OrderedDict()  # {person_id: (x1, y1, x2, y2)}
        self.max_disappeared = max_disappeared
    
    def register_person(self, centroid, box):
        """Register a new person."""
        person_id = self.next_person_id
        self.persons[person_id] = centroid
        self.person_boxes[person_id] = box
        self.disappeared[person_id] = 0
        self.next_person_id += 1
        return person_id
    
    def deregister_person(self, person_id):
        """Deregister a person."""
        if person_id in self.persons:
            del self.persons[person_id]
            del self.person_boxes[person_id]
            del self.disappeared[person_id]
            
            # Clean up face associations
            if person_id in self.person_to_face:
                face_name = self.person_to_face[person_id]
                if face_name in self.face_to_person:
                    del self.face_to_person[face_name]
                del self.person_to_face[person_id]
    
    def associate_face_to_person(self, face_name, person_id):
        """Associate a face identity with a person ID."""
        # Remove old associations
        if face_name in self.face_to_person:
            old_person_id = self.face_to_person[face_name]
            if old_person_id in self.person_to_face:
                del self.person_to_face[old_person_id]
        
        # Create new association
        self.face_to_person[face_name] = person_id
        self.person_to_face[person_id] = face_name
    
    def get_face_for_person(self, person_id):
        """Get the face name associated with a person ID."""
        return self.person_to_face.get(person_id, None)
    
    def get_person_for_face(self, face_name):
        """Get the person ID associated with a face name."""
        return self.face_to_person.get(face_name, None)
    
    def update(self, person_detections):
        """
        Update tracked persons with new detections.
        
        Args:
            person_detections: List of (x1, y1, x2, y2) person bounding boxes
            
        Returns:
            OrderedDict of {person_id: (centroid, box)}
        """
        # If no detections, mark all as disappeared
        if len(person_detections) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister_person(person_id)
            
            return self.get_tracked_persons()
        
        # Calculate centroids for new detections
        input_centroids = np.zeros((len(person_detections), 2), dtype="int")
        
        for i, (x1, y1, x2, y2) in enumerate(person_detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no persons are being tracked, register all
        if len(self.persons) == 0:
            for i in range(len(input_centroids)):
                self.register_person(input_centroids[i], person_detections[i])
        else:
            # Match existing persons to new detections
            person_ids = list(self.persons.keys())
            person_centroids = list(self.persons.values())
            
            # Calculate distance between each pair
            D = dist.cdist(np.array(person_centroids), input_centroids)
            
            # Find minimum distance for each person
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match persons to detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update person
                person_id = person_ids[row]
                self.persons[person_id] = input_centroids[col]
                self.person_boxes[person_id] = person_detections[col]
                self.disappeared[person_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared persons
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                person_id = person_ids[row]
                self.disappeared[person_id] += 1
                
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister_person(person_id)
            
            # Register new persons
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register_person(input_centroids[col], person_detections[col])
        
        return self.get_tracked_persons()
    
    def get_tracked_persons(self):
        """Get currently tracked persons with their info."""
        result = OrderedDict()
        for person_id in self.persons.keys():
            centroid = self.persons[person_id]
            box = self.person_boxes[person_id]
            result[person_id] = (centroid, box)
        return result
    
    def is_face_inside_person(self, face_box, person_box, iou_threshold=0.3):
        """
        Check if a face box is inside a person box.
        
        Args:
            face_box: (x1, y1, x2, y2) face bounding box
            person_box: (x1, y1, x2, y2) person bounding box
            iou_threshold: Minimum IoU to consider face inside person
            
        Returns:
            True if face is inside person box
        """
        fx1, fy1, fx2, fy2 = face_box
        px1, py1, px2, py2 = person_box
        
        # Calculate intersection
        ix1 = max(fx1, px1)
        iy1 = max(fy1, py1)
        ix2 = min(fx2, px2)
        iy2 = min(fy2, py2)
        
        if ix2 < ix1 or iy2 < iy1:
            return False
        
        # Calculate IoU
        intersection = (ix2 - ix1) * (iy2 - iy1)
        face_area = (fx2 - fx1) * (fy2 - fy1)
        
        if face_area == 0:
            return False
        
        overlap = intersection / face_area
        
        return overlap > iou_threshold
    
    def reset(self):
        """Reset all tracked persons."""
        self.persons.clear()
        self.face_to_person.clear()
        self.person_to_face.clear()
        self.disappeared.clear()
        self.person_boxes.clear()
        self.next_person_id = 0
