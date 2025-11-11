"""
Object Tracker Module - Tracks objects across frames.
Uses centroid tracking algorithm for smooth tracking.
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class ObjectTracker:
    """Tracks objects across frames using centroid tracking."""
    
    def __init__(self, max_disappeared=30):
        """
        Initialize object tracker.
        
        Args:
            max_disappeared: Maximum frames an object can disappear before removal
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {object_id: centroid}
        self.disappeared = OrderedDict()  # {object_id: frames_disappeared}
        self.object_info = OrderedDict()  # {object_id: (class_name, confidence)}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid, class_name, confidence):
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_info[self.next_object_id] = (class_name, confidence)
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_info[object_id]
    
    def update(self, detections):
        """
        Update tracked objects with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, class_name, confidence)
            
        Returns:
            OrderedDict of {object_id: (centroid, class_name, confidence, box)}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.get_tracked_objects()
        
        # Calculate centroids for new detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_boxes = []
        input_info = []
        
        for i, (x1, y1, x2, y2, class_name, confidence) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_boxes.append((x1, y1, x2, y2))
            input_info.append((class_name, confidence))
        
        # If no objects are being tracked, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_info[i][0], input_info[i][1])
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance between each pair
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance for each object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match objects to detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.object_info[object_id] = input_info[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], input_info[col][0], input_info[col][1])
        
        return self.get_tracked_objects()
    
    def get_tracked_objects(self):
        """Get currently tracked objects with their info."""
        result = OrderedDict()
        for object_id in self.objects.keys():
            centroid = self.objects[object_id]
            class_name, confidence = self.object_info[object_id]
            result[object_id] = (centroid, class_name, confidence)
        return result
    
    def reset(self):
        """Reset all tracked objects."""
        self.objects.clear()
        self.disappeared.clear()
        self.object_info.clear()
        self.next_object_id = 0
