"""
Face Recognizer V2 - Using face_recognition library for accurate recognition.
Much more reliable than custom embeddings.
"""
import face_recognition
import numpy as np
import pickle
import os


class FaceRecognizerV2:
    """
    Accurate face recognition using face_recognition library.
    This uses dlib's state-of-the-art face recognition built with deep learning.
    """
    
    def __init__(self, database_file="face_database_v2.pkl"):
        """Initialize face recognizer."""
        self.database_file = database_file
        self.known_encodings = []
        self.known_names = []
        self._load_database()
    
    def _load_database(self):
        """Load existing face database."""
        if os.path.exists(self.database_file):
            print(f"Loading face database from {self.database_file}...")
            with open(self.database_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            print(f"✓ Loaded {len(self.known_names)} known faces")
        else:
            print("No existing database - starting fresh")
    
    def save_database(self):
        """Save face database."""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        with open(self.database_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_person(self, name, face_encodings_list):
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            face_encodings_list: List of face encodings (128D vectors)
        """
        # Average the encodings for robustness
        avg_encoding = np.mean(face_encodings_list, axis=0)
        
        self.known_encodings.append(avg_encoding)
        self.known_names.append(name)
        self.save_database()
        
        print(f"✓ Added {name} with {len(face_encodings_list)} samples")
    
    def recognize_faces(self, frame):
        """
        Recognize all faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of tuples: (name, face_location, distance)
            face_location is (top, right, bottom, left)
        """
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"
            distance = 1.0
            
            if len(self.known_encodings) > 0:
                # Compare with all known faces
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                min_distance = np.min(distances)
                
                # Use strict threshold
                if min_distance < 0.4:  # 0.4 is good threshold for face_recognition
                    best_match_idx = np.argmin(distances)
                    name = self.known_names[best_match_idx]
                    distance = min_distance
            
            results.append((name, face_location, distance))
        
        return results
    
    def get_known_names(self):
        """Get list of all known names."""
        return self.known_names.copy()
    
    def count(self):
        """Get number of known faces."""
        return len(self.known_names)
