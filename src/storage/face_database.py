"""
Face Database Module - Manages known faces and their embeddings.
"""
import pickle
import numpy as np
import os


class FaceDatabase:
    """Manages storage and retrieval of face embeddings."""
    
    def __init__(self, database_file="face_encodings_advanced.pkl"):
        """
        Initialize face database.
        
        Args:
            database_file: Path to save/load face data
        """
        self.database_file = database_file
        self.embeddings = []
        self.names = []
        self._load()
    
    def _load(self):
        """Load existing face data."""
        if os.path.exists(self.database_file):
            print(f"Loading face database from {self.database_file}...")
            with open(self.database_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.names = data['names']
            print(f"✓ Loaded {len(self.names)} known faces")
        else:
            print("No existing database - starting fresh")
    
    def save(self):
        """Save face data to file."""
        data = {
            'embeddings': self.embeddings,
            'names': self.names
        }
        with open(self.database_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_person(self, name, embeddings_list):
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            embeddings_list: List of embeddings for this person
        """
        # Average the embeddings for robustness
        avg_embedding = np.mean(embeddings_list, axis=0)
        
        self.embeddings.append(avg_embedding)
        self.names.append(name)
        self.save()
        
        print(f"✓ Added {name} with {len(embeddings_list)} samples")
    
    def find_match(self, embedding, threshold=0.35):
        """
        Find matching person for an embedding.
        
        Args:
            embedding: Face embedding to match
            threshold: Maximum distance for a match (0.35 for strict matching)
            
        Returns:
            Tuple of (name, distance) or (None, distance) if no match
        """
        if not self.embeddings:
            return None, 1.0
        
        from recognition.face_embedder import FaceEmbedder
        
        # Calculate distances to all known faces
        distances = []
        for known_embedding in self.embeddings:
            distance = FaceEmbedder.compare(embedding, known_embedding)
            distances.append(distance)
        
        min_distance = min(distances)
        
        if min_distance < threshold:
            index = distances.index(min_distance)
            return self.names[index], min_distance
        
        return None, min_distance
    
    def get_known_names(self):
        """Get list of all known names."""
        return self.names.copy()
    
    def count(self):
        """Get number of known faces."""
        return len(self.names)
