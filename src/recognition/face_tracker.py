"""
Face Tracker Module - Tracks unknown faces for training.
"""
import numpy as np
from collections import defaultdict


class FaceTracker:
    """Tracks unknown faces and collects samples for training."""
    
    def __init__(self, samples_needed=10, similarity_threshold=0.5):
        """
        Initialize face tracker.
        
        Args:
            samples_needed: Number of samples to collect before prompting
            similarity_threshold: Threshold for matching unknown faces (0.5 for DeepFace)
        """
        self.samples_needed = samples_needed
        self.similarity_threshold = similarity_threshold
        self.unknown_faces = defaultdict(list)
        self.face_id_counter = 0
    
    def track(self, embedding):
        """
        Track an unknown face embedding.
        
        Args:
            embedding: Face embedding
            
        Returns:
            Tuple of (face_id, sample_count, is_ready)
        """
        # Find if this embedding belongs to an existing unknown face
        face_id = self._find_face_id(embedding)
        
        if face_id is None:
            # New unknown face
            face_id = self.face_id_counter
            self.face_id_counter += 1
        
        # Add embedding to this face
        self.unknown_faces[face_id].append(embedding)
        
        sample_count = len(self.unknown_faces[face_id])
        is_ready = sample_count >= self.samples_needed
        
        return face_id, sample_count, is_ready
    
    def _find_face_id(self, embedding):
        """Find if embedding belongs to an existing unknown face."""
        from .face_embedder import FaceEmbedder
        
        for face_id, embeddings in self.unknown_faces.items():
            for known_embedding in embeddings:
                distance = FaceEmbedder.compare(embedding, known_embedding)
                if distance < self.similarity_threshold:
                    return face_id
        return None
    
    def get_embeddings(self, face_id):
        """Get all embeddings for a face ID."""
        return self.unknown_faces[face_id]
    
    def remove_face(self, face_id):
        """Remove a tracked face."""
        if face_id in self.unknown_faces:
            del self.unknown_faces[face_id]
