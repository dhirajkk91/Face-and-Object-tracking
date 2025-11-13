"""
Face Embedder Module - Extracts face embeddings using DeepFace.
"""
import cv2
import numpy as np
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')


class FaceEmbedder:
    """Extracts face embeddings using DeepFace (VGG-Face model)."""
    
    def __init__(self):
        """Initialize face embedder with DeepFace."""
        print("✓ Using DeepFace for face recognition (VGG-Face model)")
        self.model_name = "VGG-Face"  # Accurate and reliable
    
    def extract(self, face_image):
        """
        Extract face embedding using DeepFace.
        
        Args:
            face_image: Cropped face image (BGR)
            
        Returns:
            Embedding vector (numpy array)
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Extract embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip'  # We already detected the face
            )
            
            # Get the embedding vector
            embedding = np.array(embedding_objs[0]["embedding"])
            return embedding
            
        except Exception as e:
            print(f"DeepFace extraction failed: {e}")
            # Return a zero vector if extraction fails
            return np.zeros(2622)  # VGG-Face embedding size
    
    @staticmethod
    def compare(embedding1, embedding2):
        """
        Compare two embeddings using cosine distance.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (0 = identical, higher = more different)
        """
        # Cosine distance
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_similarity
        
        return cosine_distance
