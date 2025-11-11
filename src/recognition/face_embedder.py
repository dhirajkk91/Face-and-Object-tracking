"""
Face Embedder Module - Extracts face embeddings for recognition.
"""
import cv2
import numpy as np
import os
import urllib.request


class FaceEmbedder:
    """Extracts face embeddings using deep learning or enhanced features."""
    
    def __init__(self):
        """Initialize face embedder."""
        self.model = self._load_model()
    
    def _load_model(self):
        """Load face recognition model."""
        model_dir = "models"
        face_rec_model = os.path.join(model_dir, "openface_nn4.small2.v1.t7")
        
        if not os.path.exists(face_rec_model):
            print("Downloading face recognition model...")
            url = "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7"
            try:
                urllib.request.urlretrieve(url, face_rec_model)
                print("✓ Face recognition model downloaded")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Using enhanced fallback feature extraction")
                return None
        
        try:
            model = cv2.dnn.readNetFromTorch(face_rec_model)
            print("✓ Face recognition model loaded")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using enhanced fallback feature extraction")
            return None
    
    def extract(self, face_image):
        """
        Extract face embedding.
        
        Args:
            face_image: Cropped face image (BGR)
            
        Returns:
            Embedding vector (numpy array)
        """
        if self.model is not None:
            return self._extract_with_model(face_image)
        else:
            return self._extract_fallback(face_image)
    
    def _extract_with_model(self, face_image):
        """Extract embedding using DNN model."""
        face_blob = cv2.dnn.blobFromImage(
            face_image, 
            1.0 / 255, 
            (96, 96), 
            (0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        self.model.setInput(face_blob)
        embedding = self.model.forward()
        return embedding.flatten()
    
    def _extract_fallback(self, face_image):
        """Extract features using enhanced fallback method."""
        face_resized = cv2.resize(face_image, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        # Divide face into regions (eyes, nose, mouth)
        top_region = gray[0:h//3, :]
        middle_region = gray[h//3:2*h//3, :]
        bottom_region = gray[2*h//3:, :]
        
        # Normalize each region
        top_norm = top_region.astype('float32') / 255.0
        middle_norm = middle_region.astype('float32') / 255.0
        bottom_norm = bottom_region.astype('float32') / 255.0
        
        # Compute histograms
        top_hist = cv2.calcHist([top_region], [0], None, [32], [0, 256]).flatten()
        middle_hist = cv2.calcHist([middle_region], [0], None, [32], [0, 256]).flatten()
        bottom_hist = cv2.calcHist([bottom_region], [0], None, [32], [0, 256]).flatten()
        
        # Normalize histograms
        top_hist = top_hist / (top_hist.sum() + 1e-7)
        middle_hist = middle_hist / (middle_hist.sum() + 1e-7)
        bottom_hist = bottom_hist / (bottom_hist.sum() + 1e-7)
        
        # Combine features
        features = np.concatenate([
            top_norm.flatten()[::4],
            middle_norm.flatten()[::4],
            bottom_norm.flatten()[::4],
            top_hist,
            middle_hist,
            bottom_hist
        ])
        
        return features
    
    @staticmethod
    def compare(embedding1, embedding2):
        """
        Compare two embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (0 = identical, 1 = completely different)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        similarity = dot_product / (norm_product + 1e-7)
        distance = 1 - similarity
        return distance
