"""
Face Detector Module - Handles face detection using DNN.
"""
import cv2
import os
import urllib.request


class FaceDetector:
    """Detects faces in images using OpenCV DNN."""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection (0.0 to 1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.detector = self._load_model()
    
    def _load_model(self):
        """Load DNN face detection model."""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Download if needed
        if not os.path.exists(prototxt_path):
            print("Downloading face detector config...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(url, prototxt_path)
        
        if not os.path.exists(model_path):
            print("Downloading face detector weights...")
            url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            urllib.request.urlretrieve(url, model_path)
        
        detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("✓ Face detector loaded")
        return detector
    
    def detect(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                faces.append((x1, y1, x2, y2, confidence))
        
        return faces
