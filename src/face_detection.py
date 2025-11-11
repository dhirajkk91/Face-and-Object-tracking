"""
Face detection using OpenCV's DNN module with Caffe model.
"""
import cv2
import os
import urllib.request


class FaceDetector:
    def __init__(self):
        """Initialize the DNN face detector."""
        self.net = self._load_dnn_model()
    
    def _load_dnn_model(self):
        """Load the DNN face detection model."""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Download model files if they don't exist
        if not os.path.exists(prototxt_path):
            print("Downloading DNN model files...")
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        if not os.path.exists(model_path):
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            print("Downloading model weights (this may take a moment)...")
            urllib.request.urlretrieve(model_url, model_path)
        
        # Load the model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("DNN model loaded successfully")
        return net
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """
        Detect faces in an image using DNN.
        
        Args:
            image: BGR image (numpy array)
            confidence_threshold: Minimum confidence (0.0 to 1.0)
            
        Returns:
            List of face locations as (x, y, w, h, confidence) tuples
        """
        h, w = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Convert to (x, y, w, h, confidence) format
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return faces
    
    def draw_faces(self, image, faces):
        """
        Draw rectangles around detected faces.
        
        Args:
            image: BGR image (numpy array)
            faces: List of face locations as (x, y, w, h, confidence) tuples
            
        Returns:
            Image with rectangles drawn around faces
        """
        output = image.copy()
        
        for x, y, w, h, confidence in faces:
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f'Face: {confidence:.2f}'
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output
    
    def detect_from_webcam(self):
        """
        Detect faces from webcam feed in real-time.
        Close the window or press 'q' to quit.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        window_name = 'Face Detection - Press Q or close window to exit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("Starting webcam...")
        print("Press 'q' or close the window to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles
            output = self.draw_faces(frame, faces)
            
            # Display face count
            cv2.putText(output, f'Faces: {len(faces)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow(window_name, output)
            
            # Check if window was closed or 'q' was pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check if window was closed by clicking X
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")


if __name__ == "__main__":
    print("Face Detection with DNN")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = FaceDetector()
        
        # Start webcam detection
        detector.detect_from_webcam()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Failed: {e}")
