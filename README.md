# Face & Object Detection System

A Python-based system for detecting and recognizing faces, as well as identifying objects in images and video streams.

## Features (In Development)
- Face detection
- Face recognition with name identification
- Object detection and classification

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── src/
│   ├── face_detection.py
│   ├── face_recognition.py
│   └── object_detection.py
├── known_faces/
├── requirements.txt
└── README.md
```

## Usage

### Phase 2: Basic Face Detection
```bash
python src/face_detection.py
```

Choose option 1 for webcam or option 2 for image file detection.
