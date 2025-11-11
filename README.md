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
│   ├── main.py                 # Entry point (run this!)
│   ├── core/                   # Core application logic
│   │   ├── app.py
│   │   └── unified_app.py
│   ├── detection/              # Detection modules
│   │   ├── face_detector.py
│   │   └── object_detector.py
│   ├── recognition/            # Recognition modules
│   │   ├── face_embedder.py
│   │   └── face_tracker.py
│   ├── tracking/               # Object tracking
│   │   └── object_tracker.py
│   ├── storage/                # Data persistence
│   │   └── face_database.py
│   └── ui/                     # User interface
│       ├── renderer.py
│       └── input_handler.py
├── models/                     # Downloaded DNN models
├── face_encodings_advanced.pkl # Face database (auto-created)
├── requirements.txt
└── README.md
```

## Usage

### Run Face Recognition System
```bash
python src/main.py
```

**Features:**
- ✓ Clean OOP architecture with modular design
- ✓ Face Recognition with automatic training
- ✓ Object Detection using YOLO
- ✓ Unified mode (Face + Object detection)
- ✓ In-window name input prompt
- ✓ Deep learning embeddings
- ✓ Real-time detection and tracking
- ✓ Persistent face database

**How it works:**
1. Run `python src/app.py`
2. System automatically collects face samples (shows "Collecting X/10")
3. When 10 samples collected, shows "Press ENTER to name"
4. Press ENTER and type name directly in the window
5. Press ENTER again to save
6. Person is immediately recognized!

**Architecture:**
- `core/` - Main application logic
- `detection/` - Face and object detection
- `recognition/` - Face embeddings and tracking
- `storage/` - Database management
- `ui/` - User interface components
