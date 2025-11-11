"""
Main entry point for Detection System.
"""
import sys
sys.path.insert(0, 'src')

from core import FaceRecognitionApp, UnifiedApp


def main():
    """Main entry point."""
    print("=" * 60)
    print("Detection System")
    print("=" * 60)
    print("1. Face Recognition Only")
    print("2. Object Detection Only")
    print("3. Unified (Face + Object Detection)")
    print("=" * 60)
    
    choice = input("Enter choice (1-3): ").strip()
    
    try:
        if choice == "1":
            app = FaceRecognitionApp()
            app.run()
        elif choice == "2":
            app = UnifiedApp(enable_face_recognition=False, enable_object_detection=True)
            app.run()
        elif choice == "3":
            app = UnifiedApp(enable_face_recognition=True, enable_object_detection=True)
            app.run()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
