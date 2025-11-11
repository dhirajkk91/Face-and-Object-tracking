"""
Input Handler Module - Handles keyboard input and state.
"""


class InputHandler:
    """Manages keyboard input and input mode state."""
    
    def __init__(self):
        """Initialize input handler."""
        self.input_mode = False
        self.input_text = ""
        self.current_face_id = None
    
    def handle_key(self, key, results, database, tracker):
        """
        Handle keyboard input with face selection support.
        
        Args:
            key: Key code from cv2.waitKey()
            results: Current frame results
            database: FaceDatabase instance
            tracker: FaceTracker instance
            
        Returns:
            True to continue, False to quit
        """
        # Check if we should enter input mode via number key selection
        if not self.input_mode:
            # Check for number key press (1-9)
            if 49 <= key <= 57:  # Keys 1-9
                selection_num = key - 48  # Convert to 1-9
                
                # Find face with this selection number
                for result in results:
                    if result.get('status') == 'ready' and result.get('selection_number') == selection_num:
                        self.input_mode = True
                        self.current_face_id = result['face_id']
                        self.input_text = ""
                        print(f"Selected face #{selection_num} for naming")
                        break
        
        if self.input_mode:
            if key == 13:  # Enter
                if self.input_text.strip():
                    # Add the person
                    embeddings = tracker.get_embeddings(self.current_face_id)
                    database.add_person(self.input_text.strip(), embeddings)
                    
                    # Clear tracker
                    tracker.remove_face(self.current_face_id)
                    
                    self.input_mode = False
                    self.input_text = ""
                    self.current_face_id = None
            elif key == 27:  # Escape
                self.input_mode = False
                self.input_text = ""
                self.current_face_id = None
            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]
            elif 32 <= key <= 126:  # Printable characters
                self.input_text += chr(key)
        else:
            if key == ord('q'):
                return False
        
        return True
    
    def is_in_input_mode(self):
        """Check if currently in input mode."""
        return self.input_mode
    
    def get_input_text(self):
        """Get current input text."""
        return self.input_text
