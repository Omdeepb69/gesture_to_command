##How It Works
The system I've created combines hand gesture tracking with voice recognition to provide a complete hands-free computer control experience:
Gesture Controls

Open Palm: Activate tracking mode (gateway gesture)
Pinch (index & thumb): Move cursor precisely
OK Sign: Left click (also activates voice recognition in text fields)
V Sign (peace sign): Right click
Three Fingers Up: Double-click
All Fingers Pinched: Drag and drop
Closed Fist: Scroll (movement direction controls scroll direction)
Thumb Up: Copy
Thumb Down: Paste
Rock Gesture (pinky & thumb out): Press Enter
Flat Hand Swipe: Backspace (left-to-right) or Delete (right-to-left)

Voice Recognition

Automatically activates when clicking on a text field
Converts your speech to text input
Works with standard text applications

Technical Details

Uses MediaPipe for accurate hand landmark detection
Speech recognition with Google's speech-to-text API
Smoothed cursor control for precision
Gesture debouncing to prevent accidental triggers
Visual feedback through webcam display

How to Use It

Setup:

Install required libraries: pip install opencv-python mediapipe pyautogui numpy speech_recognition pynput screeninfo
Run the Python script


Basic Usage:

Show an open palm to activate tracking
Use pinch gesture to move the cursor
Make OK sign to click
The system will show status feedback on screen


When Working with Text:

Click on any text field using the OK gesture
Voice recognition will automatically activate
Speak clearly to input text
Use rock gesture (🤘) to press Enter when done



Potential Enhancements
If you'd like to extend this system, consider:

Adding custom voice commands for application control
Creating a settings panel to adjust sensitivity
Adding more gesture combinations for additional functions
Implementing eye tracking for extra precision
Adding support for multi-monitor setups
