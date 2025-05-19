import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import threading
import time
import pynput
from pynput.keyboard import Key, Controller
from screeninfo import get_monitors

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize keyboard controller
keyboard = Controller()

# Get screen dimensions
screen_width, screen_height = 0, 0
for m in get_monitors():
    screen_width = m.width
    screen_height = m.height
    break

# Variables for controlling cursor
prev_x, prev_y = 0, 0
smoothing = 8  # Smoothing factor for cursor movement
is_tracking = False
is_dragging = False
clicking = False
is_text_field = False
last_gesture = None
gesture_hold_frames = 0
required_hold_frames = 5  # How many frames a gesture must be held to activate

# Variables for voice recognition
voice_active = False
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Calibrate the recognizer for ambient noise
with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Define gesture detection functions
def detect_open_palm(landmarks):
    # Check if fingers are extended
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    wrist = landmarks[0]
    
    # Check vertical positions of fingertips relative to palm
    all_fingers_up = (
        thumb_tip.y < wrist.y and
        index_tip.y < wrist.y and
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y < wrist.y
    )
    
    # Check fingers are spread out (not in a fist)
    fingers_spread = (
        abs(thumb_tip.x - index_tip.x) > 0.04 and
        abs(index_tip.x - pinky_tip.x) > 0.1
    )
    
    return all_fingers_up and fingers_spread

def detect_pinch(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # Calculate distance between thumb tip and index tip
    distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                
    # Distance threshold for pinch detection
    return distance < 0.04

def detect_ok_sign(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    index_pip = landmarks[6]  # Lower joint of index finger
    
    # Calculate distance between thumb tip and index PIP
    distance = ((thumb_tip.x - index_pip.x) ** 2 + 
                (thumb_tip.y - index_pip.y) ** 2) ** 0.5
                
    # Check if thumb and index are forming a circle
    is_circle = distance < 0.05
    
    # Make sure other fingers are extended
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    other_fingers_extended = (
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y < wrist.y
    )
    
    return is_circle and other_fingers_extended

def detect_v_sign(landmarks):
    # Check if index and middle fingers are extended while others are closed
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Index and middle fingers up
    v_sign = (
        index_tip.y < wrist.y and
        middle_tip.y < wrist.y and
        ring_tip.y > wrist.y and
        pinky_tip.y > wrist.y
    )
    
    # Check if index and middle are spread apart
    fingers_spread = abs(index_tip.x - middle_tip.x) > 0.04
    
    return v_sign and fingers_spread

def detect_three_fingers(landmarks):
    # Check if index, middle, and ring fingers are extended while others are closed
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    thumb_tip = landmarks[4]
    wrist = landmarks[0]
    
    return (
        index_tip.y < wrist.y and
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y > wrist.y and
        thumb_tip.y > wrist.y
    )

def detect_all_finger_pinch(landmarks):
    # Check if all fingertips are close to thumb tip
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Calculate distances
    distances = [
        ((thumb_tip.x - finger_tip.x) ** 2 + (thumb_tip.y - finger_tip.y) ** 2) ** 0.5
        for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    ]
    
    # All distances should be small for a pinch with all fingers
    return all(d < 0.07 for d in distances)

def detect_fist(landmarks):
    # Check if all fingers are curled into a fist
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # In a fist, all fingertips are below middle palm position and close to palm
    all_curled = (
        thumb_tip.y > wrist.y and
        index_tip.y > wrist.y and
        middle_tip.y > wrist.y and
        ring_tip.y > wrist.y and
        pinky_tip.y > wrist.y
    )
    
    # Check fingers are not spread out (compact fist)
    compact = (
        abs(index_tip.x - pinky_tip.x) < 0.1
    )
    
    return all_curled and compact

def detect_thumb_up(landmarks):
    # Check if thumb is extended upward while other fingers are curled
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    thumb_up = thumb_tip.y < wrist.y - 0.1  # Thumb is significantly above wrist
    
    # Other fingers are curled
    other_fingers_curled = (
        index_tip.y > wrist.y and
        middle_tip.y > wrist.y and
        ring_tip.y > wrist.y and
        pinky_tip.y > wrist.y
    )
    
    return thumb_up and other_fingers_curled

def detect_thumb_down(landmarks):
    # Check if thumb is extended downward while other fingers are curled
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    thumb_down = thumb_tip.y > wrist.y + 0.1  # Thumb is significantly below wrist
    
    # Other fingers are curled
    other_fingers_curled = (
        index_tip.y > wrist.y and
        middle_tip.y > wrist.y and
        ring_tip.y > wrist.y and
        pinky_tip.y > wrist.y
    )
    
    return thumb_down and other_fingers_curled

def detect_rock_gesture(landmarks):
    # Rock gesture: pinky and thumb extended, others curled (🤘)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    return (
        thumb_tip.y < wrist.y and  # Thumb extended
        index_tip.y > wrist.y and  # Index curled
        middle_tip.y > wrist.y and  # Middle curled
        ring_tip.y > wrist.y and  # Ring curled
        pinky_tip.y < wrist.y  # Pinky extended
    )

def detect_flat_hand(landmarks):
    # Flat hand: all fingers extended and together
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    all_extended = (
        index_tip.y < wrist.y and
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y < wrist.y
    )
    
    # Fingers are close to each other (not spread)
    fingers_together = (
        abs(index_tip.x - middle_tip.x) < 0.03 and
        abs(middle_tip.x - ring_tip.x) < 0.03 and
        abs(ring_tip.x - pinky_tip.x) < 0.03
    )
    
    return all_extended and fingers_together

# Function for voice recognition
def voice_recognition():
    global voice_active, is_text_field
    
    while True:
        if voice_active and is_text_field:
            try:
                with mic as source:
                    print("Listening...")
                    audio = recognizer.listen(source, timeout=5)
                
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    
                    # Type the recognized text
                    for char in text:
                        keyboard.press(char)
                        keyboard.release(char)
                        time.sleep(0.01)
                    
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                
                # After processing voice, pause for a moment
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)
        else:
            time.sleep(0.5)

# Start voice recognition in a separate thread
voice_thread = threading.Thread(target=voice_recognition, daemon=True)
voice_thread.start()

# Gesture state tracking
last_swipe_x = 0
last_flat_hand_detection = 0

# Function to perform actions based on gestures
def perform_gesture_action(gesture, landmarks, frame_height, frame_width):
    global is_tracking, is_dragging, clicking, is_text_field, voice_active
    global prev_x, prev_y, last_swipe_x, last_flat_hand_detection
    
    if gesture == "open_palm":
        is_tracking = True
        is_dragging = False
        clicking = False
        return "Tracking mode activated"
    
    elif gesture == "pinch" and is_tracking:
        # Calculate average position of thumb and index finger for cursor position
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        x = (thumb_tip.x + index_tip.x) / 2
        y = (thumb_tip.y + index_tip.y) / 2
        
        # Convert to screen coordinates with vertical flip (camera is mirrored)
        screen_x = screen_width - (x * screen_width)
        screen_y = y * screen_height
        
        # Apply smoothing
        if prev_x > 0 and prev_y > 0:
            screen_x = prev_x + (screen_x - prev_x) / smoothing
            screen_y = prev_y + (screen_y - prev_y) / smoothing
        
        prev_x, prev_y = screen_x, screen_y
        
        # Move cursor
        pyautogui.moveTo(screen_x, screen_y)
        return "Moving cursor"
    
    elif gesture == "ok_sign" and is_tracking:
        if not clicking:
            pyautogui.click()
            clicking = True
            
            # Check if we're clicking on a text field
            # This is an approximation - in a real app you'd need to detect text fields
            # more accurately based on the UI elements
            active_window = pyautogui.getActiveWindow()
            if active_window and active_window.title in ["Notepad", "Word", "TextEdit", "Google Docs"]:
                is_text_field = True
                voice_active = True
                return "Clicked - Voice recognition activated"
            else:
                is_text_field = False
                voice_active = False
                return "Clicked"
        return "Click held"
    
    elif gesture == "v_sign" and is_tracking:
        pyautogui.rightClick()
        return "Right clicked"
    
    elif gesture == "three_fingers" and is_tracking:
        pyautogui.doubleClick()
        return "Double clicked"
    
    elif gesture == "all_finger_pinch" and is_tracking:
        if not is_dragging:
            pyautogui.mouseDown()
            is_dragging = True
            return "Started dragging"
        return "Dragging..."
    
    elif gesture == "fist" and is_tracking:
        # Fist gesture for scrolling
        index_knuckle = landmarks[5]
        
        # Determine scroll direction based on hand position change
        if prev_y > 0:
            scroll_amount = (index_knuckle.y * screen_height - prev_y) * 0.1
            pyautogui.scroll(-int(scroll_amount))
        
        prev_y = index_knuckle.y * screen_height
        return "Scrolling"
    
    elif gesture == "thumb_up" and is_tracking:
        # Copy
        pyautogui.hotkey('ctrl', 'c')
        return "Copy"
    
    elif gesture == "thumb_down" and is_tracking:
        # Paste
        pyautogui.hotkey('ctrl', 'v')
        return "Paste"
    
    elif gesture == "rock_gesture" and is_tracking:
        # Press Enter
        pyautogui.press('enter')
        return "Enter pressed"
    
    elif gesture == "flat_hand" and is_tracking:
        # Track horizontal movement for swipe gestures
        wrist = landmarks[0]
        current_x = wrist.x
        
        # Initialize if first detection
        if last_swipe_x == 0:
            last_swipe_x = current_x
            last_flat_hand_detection = time.time()
            return "Ready for swipe"
        
        time_since_last = time.time() - last_flat_hand_detection
        
        # Reset if too much time has passed
        if time_since_last > 1.0:
            last_swipe_x = current_x
            last_flat_hand_detection = time.time()
            return "Swipe reset"
        
        # Check for horizontal swipe
        if abs(current_x - last_swipe_x) > 0.15:
            # Left to right swipe (backspace)
            if current_x > last_swipe_x:
                pyautogui.press('backspace')
                result = "Backspace"
            # Right to left swipe (can add another action)
            else:
                pyautogui.press('delete')
                result = "Delete"
                
            # Reset after detecting swipe
            last_swipe_x = 0
            return result
        
        last_swipe_x = current_x
        last_flat_hand_detection = time.time()
        return "Preparing swipe"
    
    else:
        # No specific gesture or tracking not active
        if not is_tracking:
            return "Show open palm to activate"
        else:
            is_dragging = False
            clicking = False
            return "Waiting for gesture"

# Main loop for webcam processing
cap = cv2.VideoCapture(0)

# Set smaller resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture from webcam.")
        break
    
    # Flip the image horizontally for a more intuitive mirror effect
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe
    results = hands.process(image_rgb)
    
    # Get image dimensions
    frame_height, frame_width, _ = image.shape
    
    status_text = "Show open palm to activate"
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark positions
            landmarks = hand_landmarks.landmark
            
            # Detect gestures
            current_gesture = None
            
            if detect_open_palm(landmarks):
                current_gesture = "open_palm"
            elif detect_pinch(landmarks):
                current_gesture = "pinch"
            elif detect_ok_sign(landmarks):
                current_gesture = "ok_sign"
            elif detect_v_sign(landmarks):
                current_gesture = "v_sign"
            elif detect_three_fingers(landmarks):
                current_gesture = "three_fingers"
            elif detect_all_finger_pinch(landmarks):
                current_gesture = "all_finger_pinch"
            elif detect_fist(landmarks):
                current_gesture = "fist"
            elif detect_thumb_up(landmarks):
                current_gesture = "thumb_up"
            elif detect_thumb_down(landmarks):
                current_gesture = "thumb_down"
            elif detect_rock_gesture(landmarks):
                current_gesture = "rock_gesture"
            elif detect_flat_hand(landmarks):
                current_gesture = "flat_hand"
            
            # Handle gesture debouncing
            if current_gesture == last_gesture:
                gesture_hold_frames += 1
            else:
                gesture_hold_frames = 0
                
            last_gesture = current_gesture
            
            # Only perform action if gesture is held for enough frames
            if gesture_hold_frames >= required_hold_frames and current_gesture:
                status_text = perform_gesture_action(current_gesture, landmarks, frame_height, frame_width)
            elif is_dragging and current_gesture != "all_finger_pinch":
                # Release mouse if was dragging but no longer using drag gesture
                pyautogui.mouseUp()
                is_dragging = False
                status_text = "Drag ended"
    else:
        # No hands detected
        is_dragging = False
        clicking = False
    
    # Display status text
    cv2.putText(image, f"Status: {status_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display voice recognition status
    voice_status = "ON" if voice_active else "OFF"
    cv2.putText(image, f"Voice: {voice_status}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Hand Gesture Control', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
