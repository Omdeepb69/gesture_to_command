import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import os
import subprocess
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import webbrowser
import keyboard
import win32gui
import win32con
from PIL import ImageGrab

class HandGestureControl:
    def __init__(self):
        # Initialize MediaPipe Hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Initialize audio controls
        self.setup_audio_controls()
        
        # Cooldown mechanism to prevent repeated triggering
        self.last_action_time = time.time()
        self.cooldown = 2  # seconds between actions
        
        # State tracking
        self.previous_gesture = None
        self.gesture_start_time = 0
        self.gesture_hold_time = 0.8  # seconds to hold gesture before triggering
        self.is_muted = False
        
        # Status message
        self.status_message = "Ready"
        self.status_time = time.time()
        
        # Command history
        self.command_history = []
        
        # Display size
        self.screen_width, self.screen_height = pyautogui.size()
    
    def setup_audio_controls(self):
        # Setup for volume control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        self.volume_range = (self.volume.GetVolumeRange()[0], self.volume.GetVolumeRange()[1])
        self.current_volume = self.volume.GetMasterVolumeLevelScalar()
    
    def detect_gesture(self, hand_landmarks):
        """Detect which gesture is being made based on hand landmarks"""
        
        # Extract landmark positions
        points = []
        for landmark in hand_landmarks.landmark:
            points.append((landmark.x, landmark.y, landmark.z))
        
        # Check for middle finger gesture (just middle finger up)
        if (points[12][1] > points[11][1] and  # Middle finger extended
            points[8][1] > points[7][1] and   # Index finger down
            points[16][1] > points[15][1] and  # Ring finger down
            points[20][1] > points[19][1] and  # Pinky down
            points[4][1] > points[3][1]):     # Thumb down
            return "middle_finger"
        
        # Check for thumbs up
        if (points[4][1] < points[3][1] and   # Thumb up
            points[8][1] > points[7][1] and   # Index finger down
            points[12][1] > points[11][1] and  # Middle finger down
            points[16][1] > points[15][1] and  # Ring finger down
            points[20][1] > points[19][1]):    # Pinky down
            return "thumbs_up"
        
        # Check for thumbs down
        if (points[4][1] > points[3][1] and   # Thumb down
            points[8][1] > points[7][1] and   # Index finger down
            points[12][1] > points[11][1] and  # Middle finger down
            points[16][1] > points[15][1] and  # Ring finger down
            points[20][1] > points[19][1] and  # Pinky down
            points[4][0] > points[0][0]):      # Thumb to the right
            return "thumbs_down"
        
        # Check for victory sign (index and middle fingers up)
        if (points[8][1] < points[7][1] and   # Index finger up
            points[12][1] < points[11][1] and  # Middle finger up
            points[16][1] > points[15][1] and  # Ring finger down
            points[20][1] > points[19][1]):    # Pinky down
            return "victory"
        
        # Check for open palm (all fingers extended)
        if (points[8][1] < points[5][1] and   # Index finger up
            points[12][1] < points[9][1] and  # Middle finger up
            points[16][1] < points[13][1] and  # Ring finger up
            points[20][1] < points[17][1]):    # Pinky up
            return "open_palm"
        
        # Check for fist (all fingers curled in)
        if (points[8][1] > points[5][1] and   # Index finger curled
            points[12][1] > points[9][1] and  # Middle finger curled
            points[16][1] > points[13][1] and  # Ring finger curled
            points[20][1] > points[17][1] and  # Pinky curled
            points[4][1] > points[2][1]):      # Thumb curled
            return "fist"
        
        # Check for pointing (just index finger up)
        if (points[8][1] < points[7][1] and   # Index finger up
            points[12][1] > points[11][1] and  # Middle finger down
            points[16][1] > points[15][1] and  # Ring finger down
            points[20][1] > points[19][1]):    # Pinky down
            return "pointing"
        
        # Check for OK sign (thumb and index finger forming circle)
        distance = np.sqrt(
            (points[4][0] - points[8][0])**2 + 
            (points[4][1] - points[8][1])**2
        )
        if distance < 0.1 and points[12][1] < points[11][1]:  # Proximity between thumb and index
            return "ok_sign"
        
        return "unknown"
    
    def execute_command(self, gesture):
        """Execute the command associated with the detected gesture"""
        
        current_time = time.time()
        # Only allow commands every few seconds to avoid rapid triggering
        if current_time - self.last_action_time < self.cooldown:
            return
        
        # Update status message and last action time
        self.last_action_time = current_time
        self.status_time = current_time
        
        try:
            if gesture == "middle_finger":
                self.status_message = "Shutting down in 10 seconds..."
                self.command_history.append("Shutdown initiated")
                # Use shutdown /s /t 10 to give 10 seconds warning before shutdown
                subprocess.run(["shutdown", "/s", "/t", "10"], shell=True)
            
            elif gesture == "thumbs_up":
                # Increase volume by 10%
                self.current_volume = min(1.0, self.current_volume + 0.1)
                self.volume.SetMasterVolumeLevelScalar(self.current_volume, None)
                volume_percent = int(self.current_volume * 100)
                self.status_message = f"Volume up: {volume_percent}%"
                self.command_history.append(f"Volume increased to {volume_percent}%")
            
            elif gesture == "thumbs_down":
                # Decrease volume by 10%
                self.current_volume = max(0.0, self.current_volume - 0.1)
                self.volume.SetMasterVolumeLevelScalar(self.current_volume, None)
                volume_percent = int(self.current_volume * 100)
                self.status_message = f"Volume down: {volume_percent}%"
                self.command_history.append(f"Volume decreased to {volume_percent}%")
            
            elif gesture == "victory":
                # Open default browser
                # Open Brave browser in incognito/private mode
                # webbrowser.open('http://hanime.tv')
                # self.status_message = "Opening web browser"
                # self.command_history.append("Web browser opened")
                url = 'http://hanime.tv'
                try:
                    # Try to open Brave in incognito mode
                    subprocess.Popen(['start', 'brave', '--incognito', url], shell=True)
                    self.status_message = "Opening Brave in incognito mode"
                    self.command_history.append("Brave browser (incognito) opened")
                except Exception as e:
                    self.status_message = f"Error opening Brave: {e}"
                    self.command_history.append("Failed to open Brave in incognito mode")
            
            elif gesture == "open_palm":
                # Show desktop (Windows key + D)
                pyautogui.hotkey('win', 'd')
                self.status_message = "Showing desktop"
                self.command_history.append("Desktop shown")
            
            elif gesture == "fist":
                # Toggle mute
                self.is_muted = not self.is_muted
                self.volume.SetMute(self.is_muted, None)
                status = "muted" if self.is_muted else "unmuted"
                self.status_message = f"Audio {status}"
                self.command_history.append(f"Audio {status}")
            
            elif gesture == "pointing":
                # Alt+Tab to switch windows
                pyautogui.hotkey('alt', 'tab')
                self.status_message = "Switching window"
                self.command_history.append("Window switched")
            
            elif gesture == "ok_sign":
                # Take screenshot
                screenshot = ImageGrab.grab()
                # Save to Pictures folder with timestamp
                pictures_dir = os.path.join(os.path.expanduser('~'), 'Pictures')
                if not os.path.exists(pictures_dir):
                    os.makedirs(pictures_dir)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = os.path.join(pictures_dir, f"screenshot_{timestamp}.png")
                screenshot.save(screenshot_path)
                self.status_message = f"Screenshot saved to {screenshot_path}"
                self.command_history.append("Screenshot taken")
            
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            print(f"Command execution error: {str(e)}")
    
    def run(self):
        """Main loop to capture video and detect gestures"""
        
        print("Hand Gesture Control System is active.")
        print("Available gestures:")
        print("- Middle Finger: Shutdown PC (10s warning)")
        print("- Thumbs Up: Increase volume")
        print("- Thumbs Down: Decrease volume")
        print("- Victory Sign: Open web browser")
        print("- Open Palm: Show desktop")
        print("- Fist: Toggle mute/unmute")
        print("- Pointing (index finger): Switch window")
        print("- OK Sign: Take screenshot")
        print("\nPress 'q' to quit")
        
        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Failed to capture image from camera")
                    break
                
                # Flip the image horizontally for a more intuitive mirror view
                image = cv2.flip(image, 1)
                
                # Convert the image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect hands
                results = self.hands.process(image_rgb)
                
                # Draw the hand annotations on the image
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Detect gesture
                        gesture = self.detect_gesture(hand_landmarks)
                        
                        # Display the detected gesture
                        cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Handle gesture persistence
                        current_time = time.time()
                        if gesture == self.previous_gesture and gesture != "unknown":
                            # If same gesture is held for enough time, execute command
                            if current_time - self.gesture_start_time >= self.gesture_hold_time:
                                self.execute_command(gesture)
                                self.gesture_start_time = current_time  # Reset timer after execution
                        else:
                            # New gesture detected, start timing
                            self.previous_gesture = gesture
                            self.gesture_start_time = current_time
                else:
                    # No hand detected
                    self.previous_gesture = None
                    cv2.putText(image, "No hand detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display status message with timeout
                if time.time() - self.status_time < 3:  # Show status for 3 seconds
                    cv2.putText(image, self.status_message, (10, image.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show most recent command in history
                if self.command_history:
                    cv2.putText(image, f"Last command: {self.command_history[-1]}", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Display the resulting image
                cv2.imshow('Hand Gesture Control', image)
                
                # Break the loop when 'q' is pressed
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        controller = HandGestureControl()
        controller.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")