import cv2
import mediapipe as mp
import math
import pyttsx3
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands object
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define gesture labels
GESTURES = {
    "call me": "Thumb and index finger extended",
    "loser": "Thumb and pinky extended",
    "high-five": "All fingers extended",
    "peace": "Index and middle fingers extended",
    "rock": "Thumb, index, and pinky extended",
    "ok": "Thumb and index finger forming a circle",
    "dislike": "Only pinky extended",
    "fist": "No fingers extended",
}

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)*2 + (landmark1.y - landmark2.y)*2)

# Function to check if a finger is extended
def is_finger_extended(hand_landmarks, finger_tip, finger_pip, finger_mcp):
    tip = hand_landmarks.landmark[finger_tip]
    pip = hand_landmarks.landmark[finger_pip]
    mcp = hand_landmarks.landmark[finger_mcp]
    return tip.y < pip.y < mcp.y  # Finger is extended if tip < pip < mcp

# Function to detect gestures
def detect_gesture(hand_landmarks):
    # Get the landmarks for the tips, pip, and mcp joints of all fingers
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    # Check which fingers are extended
    extended_fingers = [
        is_finger_extended(hand_landmarks, tip, pip, mcp)
        for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps)
    ]

    # Check for specific gestures
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Gesture: Call me (thumb and index finger extended)
    if extended_fingers[0] and extended_fingers[1] and not any(extended_fingers[2:]):
        return "call me"

    if extended_fingers[0] and extended_fingers[4] and not any(extended_fingers[1:4]):
        return "loser"

    if all(extended_fingers):
        return "high-five"

    if extended_fingers[1] and extended_fingers[2] and not any(extended_fingers[0:1] + extended_fingers[3:]):
        return "peace"

    if extended_fingers[0] and extended_fingers[1] and extended_fingers[4] and not any(extended_fingers[2:4]):
        return "rock"
    
    if calculate_distance(thumb_tip, index_tip) < 0.05:
        return "ok"
  
    if extended_fingers[4] and not any(extended_fingers[:4]):
        return "dislike"

    
    if not any(extended_fingers):
        return "fist"

    return None

def speak_gesture(gesture):
    engine.say(gesture)
    engine.runAndWait()

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            if gesture:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                threading.Thread(target=speak_gesture, args=(gesture,)).start()

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
