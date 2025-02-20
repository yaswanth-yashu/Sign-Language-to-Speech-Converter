# Sign-Language-to-Speech-Converter
This project is a real-time Sign Language to Speech Converter that uses OpenCV and MediaPipe to detect hand gestures and convert them into spoken words.

## Features
- **Real-Time Gesture Detection**: Detects hand gestures using MediaPipe's hand landmarks.
- **Multiple Gestures Supported**: Recognizes gestures like "call me", "loser", "high-five", "peace", "rock", "ok", "dislike", and "fist".
- **Text-to-Speech**: Converts detected gestures into spoken words using `pyttsx3`.
- **Smooth Video Feed**: Ensures real-time performance without hanging.
- **Easy to Extend**: Add new gestures by updating the gesture detection logic.

## How It Works
1. **Hand Detection**: MediaPipe's hand tracking model detects hand landmarks in real-time.
2. **Gesture Recognition**: The program checks the positions of the hand landmarks to identify specific gestures.
3. **Text-to-Speech**: Once a gesture is detected, the program uses `pyttsx3` to convert the gesture name into speech.
4. **Real-Time Feedback**: The detected gesture is displayed on the screen, and the corresponding word is spoken aloud.

## Adding New Gestures
To add a new gesture:
1. Update the `GESTURES` dictionary in `main.py` with the new gesture name and its corresponding finger extension pattern.
2. Add a new condition in the `detect_gesture` function to recognize the gesture based on hand landmarks.
3. Test the new gesture by running the program.
