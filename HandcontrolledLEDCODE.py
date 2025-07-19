#HandcontrolledLED
import cv2
import mediapipe as mp
import math
import serial
import time


arduino = serial.Serial('COM3', 9600) 
time.sleep(2)  

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Webcam input
cap = cv2.VideoCapture(0)

MIN_DIST = 30
MAX_DIST = 220  
def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        thumb_tip = lm.landmark[4]
        index_tip = lm.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        dist = get_distance((x1, y1), (x2, y2))

        # Normalize distance to brightness (0–255)
        norm = (dist - MIN_DIST) / (MAX_DIST - MIN_DIST)
        norm = max(0.0, min(norm, 1.0))  
        brightness = int(norm * 255)

        # Send brightness value to Arduino
        arduino.write(f"{brightness}\n".encode())
        print(f"Distance: {dist:.1f} → Brightness: {brightness}")

        # Draw hand landmarks and feedback
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"Brightness: {brightness}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show camera feed
    cv2.imshow("Finger Brightness Control", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break
# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()
