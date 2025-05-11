import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import sqrt

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

screen_width, screen_height = pyautogui.size()

prev_x, prev_y = 0, 0
smoothing = 0.3

prev_pinch_y = 0
is_pinching = False

def get_distance(point1, point2):
    return sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract key landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Use ring finger to move cursor
        ring_x = int(ring_tip.x * frame_width)
        ring_y = int(ring_tip.y * frame_height)
        screen_x = np.interp(ring_x, [0, frame_width], [0, screen_width])
        screen_y = np.interp(ring_y, [0, frame_height], [0, screen_height])

        # Apply smoothing to cursor movement
        smoothed_x = prev_x + (screen_x - prev_x) * smoothing
        smoothed_y = prev_y + (screen_y - prev_y) * smoothing
        pyautogui.moveTo(smoothed_x, smoothed_y)
        prev_x, prev_y = smoothed_x, smoothed_y

        # Pinch detection for clicks (thumb-index for left, thumb-middle for right)
        thumb_index_dist = get_distance(thumb_tip, index_tip)
        thumb_middle_dist = get_distance(thumb_tip, middle_tip)

        if thumb_index_dist < 0.05:  # Thumb-index pinch for left click
            pyautogui.click()
            pyautogui.sleep(0.2)  # Debounce to prevent multiple clicks

        if thumb_middle_dist < 0.05:  # Thumb-middle pinch for right click
            pyautogui.click(button='right')
            pyautogui.sleep(0.2)  # Debounce

        # Detect index-middle pinch for scrolling
        index_middle_dist = get_distance(index_tip, middle_tip)
        
        pinch_y = (index_tip.y + middle_tip.y) / 2 * frame_height

        if index_middle_dist < 0.05:  # Index-middle pinch detected
            if not is_pinching:  # First frame of pinch detection
                prev_pinch_y = pinch_y
                is_pinching = True
            delta_y = prev_pinch_y - pinch_y
            if abs(delta_y) > 5:  # Lowered threshold for more responsiveness
                scroll_amount = int(delta_y * 2.0)  # Increased sensitivity (was 1.5, now 2.0)
                pyautogui.scroll(scroll_amount)
                pyautogui.sleep(0.05)  # Reduced sleep for faster scrolling
            prev_pinch_y = pinch_y
        else:
            is_pinching = False  # Stop scrolling when fingers separate

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()



# Actions

# pinch thumb and index finger for left click
# pinch thumb and middle finger for right click
# bring togeather the index and middle finger and move up or down to scroll
