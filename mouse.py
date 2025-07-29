# Final code logic as per user's request:
# - Index finger moves cursor
# - Thumb + Index = Click
# - Any 2 fingers (except thumb) = Scroll
# - All 5 fingers = Hold and drag

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# Initialize modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

wScr, hScr = pyautogui.size()

# Smoothing cursor movement
smoothening = 5
prev_loc = np.array([0, 0])
curr_loc = np.array([0, 0])

# Helper to count fingers
def count_fingers(lm_list):
    finger_states = []
    tip_ids = [4, 8, 12, 16, 20]
    for i in range(1, 5):  # Fingers (excluding thumb)
        if lm_list[tip_ids[i]].y < lm_list[tip_ids[i] - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    # Thumb (horizontal check)
    if lm_list[tip_ids[0]].x < lm_list[tip_ids[0] - 1].x:
        finger_states.insert(0, 1)
    else:
        finger_states.insert(0, 0)

    return finger_states

cap = cv2.VideoCapture(0)
click_cooldown = 1
last_click_time = 0
dragging = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    h, w, _ = img.shape

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        lm_list = [lm for lm in handLms.landmark]

        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        if len(lm_list) == 21:
            fingers = count_fingers(lm_list)
            total_fingers = sum(fingers)

            # Cursor move with index finger only
            if fingers == [0,1,0,0,0]:
                x = int(lm_list[8].x * wScr)
                y = int(lm_list[8].y * hScr)

                curr_loc = prev_loc + (np.array([x, y]) - prev_loc) / smoothening
                pyautogui.moveTo(curr_loc[0], curr_loc[1])
                prev_loc = curr_loc

            # Click with thumb touching index finger
            elif fingers[0] == 1 and fingers[1] == 1:
                # Distance between thumb and index
                x1, y1 = lm_list[4].x * w, lm_list[4].y * h
                x2, y2 = lm_list[8].x * w, lm_list[8].y * h
                dist = np.linalg.norm([x2 - x1, y2 - y1])

                if dist < 0.05 * w:
                    if time.time() - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = time.time()

            # Scroll with any 2 fingers (excluding thumb)
            elif total_fingers == 2 and fingers[0] == 0:
                y = lm_list[8].y
                if y < 0.5:
                    pyautogui.scroll(30)
                else:
                    pyautogui.scroll(-30)

            # Drag with all 5 fingers
            elif fingers == [1,1,1,1,1]:
                x = int(lm_list[8].x * wScr)
                y = int(lm_list[8].y * hScr)
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                pyautogui.moveTo(x, y)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    cv2.imshow("Air Touch Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


