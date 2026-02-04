import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    h, w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        # DRAW HAND LANDMARKS (LINES & DOTS)
        mpDraw.draw_landmarks(
            img,
            handLms,
            mpHands.HAND_CONNECTIONS
        )

        # Index finger tip (id 8)
        index_tip = handLms.landmark[8]
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)

        # Thumb tip (id 4)
        thumb_tip = handLms.landmark[4]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # DRAW TIP DOTS
        cv2.circle(img, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (tx, ty), 10, (0, 0, 255), cv2.FILLED)

        # MOVE MOUSE
        screen_x = int(index_tip.x * screen_w)
        screen_y = int(index_tip.y * screen_h)
        pyautogui.moveTo(screen_x, screen_y)

        # CLICK DETECTION (camera coordinates)
        distance = math.hypot(ix - tx, iy - ty)

        if distance < 35:
            cv2.putText(
                img,
                "CLICK",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            pyautogui.click()
            pyautogui.sleep(0.25)

    cv2.imshow("Gesture Controlled Virtual Mouse", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
