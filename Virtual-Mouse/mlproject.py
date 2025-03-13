import tkinter as tk
from tkinter import *
from tkinter import ttk
import threading
import cv2
import mediapipe as mp
import pyautogui        


win = tk.Tk()

win.title("Virtual Mouse")
win.geometry("300x220")

bg = PhotoImage(file='D:/Downloads/Downloads/OneDrive/Desktop/Virtual Mouse/Virtual-Mouse/images.png')

label17 = Label(win, image=bg)
label17.place(x=40, y=0)

def hand_detection():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    index_y = 0

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x*frame_width)
                    y = int(landmark.y*frame_height)
                    if id == 8:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                        index_x = screen_width/frame_width*x
                        index_y = screen_height/frame_height*y

                    if id == 4:
                        cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                        thumb_x = screen_width/frame_width*x
                        thumb_y = screen_height/frame_height*y
                        if abs(index_y - thumb_y) < 50:
                            pyautogui.click()
                            pyautogui.sleep(1)
                        elif abs(index_y - thumb_y) < 200:
                            pyautogui.moveTo(index_x, index_y)
                        elif abs(index_y - thumb_y) < 30:
                            pyautogui.click(button='right')
                        elif abs(index_y - thumb_y) < 15:
                            pyautogui.click(button='left')
                            pyautogui.scroll(10)
        cv2.imshow('Virtual Mouse', frame)
        cv2.waitKey(1)

def eye_detection():
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            if (left[0].y - left[1].y) < 0.5:
                pyautogui.click()
                pyautogui.sleep(1)
        cv2.imshow('Eye Controlled Mouse', frame)
        cv2.waitKey(1)


button = ttk.Button(win, text="Virtual Mouse Using Hand", command=lambda: threading.Thread(target=hand_detection).start())
button2 = ttk.Button(win, text="Virtual Mouse Using Eye", command=lambda: threading.Thread(target=eye_detection).start())

button.pack()
button2.pack()

win.mainloop()
