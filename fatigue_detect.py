# Importing necessary libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame  # For playing and stopping alert sounds
import threading  # To play sound without blocking the main thread

# Initialize pygame mixer for sound management
pygame.mixer.init()

# Load the alert sound
alert_sound = "alert_sound.mp3"
pygame.mixer.music.load(alert_sound)

# Initialize the camera and take the instance
cap = cv2.VideoCapture(0)

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status markers for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Flag to control alert sound
alert_playing = False

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def play_alert_sound():
    global alert_playing
    if not alert_playing:  # Only start sound if not already playing
        pygame.mixer.music.play(-1)  # -1 loops the sound indefinitely
        alert_playing = True

def stop_alert_sound():
    global alert_playing
    if alert_playing:
        pygame.mixer.music.stop()  # Stop the alert sound
        alert_playing = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    face_frame = frame.copy()  # Ensure face_frame is always defined

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Check eye blinks
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Determine drowsiness state and play or stop alert sound
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                threading.Thread(target=play_alert_sound).start()

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                threading.Thread(target=play_alert_sound).start()

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                stop_alert_sound()  # Stop alert sound in active state

        # Display status on the frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks on the face frame
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Display the frames
    cv2.imshow("Frame", frame)
    if len(faces) > 0:
        cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
