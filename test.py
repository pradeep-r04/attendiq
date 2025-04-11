from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load trained data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Loaded faces shape:', FACES.shape)

# Initialize recognizer
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Video and face detector setup
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    if not ret:
        print("Camera error or unplugged!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        prediction = knn.predict(resized_img)
        name = str(prediction[0])

        # Draw name above face with safe position
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = x
        text_y = y - 10 if y - 10 > th else y + h + th + 10

        cv2.rectangle(frame, (text_x, text_y - th - 5), (text_x + tw + 5, text_y + 5), (50, 50, 255), -1)
        cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filepath = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(filepath)

        attendance = [name, timestamp]

    cv2.imshow("Frame", frame)

    # Key detection
    k = cv2.waitKey(1) & 0xFF
    if k == ord('o'):
        speak("Attendance Taken.")
        time.sleep(5)

        # Get current date and time again for consistent file naming
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        filepath = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(filepath)

        with open(filepath, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord('q') or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

video.release()
cv2.destroyAllWindows()
