import streamlit as st
import cv2
import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import csv
import time
import pandas as pd

# Create necessary folders
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Utility function to load training data
def load_data():
    if os.path.exists("data/faces_data.pkl") and os.path.exists("data/names.pkl"):
        with open("data/faces_data.pkl", "rb") as f:
            faces_data = pickle.load(f)
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        return faces_data, names
    return np.array([]), []

# Utility function to save training data
def save_data(faces_data, names):
    with open("data/faces_data.pkl", "wb") as f:
        pickle.dump(faces_data, f)
    with open("data/names.pkl", "wb") as f:
        pickle.dump(names, f)

# Register user section
st.title("Face Recognition Attendance System")
st.header("Register New User")
user_id = st.text_input("Enter ID")
user_name = st.text_input("Enter Name")
capture = st.button("Capture Face Samples")

if capture and user_id and user_name:
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0
    stframe = st.empty()
    st.info("Capturing 5 face samples. Please look at the camera.")

    while len(faces_data) < 5:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to open webcam.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            faces_data.append(resized_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {len(faces_data)}/5", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cv2.waitKey(100)

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data).reshape(5, -1)
    existing_faces, existing_names = load_data()

    if existing_faces.size == 0:
        new_faces = faces_data
        new_names = [f"{user_name} ({user_id})"] * 5
    else:
        new_faces = np.append(existing_faces, faces_data, axis=0)
        new_names = existing_names + [f"{user_name} ({user_id})"] * 5

    save_data(new_faces, new_names)
    st.success("Face samples captured and saved!")

# Attendance section
st.header("Mark Attendance")
if st.button("Start Attendance Camera"):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    st.warning("Press 'o' key in the webcam window to mark attendance. Press 'q' to quit.")

    COL_NAMES = ['NAME', 'TIME']
    marked_names = set()
    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Webcam error.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            name = str(output[0])

            # Draw name
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y), (50, 50, 255), -1)
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Attendance", frame)
        key = cv2.waitKey(1)

        if key == ord('o'):
            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = knn.predict(resized_img)
                name = str(output[0])
                if name not in marked_names:
                    ts = time.time()
                    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    filepath = f"Attendance/Attendance_{date}.csv"
                    exist = os.path.exists(filepath)
                    with open(filepath, "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not exist:
                            writer.writerow(COL_NAMES)
                        writer.writerow([name, timestamp])
                    marked_names.add(name)
        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# View attendance section
st.header("Today's Attendance")
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
filepath = f"Attendance/Attendance_{date}.csv"
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    st.dataframe(df)
else:
    st.info("No attendance records found for today.")
