import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

user_id = input("Enter Your ID: ").strip()
user_name = input("Enter Your Name: ").strip()
label = f"{user_id}_{user_name}"  # e.g., 101_Pradeep

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Camera not accessible.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < 10 and i % 3 == 0:  # slight delay in sampling
            faces_data.append(resized_img)

        i += 1

        # Draw rectangle and label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.rectangle(frame, (x, y-40), (x + tw + 10, y), (50, 50, 255), -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.putText(frame, f"Samples: {len(faces_data)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 10:
        break

video.release()
cv2.destroyAllWindows()

# Convert and reshape face data
faces_data = np.asarray(faces_data).reshape(10, -1)

# Save faces
if not os.path.exists('data'):
    os.makedirs('data')

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump([label]*10, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        old_faces = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        old_labels = pickle.load(f)

    all_faces = np.append(old_faces, faces_data, axis=0)
    all_labels = old_labels + [label]*10

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(all_faces, f)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(all_labels, f)

print(f"✅ Successfully registered {label} with 10 samples.")
