#  🎯 AttendIQ – Face Recognition Attendance System
AttendIQ is a Face Recognition Attendance System designed to automate and streamline the attendance process with precision and ease. By leveraging real-time face detection and recognition technology, AttendIQ eliminates the need for manual roll calls or ID-based check-ins.  The system captures facial data during a quick registration process .


---

## 🚀 Features

- 🔍 Real-time Face Detection & Recognition
- 🧑‍💼 User Registration with ID & Name
- 📸 Captures only 5 face samples per user
- ✅ Attendance marked only on pressing the `o` key
- 🗂 Attendance stored in timestamped CSV files
- 🗣 Voice feedback for successful attendance
- 📊 Streamlit dashboard to view attendance data
- 📁 Modular structure with separate files for training, recognition, and interface

---

## 🧰 Technologies Used

- Python 3.x  
- OpenCV  
- NumPy  
- Scikit-learn (KNN)  
- Streamlit  
- win32com (for text-to-speech on Windows)  
- CSV, Pickle (for data storage)  

---

## 📁 Project Structure
face_recog/ ├── data/ │ ├── haarcascade_frontalface_default.xml │ ├── names.pkl │ └── faces_data.pkl ├── Attendance/ │ └── Attendance_dd-mm-yyyy.csv ├──  ├── main.py # User registration and sample capture ├── test.py # Face recognition & attendance logging ├── app.py  # Streamlit interface └── README.md


---

## 🧑‍🎓 How It Works

1. **Register User (main.py)**  
   - Input user ID and Name
   - System captures 5 face samples
   - Saves data into `faces_data.pkl` and `names.pkl`

2. **Recognize & Mark Attendance (test.py)**  
   - Launches webcam feed
   - Detects and recognizes registered faces
   - Press `o` key to log attendance into a dated CSV file
   - Press `q` to exit

3. **Streamlit Dashboard (app.py)**  
   - Run the UI with `streamlit run app.py`
   - Register users, capture faces, and view attendance data in a user-friendly interface

---

## ▶️ Getting Started


## 📌 Notes  
Ensure your webcam is working properly.  
Press o to mark attendance after face is recognized.  
Each user is registered with exactly 5 face samples.  
Attendance records are saved in the Attendance/ folder, labeled by date.  

## 🙌 Acknowledgements  
OpenCV – for real-time face detection  
scikit-learn – for implementing KNN classification  
Streamlit – for making the interface interactive  
Microsoft Speech API – for text-to-speech feature 

## 📜 License
This project is for educational and personal use only.  

## 💡 Future Enhancements  
Database integration (e.g., SQLite or Firebase)  
Email/SMS notification support  
Admin login for secured access  
Attendance analytics dashboard  


| Name    | Email              | LinkedIn                                      | GitHub                      | Instagram                     |
|---------|--------------------|-----------------------------------------------|-----------------------------|-------------------------------|
| Pradeep | pradeep.singh04r@gmail.com  | [LinkedIn](https://linkedin.com/in/pradeep-singh4) | [GitHub](https://github.com/pradeep-r04) | [Instagram](https://instagram.com/whypradeeep) |
