# 🧠 Face Recognition Attendance System

A smart attendance tracking system using **Face Recognition** and **Python**.  
This project automates attendance marking by identifying faces through the webcam in real-time and storing attendance details automatically in CSV and Excel formats.

---

## 🚀 Features

- 🎥 **Real-time face detection & recognition** using OpenCV  
- 🧾 **Automatic attendance logging** (Name, Roll, Time)  
- 💾 **Exports attendance** to CSV and Excel  
- 🌐 **Web-based interface** built with Flask  
- 🧍‍♂️ Stores registered user face images for recognition  
- ⚡ Lightweight, fast, and easy to use  

---


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DhananjayKothawale/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System


## 2️⃣ Create a Virtual Environment (Optional)
python -m venv venv  
venv\Scripts\activate    # For Windows  
# OR  
source venv/bin/activate # For Linux/Mac  

---

## 3️⃣ Install Dependencies
pip install -r requirements.txt  

---

## 4️⃣ Run the Application
python app.py  

Now open your browser and go to 👉 http://127.0.0.1:5000/

---

## 🧠 How It Works

1. Load Haar Cascade classifier for face detection.  
2. Encode and store facial data of registered users.  
3. Start webcam — detect and recognize faces in real-time.  
4. Match detected faces with stored encodings.  
5. If a match is found → mark attendance with Name, Roll, and Timestamp.  
6. Save attendance automatically in both CSV and Excel formats.  

---

## 🧩 Technologies Used

| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Web Framework | Flask |
| Face Detection | OpenCV, Haar Cascade |
| Face Recognition | face_recognition library |
| Data Storage | CSV, Excel |
| UI | HTML, CSS |
| Libraries | Pandas, OpenPyXL, Numpy |

---

## 📊 Example Output

| Name | Roll | Time |
|------|------|------|
| Dhananjay | 101 | 10:35 AM |
| Rohit | 102 | 10:36 AM |

**Attendance file example:**  
Attendance/Attendance-2025-10-17.csv  
Attendance/Attendance-2025-10-17.xlsx  

---

## 🧾 Future Enhancements

- 🔐 Admin login & role-based dashboard  
- ☁️ Cloud deployment with live database  
- 📲 Mobile app integration  
- 🤖 Deep learning model for improved accuracy  
- 📤 Email or SMS notification for absent students  

---

## 👨‍💻 Author

**Dhananjay Machindra Kothawale**  
🎓 Dr. Babasaheb Ambedkar Technological University, Lonere  
📘 Department of Computer Science and Engineering (Data Science)  
🔗 [GitHub Profile](https://github.com/DhananjayKothawale)

---

## 🪪 License

This project is licensed under the **MIT License**.  
See the LICENSE file for more details.
