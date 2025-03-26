# Face Attendance System

This project is a Python-based **Face Recognition Attendance System** that uses OpenCV to recognize faces from a live webcam feed and logs attendance details into an Excel file.

## 🧠 Features

- Real-time face detection using OpenCV.
- Face recognition and attendance logging.
- Automatically creates and updates an Excel sheet (`face_attendance.xlsx`).
- Simple GUI/terminal-based interface.

## 📂 Project Structure

```
face_attendece/
├── new.py                  # Main script for face recognition and attendance
├── face_attendance.xlsx    # Output Excel file storing attendance
├── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x and the following packages installed:

```bash
pip install opencv-python openpyxl numpy
```

### Running the Project

```bash
python new.py
```

- Ensure your webcam is connected.
- The system will detect and recognize faces.
- Attendance will be automatically logged.

## 📒 Output

- **Excel File:** `face_attendance.xlsx` will be generated with columns like `Name`, `Time`, and `Date`.

## 📌 Notes

- Pre-trained face data might need to be added or modified depending on implementation.
- Ensure proper lighting conditions for accurate face detection.
