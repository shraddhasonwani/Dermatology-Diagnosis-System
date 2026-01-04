🩺 Dermatology Diagnosis System

An AI-based Dermatology Diagnosis System that helps in identifying different types of skin diseases using Deep Learning (CNN).
The system allows users to upload a skin lesion image and predicts the disease class using a trained neural network model.

📌 Project Overview

Skin diseases are often difficult to diagnose at an early stage. This project aims to assist dermatologists and users by providing a computer-aided diagnosis system using Convolutional Neural Networks (CNNs).

The model is trained on the HAM10000 dataset, which contains dermatoscopic images of common skin lesions.

🎯 Features

Upload skin lesion images

Automatic disease prediction

Deep learning model using CNN

Web-based interface

Fast and accurate results

🧠 Technologies Used
Programming & Tools

Python

HTML, CSS, JavaScript

Flask (Web Framework)

Machine Learning / AI

TensorFlow

Keras

Convolutional Neural Networks (CNN)

Database (if used)

MySQL / SQLite

📊 Dataset

Dataset Name: HAM10000 (Human Against Machine with 10000 training images)

Source: Kaggle

Classes: 7 different skin lesion categories

Note: Dataset is not included in this repository due to large size.

🏗️ Project Structure
Dermatology-Diagnosis-System/
│
├── static/              # CSS, JS, Images
├── templates/           # HTML templates
├── model/               # Trained ML model
├── app.py               # Flask application
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── .gitignore           # Ignored files

▶️ How to Run the Project
Step 1: Clone the repository
git clone https://github.com/shraddhasonwani/Dermatology-Diagnosis-System.git
cd Dermatology-Diagnosis-System

Step 2: Create virtual environment
python -m venv venv
venv\Scripts\activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Run the application
python app.py

Step 5: Open browser
http://127.0.0.1:5000/

📈 Model Details

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow & Keras

Input: Skin lesion image

Output: Predicted skin disease class

🚀 Future Enhancements

Improve model accuracy

Add more skin disease classes

Deploy on cloud (AWS / Render / Heroku)

Add user authentication

Mobile app integration

👩‍💻 Author

Shraddha Sonwani
🎓 MCA Student
💻 Aspiring Web Developer
📍 Lucknow, India

GitHub: https://github.com/shraddhasonwani

LinkedIn: www.linkedin.com/in/shraddha-sonwani-a22b62247

📜 License

This project is for educational purposes.
