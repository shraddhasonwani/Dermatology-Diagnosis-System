from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model  # type: ignore

import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import mysql.connector
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a strong secret key

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model("skin_model.keras")

# Initialize bcrypt for password hashing
bcrypt = Bcrypt(app)



try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
    )
    cursor = db.cursor(dictionary=True)
    print(" Connected successfully!")
except mysql.connector.Error as err:
    print(" Connection error:", err)


# Class label mapping and disease info (unchanged)
label_map = {
    'akiec': "Actinic Keratosis (Pre-cancerous)",
    'bcc': "Basal Cell Carcinoma (Skin Cancer)",
    'bkl': "Benign Keratosis-like Lesion",
    'df': "Dermatofibroma (Benign)",
    'mel': "Melanoma (Dangerous Skin Cancer)",
    'nv': "Melanocytic Nevus (Mole)",
    'vasc': "Vascular Lesion (e.g., Angioma)"
}

disease_info = {
    'akiec': {
        'name': label_map['akiec'],
        'description': "Actinic Keratosis (AK) is a rough, scaly patch on the skin caused by years of sun exposure...",
        'symptoms': "Dry, scaly, or crusty patches on sun-exposed areas...",
        'treatment': "Cryotherapy, topical treatments, photodynamic therapy, etc."
    },
    'bcc': {
        'name': label_map['bcc'],
        'description': "A common type of skin cancer due to long-term sun exposure...",
        'symptoms': "Pearly bump, pink growth, sore that doesn’t heal...",
        'treatment': "Surgical removal, Mohs surgery, cryotherapy, etc."
    },
    'bkl': {
        'name': label_map['bkl'],
        'description': "Benign lesions like seborrheic keratosis or lentigo...",
        'symptoms': "Waxy, stuck-on appearance; brown/black/tan...",
        'treatment': "Usually none unless irritated or cosmetic concern."
    },
    'df': {
        'name': label_map['df'],
        'description': "A harmless fibrous skin nodule...",
        'symptoms': "Firm small bump, may itch or be tender...",
        'treatment': "No treatment unless bothersome."
    },
    'mel': {
        'name': label_map['mel'],
        'description': "Melanoma is the most serious form of skin cancer...",
        'symptoms': "Irregular moles, asymmetry, border changes...",
        'treatment': "Surgery, immunotherapy, radiation, chemotherapy..."
    },
    'nv': {
        'name': label_map['nv'],
        'description': "Common benign moles formed by melanocytes...",
        'symptoms': "Small, symmetrical spots; flat or raised...",
        'treatment': "None unless it changes shape or size."
    },
    'vasc': {
        'name': label_map['vasc'],
        'description': "Benign blood vessel growths like angiomas...",
        'symptoms': "Red/purple spots or lumps on/under skin...",
        'treatment': "Laser therapy or removal if needed."
    }
}

labels = list(label_map.keys())

# ------------------ USER AUTH ROUTES ------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

        # Open DB connection inside the route
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="dermatology_db",
            charset="utf8"
        )
        cursor = db.cursor(dictionary=True)

        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE username=%s OR email=%s", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username or email already exists.')
            return redirect('/signup')

        # Insert new user
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                        (username, email, hashed_pw))
        db.commit()
        cursor.close()
        db.close()

        flash('Signup successful! Please log in.')
        return redirect('/login')
    
    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="dermatology_db",
            charset="utf8"
        )
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()

        cursor.close()
        db.close()

        if user and bcrypt.check_password_hash(user['password'], password_input):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!')
            return redirect('/')
        else:
            flash('Invalid username or password.')
            return redirect('/login')

    return render_template('login.html')



@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

# ------------------ MAIN APP ROUTES ------------------

@app.route('/')
def index():
    if 'user_id' not in session:
        flash("Please login to access the diagnosis system.", "warning")
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash("Please login to upload and predict.", "warning")
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return "No file part found in request."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        if img is None:
            return "Error: Could not read the uploaded image."

        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        predicted_index = np.argmax(pred)
        confidence = np.max(pred) * 100
        predicted_label = labels[predicted_index]

        info = disease_info[predicted_label]

        return render_template("result.html",
                                name=info['name'],
                                confidence=confidence,
                                description=info['description'],
                                symptoms=info['symptoms'],
                                treatment=info['treatment'],
                                username=session['username'])

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
