from flask import Flask, render_template, redirect, url_for, request, flash
import pyrebase # Firebase library
import os


app = Flask(__name__)
app.secret_key = 'recruitervision_secret_key' # Flash messages ke liye zaruri hai

# --- FIREBASE SETUP ---
# Apne Firebase Console se copy ki hui keys yahan dalna
firebaseConfig = {
  "apiKey": "AIzaSyBwZXgIphJqhjjut59nYziPeCEJ_9na3jU",

  "authDomain": "recruitervisionai.firebaseapp.com",

  "projectId": "recruitervisionai",

  "storageBucket": "recruitervisionai.firebasestorage.app",

  "messagingSenderId": "188727590307",

  "appId": "1:188727590307:web:6e127b4338b45ba547ee16",

  "measurementId" : "G-MFN39N9T5T",

  "databaseURL" : "https://recruitervisionai-default-rtdb.firebaseio.com"

};
# Firebase initialize karna
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database() 

# --- ROUTES ---

@app.route('/')
def home():
    # By default, login page par redirect karega
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            # Firebase Auth check karega credentials
            user = auth.sign_in_with_email_and_password(email, password)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
            
    return render_template('auth.html', mode='login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password') # Naya confirm password lena
        
        # 1. Check karna ki dono passwords match ho rahe hain ya nahi
        if password != confirm_password:
            flash('Passwords do not match! Please try again.', 'error')
            return redirect(url_for('signup'))
            
        try:
            # 2. Agar match ho gaye, toh Firebase mein user create karna
            user = auth.create_user_with_email_and_password(email, password)
            
            # 3. Extra details Realtime DB mein save karna
            data = {"name": name, "email": email}
            db.child("users").child(user['localId']).set(data)
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Email already exists or password is too weak', 'error')
            return redirect(url_for('signup'))
            
    return render_template('auth.html', mode='signup')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume_file' not in request.files:
        flash('No file part detected in system pipeline.', 'error')
        return redirect(url_for('dashboard'))
        
    file = request.files['resume_file']
    jd_text = request.form.get('job_description') # Job description string input
    
    if file.filename == '':
        flash('No resume file selected.', 'error')
        return redirect(url_for('dashboard'))
        
    if file and file.filename.endswith('.pdf'):
        # Upload folder ka path setup
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Terminal pe metadata debug statements printable rahengi
        print(f"DEBUG PIPELINE: File saved successfully at {file_path}")
        print(f"DEBUG PIPELINE: Job Description length parsed: {len(jd_text or '')}")
        
        flash(f'File "{file.filename}" successfully uploaded for system parsing!', 'success')
        
        # Jese hi parsing logic setup hoga, yahan se algorithms connect karenge!
        return redirect(url_for('dashboard'))
    else:
        flash('Extension error. Please process structural PDF formats only.', 'error')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)