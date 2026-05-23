<<<<<<< HEAD
from flask import Flask, render_template, redirect, url_for, request, flash
import pyrebase # Firebase library
import os

=======
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
import pyrebase
import os
import re
import pickle
import numpy as np
import fitz          # PyMuPDF for PDF
import docx          # python-docx for Word
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt_tab')

# TensorFlow removed — MLPClassifier (sklearn) use ho raha hai
# No segfault, no GPU required
>>>>>>> 16e51dd (New updates)

app = Flask(__name__)
app.secret_key = 'recruitervision_secret_key'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

firebaseConfig = {
    "apiKey": "AIzaSyBwZXgTpHJqhjjutS9nyZiPeCEJ_9na3jU",
    "authDomain": "recruitervisionai.firebaseapp.com",
    "projectId": "recruitervisionai",
    "storageBucket": "recruitervisionai.firebasestorage.app",
    "messagingSenderId": "188727590307",
    "appId": "1:188727590307:web:6e127b4338b45ba547ee16",
    "measurementId": "G-MFN39N9T5T",
    "databaseURL": "https://recruitervisionai-default-rtdb.firebaseio.com"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth     = firebase.auth()
db       = firebase.database()

print("--> Loading Sentence Transformer Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("--> Model loaded successfully and ready for ATS processing!")

# ── Skill Model Paths ─────────────────────────────────────────────────────────
SKILL_MODEL_PATH = os.path.join("models", "skill_recommender.keras")
TFIDF_SKILL_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
MLB_PATH         = os.path.join("models", "mlb_encoder.pkl")
TOP_SKILLS_PATH  = os.path.join("models", "top_skills.pkl")

skill_model  = None
skill_tfidf  = None
skill_mlb    = None
top_skills   = None

def load_skill_model():
    global skill_model, skill_tfidf, skill_mlb, top_skills
    missing = [p for p in [SKILL_MODEL_PATH, TFIDF_SKILL_PATH, MLB_PATH, TOP_SKILLS_PATH]
               if not os.path.exists(p)]
    if missing:
        print(f"[SkillRec] Missing files: {missing}")
        print("[SkillRec] Run  python train_skill_model.py  first.")
        return
    try:
        # MLPClassifier pickle se load hota hai (.keras extension, pickle format)
        with open(SKILL_MODEL_PATH, "rb") as f: skill_model = pickle.load(f)
        with open(TFIDF_SKILL_PATH, "rb") as f: skill_tfidf = pickle.load(f)
        with open(MLB_PATH,         "rb") as f: skill_mlb   = pickle.load(f)
        with open(TOP_SKILLS_PATH,  "rb") as f: top_skills  = pickle.load(f)
        print("[SkillRec] Skill recommendation model loaded!")
    except Exception as e:
        print(f"[SkillRec] Load error: {e}")

# ── Skill Library ─────────────────────────────────────────────────────────────
SKILL_LIBRARY = [
    "python","java","c++","c#","javascript","typescript","kotlin","swift",
    "go","rust","r","scala","php","ruby","matlab","bash","shell","perl",
    "html","css","react","angular","vue","node.js","express","django",
    "flask","fastapi","spring boot","asp.net","bootstrap","tailwind","jquery",
    "mysql","postgresql","mongodb","sqlite","redis","oracle","cassandra",
    "dynamodb","firebase","elasticsearch","sql","nosql","hive","hbase",
    "aws","azure","gcp","docker","kubernetes","jenkins","terraform",
    "ansible","ci/cd","git","github","gitlab","linux","nginx","apache",
    "machine learning","deep learning","nlp","computer vision","tensorflow",
    "keras","pytorch","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "opencv","transformers","bert","xgboost","random forest","neural network",
    "data analysis","data visualization","feature engineering","statistics",
    "hadoop","spark","kafka","airflow","tableau","power bi",
    "android","ios","flutter","react native","xamarin","swift ui",
    "selenium","pytest","junit","jest","cypress","postman","rest api","graphql",
    "excel","jira","figma","git","jupyter","vs code","pycharm","intellij",
    "maven","gradle","webpack",
    "communication","leadership","teamwork","problem solving",
    "time management","project management","agile","scrum","critical thinking",
]

def extract_skills_from_text(text: str) -> list:
    text_lower = str(text).lower()
    found = []
    for skill in SKILL_LIBRARY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))

def cosine_skill_gap(resume_text: str, jd_text: str):
    """
    TF-IDF + Cosine Similarity se JD-specific skill gap nikalo.

    Logic:
    1. JD text se saari skills extract karo
    2. Resume mein jo nahi hain woh = hard gap
    3. Har missing skill ka JD vector se cosine similarity nikalo
       → JD se zyada related skill = higher score = pehle dikhao
    """
    jd_skills     = extract_skills_from_text(jd_text)
    resume_skills = extract_skills_from_text(resume_text)
    resume_lower  = [s.lower() for s in resume_skills]

    # Hard missing: JD mein hain, resume mein bilkul nahi
    hard_gap = [s for s in jd_skills if s.lower() not in resume_lower]

    if not jd_skills:
        return [], 0.0

    if not hard_gap:
        # Sab skills already hain — similarity score calculate karke return karo
        try:
            local_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            tfidf_mat   = local_tfidf.fit_transform([resume_text, jd_text]).toarray()
            sim_score   = float(cosine_similarity(
                tfidf_mat[0].reshape(1, -1),
                tfidf_mat[1].reshape(1, -1)
            )[0][0])
        except Exception:
            sim_score = 1.0
        return [], round(sim_score, 3)

    try:
        local_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_mat   = local_tfidf.fit_transform([resume_text, jd_text]).toarray()
        resume_vec  = tfidf_mat[0].reshape(1, -1)
        jd_vec      = tfidf_mat[1].reshape(1, -1)
        sim_score   = float(cosine_similarity(resume_vec, jd_vec)[0][0])

        # Har missing skill ka JD se similarity score
        ranked = []
        for skill in hard_gap:
            sv = local_tfidf.transform([skill]).toarray()
            sc = float(cosine_similarity(sv, jd_vec)[0][0])
            ranked.append((skill, round(sc, 3)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked, round(sim_score, 3)

    except Exception:
        return [(s, 0.0) for s in hard_gap], 0.0


def get_jd_domain_skills(jd_text: str, threshold: float = 0.20) -> list:
    """
    Trained MLPClassifier model JD text ko directly predict karta hai.
    Koi hardcoded map nahi — model ne JD data se khud seekha hai.
    Agar model load nahi hua toh empty list return karo.
    """
    if skill_model is None or skill_tfidf is None or skill_mlb is None:
        return []
    try:
        X          = skill_tfidf.transform([jd_text]).toarray().astype(np.float32)
        proba_list = skill_model.predict_proba(X)
        matched    = []
        for sk, proba_arr in zip(skill_mlb.classes_, proba_list):
            if float(proba_arr[0][1]) >= threshold:
                matched.append(sk)
        return matched
    except Exception as e:
        print(f"[SkillRec] get_jd_domain_skills error: {e}")
        return []


def get_skill_recommendations(resume_text, jd_text="", top_k=10, threshold=0.20):
    existing_skills = extract_skills_from_text(resume_text)
    existing_lower  = [s.lower() for s in existing_skills]

    # ── Step 1: JD se exact skill gap (TF-IDF + Cosine Similarity) ────────
    jd_gap_ranked = []
    jd_sim_score  = 0.0
    jd_gap_skills = []

    if jd_text.strip():
        jd_gap_ranked, jd_sim_score = cosine_skill_gap(resume_text, jd_text)
        jd_gap_skills = [s for s, _ in jd_gap_ranked[:top_k]]

    # ── Step 2: JD domain se expected skills (keyword matching) ───────────
    domain_skills = []
    if jd_text.strip():
        domain_skills = [
            s for s in get_jd_domain_skills(jd_text)
            if s.lower() not in existing_lower
        ]

    # ── Step 3: MLPClassifier predictions (JD se filter karke) ───────────
    model_recs   = []
    model_conf   = {}
    model_loaded = False

    if skill_model is not None and skill_tfidf is not None and skill_mlb is not None:
        try:
            X          = skill_tfidf.transform([resume_text + " " + jd_text]).toarray().astype(np.float32)
            proba_list = skill_model.predict_proba(X)

            # JD diya hai toh sirf JD-relevant skills lo model se
            jd_relevant = set(s.lower() for s in (jd_gap_skills + domain_skills))

            for sk, proba_arr in zip(skill_mlb.classes_, proba_list):
                sc = float(proba_arr[0][1])
                if sc >= threshold and sk.lower() not in existing_lower:
                    # Agar JD diya hai → sirf JD se related skills lo
                    # Agar JD nahi diya → sab lo
                    if jd_text.strip():
                        if sk.lower() in jd_relevant:
                            model_recs.append(sk)
                            model_conf[sk] = round(sc, 3)
                    else:
                        model_recs.append(sk)
                        model_conf[sk] = round(sc, 3)

            model_recs.sort(key=lambda s: model_conf[s], reverse=True)
            model_recs   = model_recs[:top_k]
            model_loaded = True
        except Exception as e:
            print(f"[SkillRec] Prediction error: {e}")

    # ── Step 4: Final priority list banao ────────────────────────────────
    # Order: JD gap first → domain skills → model recs
    priority = []
    seen     = set()

    def add(skill):
        if skill.lower() not in seen and skill.lower() not in existing_lower:
            priority.append(skill)
            seen.add(skill.lower())

    # JD se direct missing skills (highest priority)
    for s in jd_gap_skills:
        add(s)

    # Domain keywords se expected skills
    for s in domain_skills:
        add(s)

    # Model predictions jo JD se match karte hain
    for s in model_recs:
        add(s)

    # Agar kuch nahi mila (no JD, no model) → generic fallback
    if not priority:
        priority = [s for s in SKILL_LIBRARY if s not in existing_lower][:top_k]

    # ── Step 5: Skills to improve ─────────────────────────────────────────
    # JD mein hain + resume mein bhi hain → improve karo inhe
    skills_to_improve = []
    if jd_text.strip():
        jd_present        = extract_skills_from_text(jd_text)
        domain_expected   = get_jd_domain_skills(jd_text)
        all_jd_skills     = list(set(jd_present + domain_expected))
        skills_to_improve = [
            s for s in all_jd_skills
            if s.lower() in existing_lower
        ][:5]

    return {
        "existing_skills"    : existing_skills,
        "recommended_skills" : priority[:top_k],
        "jd_gap_skills"      : jd_gap_skills,
        "jd_gap_ranked"      : jd_gap_ranked[:top_k],
        "skills_to_improve"  : skills_to_improve,
        "model_confidence"   : model_conf,
        "jd_similarity_score": jd_sim_score,
        "model_loaded"       : model_loaded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        if (not email and not password) or (email == "admin@test.com" and password == "admin123"):
            flash('Logged in via developer bypass mode.', 'success')
            return redirect(url_for('dashboard'))
        try:
            auth.sign_in_with_email_and_password(email, password)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash('Invalid credentials.', 'error')
            return redirect(url_for('login'))
    return render_template('auth.html', mode='login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name             = request.form.get('name')
        email            = request.form.get('email')
        password         = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))
        try:
            user = auth.create_user_with_email_and_password(email, password)
            db.child("users").child(user['localId']).set({"name": name, "email": email})
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Registration error or weak password.', 'error')
            return redirect(url_for('signup'))
    return render_template('auth.html', mode='signup')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', score=None, summary=None)

@app.route('/logout')
def logout():
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume_file' not in request.files:
        flash('No file part detected in system pipeline.', 'error')
        return redirect(url_for('dashboard'))

    file    = request.files['resume_file']
    jd_text = request.form.get('job_description', '')

    if file.filename == '':
        flash('No resume file selected.', 'error')
        return redirect(url_for('dashboard'))
    if not jd_text or jd_text.strip() == "":
        flash('Please enter a Job Description to match against.', 'error')
        return redirect(url_for('dashboard'))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    extracted_text = ""
    try:
        if file.filename.endswith('.pdf'):
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                if text:
                    extracted_text += text + " "
            doc.close()
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text:
                    extracted_text += para.text + " "
        else:
            flash('Unsupported file format! Please upload PDF or .docx only.', 'error')
            return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"FILE PARSING ERROR: {str(e)}")
        flash('Error extracting text from the document.', 'error')
        return redirect(url_for('dashboard'))

    clean_text = " ".join(extracted_text.split())
    if not clean_text:
        flash('Missing text dataset content for AI calculation.', 'error')
        return redirect(url_for('dashboard'))

    # Sumy summarization
    try:
        parser            = PlaintextParser.from_string(clean_text, Tokenizer("english"))
        summarizer        = LsaSummarizer()
        summary_sentences = summarizer(parser.document, 6)
        resume_summary    = " ".join([str(s) for s in summary_sentences])
        if not resume_summary.strip():
            resume_summary = clean_text[:450] + "..."
    except Exception as e:
        print(f"SUMY PROCESSING ERROR: {str(e)}")
        resume_summary = clean_text[:450] + "..."

    # ATS score
    try:
        resume_embedding = model.encode([clean_text])
        jd_embedding     = model.encode([jd_text])
        similarity       = cosine_similarity(resume_embedding, jd_embedding)
        ats_score        = round(float(similarity[0][0]) * 100, 2)
    except Exception as e:
        print(f"AI PIPELINE ERROR: {str(e)}")
        flash('Error executing advanced matching algorithm.', 'error')
        return redirect(url_for('dashboard'))

    # Skill recommendations
    skill_data = get_skill_recommendations(
        resume_text = clean_text,
        jd_text     = jd_text,
        top_k       = 10,
        threshold   = 0.20,
    )

    return render_template(
        'dashboard.html',
        # Existing
        summary  = resume_summary,
        filename = file.filename,
        score    = ats_score,
        # Skill data
        existing_skills     = skill_data["existing_skills"],
        recommended_skills  = skill_data["recommended_skills"],
        jd_gap_skills       = skill_data["jd_gap_skills"],
        jd_gap_ranked       = skill_data["jd_gap_ranked"],
        skills_to_improve   = skill_data["skills_to_improve"],
        model_confidence    = skill_data["model_confidence"],
        jd_similarity_score = skill_data["jd_similarity_score"],
        skill_model_loaded  = skill_data["model_loaded"],
    )


# ── NEW ROUTE: /recommend_skills  (JSON — AJAX se bhi call kar sakte ho) ──────
@app.route('/recommend_skills', methods=['POST'])
def recommend_skills():
    try:
        resume_text = request.form.get("resume_text", "").strip()
        jd_text     = request.form.get("job_description", "").strip()
        top_k       = int(request.form.get("top_k", 10))
        threshold   = float(request.form.get("threshold", 0.20))

        if not resume_text:
            return jsonify({"success": False, "error": "resume_text is required."}), 400

        result = get_skill_recommendations(resume_text, jd_text, top_k, threshold)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



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
    load_skill_model()
    app.run(debug=True)