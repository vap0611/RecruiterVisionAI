from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
import pyrebase
import os
import re
import random
import pickle
import numpy as np
import fitz          # PyMuPDF
import docx          # python-docx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'recruitervision_ultimate_production_v16'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── FIREBASE ──────────────────────────────────────────────────────────────────
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

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("models", "skill_recommender.keras")
TFIDF_PATH      = os.path.join("models", "tfidf_vectorizer.pkl")
MLB_PATH        = os.path.join("models", "mlb_encoder.pkl")
TOP_SKILLS_PATH = os.path.join("models", "top_skills.pkl")

# ── GLOBAL VARS ───────────────────────────────────────────────────────────────
skill_model    = None
skill_tfidf    = None
skill_mlb      = None
top_skills     = []

# ── SKILL LIBRARY  (for JD-mode regex extraction) ─────────────────────────────
SKILL_LIBRARY = [
    "python","java","c++","c#","javascript","typescript","kotlin","swift",
    "go","rust","r","scala","php","ruby","matlab","bash","shell","perl",
    "html","css","react","angular","vue","node.js","express","django",
    "flask","fastapi","spring boot","asp.net","bootstrap","tailwind","jquery",
    "wordpress","next.js","nuxt.js","gatsby","laravel","symfony",
    "mysql","postgresql","mongodb","sqlite","redis","oracle","cassandra",
    "dynamodb","firebase","elasticsearch","sql","nosql","hive","hbase",
    "aws","azure","gcp","docker","kubernetes","jenkins","terraform",
    "ansible","ci/cd","git","github","gitlab","linux","nginx","apache",
    "machine learning","deep learning","nlp","computer vision","tensorflow",
    "keras","pytorch","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "opencv","transformers","bert","xgboost","random forest","neural network",
    "data analysis","data visualization","feature engineering","statistics",
    "hadoop","spark","kafka","airflow","tableau","power bi",
    "android","ios","flutter","react native","xamarin",
    "figma","adobe xd","sketch","canva","wireframing","prototyping",
    "selenium","pytest","junit","jest","cypress","postman","rest api","graphql",
    "excel","jira","jupyter","maven","gradle","webpack",
    "networking","tcp/ip","dns","firewall","vpn","network security","wireshark",
    "communication","leadership","teamwork","problem solving",
    "time management","project management","agile","scrum","critical thinking",
]


# ── MODEL LOADER ──────────────────────────────────────────────────────────────
def load_skill_models():
    """Load trained MLPClassifier + TF-IDF + MLB at app startup."""
    global skill_model, skill_tfidf, skill_mlb, top_skills

    required = [MODEL_PATH, TFIDF_PATH, MLB_PATH, TOP_SKILLS_PATH]
    missing  = [p for p in required if not os.path.exists(p)]

    if missing:
        print(f"[SkillRec] Missing model files: {missing}")
        print("[SkillRec] Run  python train_skill_model.py  first.")
        return

    try:
        with open(MODEL_PATH,      "rb") as f: skill_model  = pickle.load(f)
        with open(TFIDF_PATH,      "rb") as f: skill_tfidf  = pickle.load(f)
        with open(MLB_PATH,        "rb") as f: skill_mlb    = pickle.load(f)
        with open(TOP_SKILLS_PATH, "rb") as f: top_skills   = pickle.load(f)
        print(f"[SkillRec] Model loaded — {len(top_skills)} skills tracked.")
    except Exception as e:
        print(f"[SkillRec] Load error: {e}")


# ── SKILL HELPERS ─────────────────────────────────────────────────────────────
def extract_skills(text: str, library: list = None) -> list:
    """
    Extract skills from text using regex match against library.
    library = SKILL_LIBRARY by default; can also pass top_skills.
    """
    if not text:
        return []
    lib        = library if library else SKILL_LIBRARY
    text_lower = str(text).lower()
    return list({s for s in lib
                 if re.search(r'\b' + re.escape(s) + r'\b', text_lower)})


def cosine_skill_gap(resume_text: str, jd_text: str):
    """
    TF-IDF + Cosine Similarity:
    Find skills present in JD but missing in resume,
    ranked by their cosine similarity to the JD vector.
    """
    jd_skills     = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    resume_lower  = {s.lower() for s in resume_skills}
    missing       = [s for s in jd_skills if s.lower() not in resume_lower]

    if not jd_skills:
        return [], 0.0

    # Overall resume–JD similarity
    try:
        local_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        local_tfidf.fit([resume_text, jd_text] + jd_skills)
        mat        = local_tfidf.transform([resume_text, jd_text]).toarray()
        sim_score  = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])

        # Rank each missing skill by its cosine sim to JD vector
        jd_vec = mat[1:2]
        ranked = []
        for skill in missing:
            sv = local_tfidf.transform([skill]).toarray()
            sc = float(cosine_similarity(sv, jd_vec)[0][0])
            ranked.append((skill, round(sc, 3)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked, round(sim_score, 3)

    except Exception:
        return [(s, 0.0) for s in missing], 0.0


def get_smart_recommendations(resume_text: str, jd_text: str = "",
                               top_k: int = 10, threshold: float = 0.20) -> dict:
    """
    Two-mode recommendation engine:

    ── JD MODE (jd_text given) ─────────────────────────────────────────────
      Priority 1 : Skills explicitly in JD but missing in resume
                   (TF-IDF cosine-ranked)
      Priority 2 : Skills predicted by MLP that overlap with JD vocabulary
                   (model filters to JD-relevant only)

    ── RESUME MODE (no jd_text) ────────────────────────────────────────────
      MLP model predicts missing skills from resume text alone
    """
    existing_skills = extract_skills(resume_text)
    existing_lower  = {s.lower() for s in existing_skills}

    jd_gap_ranked = []
    jd_sim_score  = 0.0
    jd_gap_skills = []
    model_recs    = []
    model_conf    = {}
    model_loaded  = False

    # ── Step 1: JD-based gap (TF-IDF + Cosine Similarity) ─────────────────
    if jd_text.strip():
        jd_gap_ranked, jd_sim_score = cosine_skill_gap(resume_text, jd_text)
        jd_gap_skills = [s for s, _ in jd_gap_ranked]

    # ── Step 2: MLP model predictions ─────────────────────────────────────
    if skill_model is not None and skill_tfidf is not None and skill_mlb is not None:
        try:
            combined   = resume_text + " " + jd_text
            X          = skill_tfidf.transform([combined]).toarray().astype(np.float32)
            proba_list = skill_model.predict_proba(X)
            jd_vocab   = {s.lower() for s in extract_skills(jd_text)} if jd_text.strip() else set()

            def get_prob(pa):
                """Extract P(label=1) safely from any predict_proba format."""
                if hasattr(pa, "ndim"):
                    if pa.ndim == 2:   return float(pa[0][1])   # shape (1,2)
                    if pa.ndim == 1:   return float(pa[1])      # shape (2,)
                if hasattr(pa, "__len__") and len(pa) >= 2:
                    return float(pa[1])
                return float(pa)  # scalar fallback

            for sk, pa in zip(skill_mlb.classes_, proba_list):
                try:
                    sc = get_prob(pa)
                except Exception:
                    continue
                if sc < threshold:             continue
                if sk.lower() in existing_lower: continue
                if jd_text.strip() and sk.lower() not in jd_vocab: continue
                model_recs.append(sk)
                model_conf[sk] = round(sc, 3)

            model_recs.sort(key=lambda s: model_conf[s], reverse=True)
            model_recs   = model_recs[:top_k]
            model_loaded = True

        except Exception as e:
            print(f"[SkillRec] MLP prediction error: {e}")

    # ── Step 3: Merge priority list ────────────────────────────────────────
    seen     = set()
    priority = []

    def add(skill):
        sk = skill.lower()
        if sk not in seen and sk not in existing_lower:
            priority.append(skill)
            seen.add(sk)

    if jd_text.strip():
        for s in jd_gap_skills: add(s)
        for s in model_recs:    add(s)
        # Fallback: JD mein skills explicitly nahi thi, model se bina filter ke lo
        if not priority and skill_model is not None:
            try:
                X2    = skill_tfidf.transform([resume_text + " " + jd_text]).toarray().astype(np.float32)
                plist = skill_model.predict_proba(X2)
                recs2 = []
                for sk2, pa in zip(skill_mlb.classes_, plist):
                    try:
                        if hasattr(pa, "ndim") and pa.ndim == 2: sc2 = float(pa[0][1])
                        elif hasattr(pa, "__len__") and len(pa) >= 2: sc2 = float(pa[1])
                        else: sc2 = float(pa)
                    except Exception: continue
                    if sc2 >= threshold and sk2.lower() not in existing_lower:
                        recs2.append((sk2, sc2))
                recs2.sort(key=lambda x: x[1], reverse=True)
                for sk2, _ in recs2[:top_k]: add(sk2)
            except Exception as e:
                print(f"[SkillRec] Fallback error: {e}")
        if not priority:
            for s in top_skills:
                if s.lower() not in existing_lower: add(s)
    else:
        for s in model_recs: add(s)
        if not priority:
            for s in top_skills:
                if s.lower() not in existing_lower: add(s)

    # ── Step 4: Skills to improve ──────────────────────────────────────────
    # Already in resume AND mentioned in JD → needs more depth
    skills_to_improve = []
    if jd_text.strip():
        jd_present        = extract_skills(jd_text)
        skills_to_improve = [s for s in jd_present
                             if s.lower() in existing_lower][:5]

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



# ── ROAST GENERATOR ───────────────────────────────────────────────────────────
def generate_constructive_roast(score, missing_skills):
    if score >= 75:
        roasts = [
            "Decent match, but decent is not hired. Everyone applying has basics — what makes you irreplaceable?",
            "Good score, but tech interviews don't care about scores, they care about depth. Are you deep or just wide?",
            "You match the JD, cool. So do 300 other applicants. Your differentiator is missing from this resume.",
            "Strong foundation, weak execution. The gap between knowing and doing is where most careers stall.",
            "You are in the top half, which means you are still competing with everyone else in the top half. Think about that.",
        ]
    elif score >= 45:
        roasts = [
            "Your resume looks like it was compiled with legacy dependencies. If your career was an API, it would throw a 410 Gone error.",
            "In this domain, keeping up is not enough and you are barely walking. Git push some actual skills before the market force-quits your career.",
            "You have got the basics, cool, but so do a million other bootcamp graduates. Stand out or get filtered out by the first regex the recruiter runs.",
            "Refactoring your code won't save you if you don't refactor your skill set. The tech stack you're proud of is already deprecated in production.",
            "Nice profile, but automation is coming for jobs exactly like this. Better plug those skill holes before an open-source script replaces you.",
        ]
    else:
        roasts = [
            "The field you are in, nothing is good enough. You think learning a framework makes you an engineer? Your stack is aging faster than milk.",
            "Market standard is running on Kubernetes and you are still struggling with basics. Level up or stay local forever.",
            "If your career was a Docker container, it would be running on an unsupported base image. Time to rebuild from scratch.",
            "Your skill gap is not a bug, it is a feature — a feature that gets resumes auto-rejected. Patch it.",
            "Git blame would point straight at your skill set for this mismatch. The commit history of your career needs a serious rebase.",
        ]

    selected = random.choice(roasts)
    if missing_skills:
        punch = f" Specifically, ignoring {', '.join(missing_skills[:3])} is career suicide right now. Fix this."
        return selected + punch
    return selected + " Even with a decent match, complacency is your biggest bug. Optimize now."


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('landing.html')  # ← bas itna

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        if (not email and not password) or (email == "admin@test.com" and password == "admin123"):
            flash('Logged in via developer bypass.', 'success')
            return redirect(url_for('dashboard'))
        try:
            auth.sign_in_with_email_and_password(email, password)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        except Exception:
            flash('Invalid credentials.', 'error')
            return redirect(url_for('login'))
    return render_template('auth.html', mode='login')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name, email = request.form.get('name'), request.form.get('email')
        password    = request.form.get('password')
        try:
            user = auth.create_user_with_email_and_password(email, password)
            db.child("users").child(user['localId']).set({"name": name, "email": email})
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Registration error or weak password.', 'error')
            return redirect(url_for('signup'))
    return render_template('auth.html', mode='signup')


@app.route('/logout')
def logout():
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    return render_template(
        'dashboard.html', score=None, summary=None,
        existing_skills=[], recommended_skills=[],
        jd_gap_skills=[], skills_to_improve=[],
        model_confidence={}, jd_similarity_score=0,
        skill_model_loaded=False,
        user_name='Recruiter',
        roast=None,
    )


@app.route('/upload_resume', methods=['POST'])
def upload_resume():

    if 'resume_file' not in request.files:
        flash('No file detected.', 'error')
        return redirect(url_for('dashboard'))

    file    = request.files['resume_file']
    jd_text = request.form.get('job_description', '').strip()

    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('dashboard'))
    if not jd_text:
        flash('Please enter a Job Description.', 'error')
        return redirect(url_for('dashboard'))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)



    # ── Text extraction ────────────────────────────────────────────────────
    extracted_text = ""
    try:
        if file.filename.endswith('.pdf'):
            doc = fitz.open(file_path)
            for page in doc:
                extracted_text += (page.get_text() or "") + " "
            doc.close()
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                extracted_text += (para.text or "") + " "
        else:
            flash('Upload PDF or .docx only.', 'error')
            return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"[Parse] Error: {e}")
        flash('Error reading the document.', 'error')
        return redirect(url_for('dashboard'))

    clean_text = " ".join(extracted_text.split())


    if not clean_text:
        flash('Could not extract text from document.', 'error')
        return redirect(url_for('dashboard'))

    # ── Fast summary (first 3 sentences) ─────────────────────────────────
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 40]
    resume_summary = " ".join(sentences[:5]) if sentences else clean_text[:450]
    if len(resume_summary) > 600:
        resume_summary = resume_summary[:600] + "..."

    # ── ATS score (TF-IDF cosine similarity — no torch needed) ──────────────
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        _tv  = _TV(stop_words='english', max_features=3000)
        _mat = _tv.fit_transform([clean_text, jd_text]).toarray()
        ats_score = round(float(cosine_similarity(_mat[0:1], _mat[1:2])[0][0]) * 100, 2)
    except Exception as e:
        print(f"[ATS] Error: {e}")
        ats_score = 0.0

    # ── Skill recommendations ──────────────────────────────────────────────
    skill_data = get_smart_recommendations(
        resume_text = clean_text,
        jd_text     = jd_text,
        top_k       = 10,
        threshold   = 0.20,
    )

    roast = generate_constructive_roast(ats_score, skill_data['jd_gap_skills'])

    return render_template(
        'dashboard.html',
        # existing vars
        summary              = resume_summary,
        filename             = file.filename,
        score                = ats_score,
        # skill vars
        existing_skills      = skill_data["existing_skills"],
        recommended_skills   = skill_data["recommended_skills"],
        jd_gap_skills        = skill_data["jd_gap_skills"],
        skills_to_improve    = skill_data["skills_to_improve"],
        model_confidence     = skill_data["model_confidence"],
        jd_similarity_score  = skill_data["jd_similarity_score"],
        skill_model_loaded   = skill_data["model_loaded"],
        user_name            = 'Recruiter',
        roast                = roast,
    )




# ── AJAX ROUTE — returns JSON, no page reload ────────────────────────────────
@app.route('/')
def landing():
    return render_template('landing.html')
@app.route('/analyse', methods=['POST'])
def analyse():
    import random
    if 'resume_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file    = request.files['resume_file']
    jd_text = request.form.get('job_description', '').strip()

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not jd_text:
        return jsonify({'error': 'Please enter a Job Description'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    extracted_text = ""
    try:
        if file.filename.endswith('.pdf'):
            doc = fitz.open(file_path)
            for page in doc:
                extracted_text += (page.get_text() or "") + " "
            doc.close()
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                extracted_text += (para.text or "") + " "
        else:
            return jsonify({'error': 'Upload PDF or .docx only'}), 400
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500

    clean_text = " ".join(extracted_text.split())
    if not clean_text:
        return jsonify({'error': 'Could not extract text'}), 400

    # Summary
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 40]
    resume_summary = " ".join(sentences[:5]) if sentences else clean_text[:450]
    if len(resume_summary) > 600:
        resume_summary = resume_summary[:600] + "..."

    # ATS score
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        _tv  = _TV(stop_words='english', max_features=3000)
        _mat = _tv.fit_transform([clean_text, jd_text]).toarray()
        ats_score = round(float(cosine_similarity(_mat[0:1], _mat[1:2])[0][0]) * 100, 2)
    except Exception:
        ats_score = 0.0

    # Skill recommendations
    skill_data = get_smart_recommendations(clean_text, jd_text, top_k=10, threshold=0.20)

    # Roast
    roast = generate_constructive_roast(ats_score, skill_data['jd_gap_skills'])

    return jsonify({
        'filename'          : file.filename,
        'score'             : ats_score,
        'summary'           : resume_summary,
        'existing_skills'   : skill_data['existing_skills'],
        'recommended_skills': skill_data['recommended_skills'],
        'jd_gap_skills'     : skill_data['jd_gap_skills'],
        'skills_to_improve' : skill_data['skills_to_improve'],
        'roast'             : roast,
    })

# ── JSON endpoint (optional AJAX use) ────────────────────────────────────────
@app.route('/recommend_skills', methods=['POST'])
def recommend_skills():
    try:
        resume_text = request.form.get("resume_text", "").strip()
        jd_text     = request.form.get("job_description", "").strip()
        top_k       = int(request.form.get("top_k", 10))
        threshold   = float(request.form.get("threshold", 0.20))
        if not resume_text:
            return jsonify({"success": False, "error": "resume_text required"}), 400
        result = get_smart_recommendations(resume_text, jd_text, top_k, threshold)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── STARTUP ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_skill_models()
    app.run(debug=True, port=8080, threaded=True)