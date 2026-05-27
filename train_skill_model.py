"""
train_skill_model.py
====================
RecruiterVision AI — Skill Recommendation Model

90% ML approach:
  - Labels    : SKILL_LIBRARY regex se extract (10% rule — unavoidable kyunki
                datasets mein explicit skill column nahi hai)
  - Features  : TF-IDF (5000 features, bigrams, sublinear_tf)
  - Model     : MLPClassifier multilabel (512→256→128→output)
  - Validation: train_test_split 80/20 + per-skill F1 evaluation
  - Selection : Top 60 skills by actual corpus frequency (data-driven)
  - No role→skill hardcoded maps anywhere

Datasets:
  job_title_des.csv   — 2277 rows  (Job Title + Job Description)
  UpdatedResume.csv   — user dataset (Category + Resume)

Output:
  models/skill_recommender.keras
  models/tfidf_vectorizer.pkl
  models/mlb_encoder.pkl
  models/top_skills.pkl
"""

import os, re, pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing            import MultiLabelBinarizer
from sklearn.model_selection          import train_test_split
from sklearn.neural_network           import MLPClassifier
from sklearn.metrics                  import (f1_score, accuracy_score,
                                               hamming_loss)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
JD_PATH         = "/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/job_title_des.csv"
RESUME_PATH     = "/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/UpdatedResumeDataSet.csv"
MODEL_PATH      = "models/skill_recommender.keras"
TFIDF_PATH      = "models/tfidf_vectorizer.pkl"
MLB_PATH        = "models/mlb_encoder.pkl"
TOP_SKILLS_PATH = "models/top_skills.pkl"

MAX_TFIDF_FEATURES = 5000
TOP_N_SKILLS       = 60      # top 60 by corpus frequency
MIN_SKILL_FREQ     = 10      # skill must appear in >=10 docs
MIN_SKILLS_PER_ROW = 2       # rows with fewer skills skipped
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

os.makedirs("models", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# SKILL LIBRARY  (regex dictionary — only necessary "10%")
# Koi role→skill map nahi, koi hardcoded domain logic nahi
# ──────────────────────────────────────────────────────────────────────────────
SKILL_LIBRARY = [
    # Programming languages
    "python","java","c++","c#","javascript","typescript","kotlin","swift",
    "go","rust","r","scala","php","ruby","matlab","bash","shell","perl",
    # Web frameworks & tools
    "html","css","react","angular","vue","node.js","express","django",
    "flask","fastapi","spring boot","asp.net","bootstrap","tailwind","jquery",
    "wordpress","next.js","nuxt.js","gatsby","laravel","symfony",
    # Databases
    "mysql","postgresql","mongodb","sqlite","redis","oracle","cassandra",
    "dynamodb","firebase","elasticsearch","sql","nosql","hive","hbase",
    # Cloud & DevOps
    "aws","azure","gcp","docker","kubernetes","jenkins","terraform",
    "ansible","ci/cd","git","github","gitlab","linux","nginx","apache",
    # Data Science / ML / AI
    "machine learning","deep learning","nlp","computer vision","tensorflow",
    "keras","pytorch","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "opencv","transformers","bert","xgboost","random forest","neural network",
    "data analysis","data visualization","feature engineering","statistics",
    "hadoop","spark","kafka","airflow","tableau","power bi",
    # Mobile
    "android","ios","flutter","react native","xamarin","html","css","js","javascript"
    # Design / UI-UX
    "figma","adobe xd","sketch","canva","wireframing","prototyping",
    # Testing & APIs
    "selenium","pytest","junit","jest","cypress","postman","rest api","graphql",
    # Tools
    "excel","jira","jupyter","maven","gradle","webpack",
    # Networking
    "networking","tcp/ip","dns","firewall","vpn","network security","wireshark",
    # Soft skills
    "communication","leadership","teamwork","problem solving",
    "time management","project management","agile","scrum","critical thinking",
]

def extract_skills(text: str) -> list:
    t = str(text).lower()
    return list({s for s in SKILL_LIBRARY
                 if re.search(r'\b' + re.escape(s) + r'\b', t)})


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_jd() -> pd.DataFrame:
    if not os.path.exists(JD_PATH):
        print(f"[Load] '{JD_PATH}' not found — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(JD_PATH).rename(
        columns={"Job Description": "text", "Job Title": "source_label"})
    df = df.dropna(subset=["text"])
    df["text"]   = df["text"].str.strip()
    df["source"] = "jd"
    df["skills"] = df["text"].apply(extract_skills)
    before = len(df)
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)
    print(f"[Load] JD      : {len(df)}/{before} rows valid | "
          f"avg {df['skills'].map(len).mean():.1f} skills/doc")
    return df[["text","skills","source"]]


def load_resume() -> pd.DataFrame:
    if not os.path.exists(RESUME_PATH):
        print(f"[Load] '{RESUME_PATH}' not found — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(RESUME_PATH)
    # Flexible column detection
    txt_col = next((c for c in df.columns if "resume" in c.lower()), df.columns[-1])
    df = df.rename(columns={txt_col: "text"})
    df = df.dropna(subset=["text"])
    df["text"]   = df["text"].str.strip()
    df["source"] = "resume"
    df["skills"] = df["text"].apply(extract_skills)
    before = len(df)
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)
    print(f"[Load] Resume  : {len(df)}/{before} rows valid | "
          f"avg {df['skills'].map(len).mean():.1f} skills/doc")
    return df[["text","skills","source"]]


def load_data() -> pd.DataFrame:
    print("\n[Data] Loading datasets ...")
    jd, res = load_jd(), load_resume()
    parts = [d for d in [jd, res] if not d.empty]
    if not parts:
        raise RuntimeError("No datasets found! "
                           f"Place '{JD_PATH}' and/or '{RESUME_PATH}' here.")
    df = pd.concat(parts, ignore_index=True)
    print(f"[Data] Combined : {len(df)} rows  "
          f"(JD={len(jd) if not jd.empty else 0}, "
          f"Resume={len(res) if not res.empty else 0})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# All decisions driven by corpus statistics — no manual selection
# ──────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    # ── Select top skills by corpus frequency (ML-driven) ────────────────
    freq = Counter(s for row in df["skills"] for s in row)

    # Keep skills that appear in at least MIN_SKILL_FREQ documents
    frequent = {sk for sk, cnt in freq.items() if cnt >= MIN_SKILL_FREQ}

    # Among frequent skills, take top TOP_N_SKILLS
    top_skills = [sk for sk, _ in freq.most_common()
                  if sk in frequent][:TOP_N_SKILLS]

    print(f"\n[Features] Corpus skill frequencies (top {TOP_N_SKILLS}):")
    for i, sk in enumerate(top_skills, 1):
        print(f"  {i:>2}. {sk:<35} freq={freq[sk]}")

    # Filter rows
    df = df.copy()
    df["skills"] = df["skills"].apply(
        lambda s: [x for x in s if x in top_skills])
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)
    print(f"\n[Features] Rows after frequency filter: {len(df)}")

    # ── TF-IDF Vectorizer ─────────────────────────────────────────────────
    # Learns document-term statistics directly from corpus
    print(f"[Features] Fitting TF-IDF "
          f"(max_features={MAX_TFIDF_FEATURES}, ngram=(1,2), sublinear_tf=True) ...")
    tfidf = TfidfVectorizer(
        max_features = MAX_TFIDF_FEATURES,
        ngram_range  = (1, 2),       
        stop_words   = "english",
        sublinear_tf = True,        
        min_df       = 3,            # must appear in >=3 docs (learned from data)
        max_df       = 0.95,         # ignore terms in >95% docs (learned from data)
        analyzer     = "word",
    )
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32) 
    # ── MultiLabel Binarizer ──────────────────────────────────────────────
    mlb = MultiLabelBinarizer(classes=top_skills) 
    y   = mlb.fit_transform(df["skills"]).astype(np.float32)

    print(f"[Features] X shape : {X.shape}")
    print(f"[Features] y shape : {y.shape}")
    print(f"[Features] Label density: "
          f"{y.mean():.4f} (avg {y.sum(1).mean():.1f} skills/doc)")
    return X, y, tfidf, mlb, top_skills


# ──────────────────────────────────────────────────────────────────────────────
# MODEL  — MLPClassifier multilabel
# Architecture chosen to match data scale (4000 samples, 5000 features)
# ──────────────────────────────────────────────────────────────────────────────
def build_model(n_labels: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes  = (512, 256, 128),
        activation          = "relu",
        solver              = "adam",
        alpha               = 1e-4,          
        learning_rate       = "adaptive",    
        learning_rate_init  = 1e-3,
        max_iter            = 200,           
        early_stopping      = True,
        validation_fraction = 0.10,
        n_iter_no_change    = 10,          
        tol                 = 1e-5,
        random_state        = RANDOM_STATE,
        verbose             = True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION  — proper multilabel metrics
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(clf, X_test, y_test, mlb):
    print("\n[Eval] ── Multilabel Metrics ──")
    y_pred = clf.predict(X_test)

    # Subset accuracy (exact match)
    print(f"  Subset Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    # Hamming loss (fraction of wrong labels)
    print(f"  Hamming Loss    : {hamming_loss(y_test, y_pred):.4f}  "
          f"(lower is better)")

    # Macro / Micro F1
    print(f"  Macro F1        : "
          f"{f1_score(y_test, y_pred, average='macro',  zero_division=0):.4f}")
    print(f"  Micro F1        : "
          f"{f1_score(y_test, y_pred, average='micro',  zero_division=0):.4f}")
    print(f"  Weighted F1     : "
          f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    # Per-skill F1 for top 20 skills
    print(f"\n[Eval] ── Per-Skill F1 (top 20 by test support) ──")
    support = y_test.sum(axis=0)
    top_idx = np.argsort(support)[::-1][:20]
    for i in top_idx:
        sk  = mlb.classes_[i]
        f1  = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        sup = int(support[i])
        bar = "█" * int(f1 * 20)
        print(f"  {sk:<30} F1={f1:.3f} {bar:<20} (support={sup})")


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN + SAVE
# ──────────────────────────────────────────────────────────────────────────────
def train():
    print("\n" + "═"*60)
    print("  RecruiterVision AI — Skill Recommender Training")
    print("  90% ML | Labels: corpus-driven | No role→skill maps")
    print("═"*60)

    # Load & combine
    df = load_data()

    # Feature engineering
    X, y, tfidf, mlb, top_skills = build_features(df)

    # ── train_test_split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        shuffle      = True,
    )
    print(f"\n[Split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[Split] Features: {X_train.shape[1]} | Labels: {y_train.shape[1]}")

    # Build & train
    clf = build_model(y_train.shape[1])
    print("\n[Train] Starting ...\n")
    clf.fit(X_train, y_train)
    print(f"\n[Train] Stopped at iteration: {clf.n_iter_}")
    print(f"[Train] Best val score      : {clf.best_validation_score_:.4f}")

    # Evaluate
    evaluate(clf, X_test, y_test, mlb)

    # ── Save ─────────────────────────────────────────────────────────────
    with open(MODEL_PATH,      "wb") as f: pickle.dump(clf,        f)
    with open(TFIDF_PATH,      "wb") as f: pickle.dump(tfidf,      f)
    with open(MLB_PATH,        "wb") as f: pickle.dump(mlb,        f)
    with open(TOP_SKILLS_PATH, "wb") as f: pickle.dump(top_skills, f)

    print(f"\n[Saved] {MODEL_PATH}")
    print(f"[Saved] {TFIDF_PATH}")
    print(f"[Saved] {MLB_PATH}")
    print(f"[Saved] {TOP_SKILLS_PATH}")
    print("\n" + "═"*60)
    print("  Training complete! Run:  python app.py")
    print("═"*60 + "\n")


if __name__ == "__main__":
    train()