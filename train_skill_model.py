"""
train_skill_model.py
====================
RecruiterVision AI — Skill Recommendation Model

Datasets : job_title_des.csv   (Job Title, Job Description)  — 2277 rows
           UpdatedResume.csv   (Category, Resume)             — user dataset

Strategy : ZERO hardcoding
           Labels = skills extracted from raw text using SKILL_LIBRARY regex
           Model  = MLPClassifier (TF-IDF input → skill probability output)

           job_title_des  trains model to understand: "JD mein ye words → ye skills"
           UpdatedResume  trains model to understand: "Resume mein ye words → ye skills"
           Combined       → model works well for both JD and resume inputs

Output   : models/skill_recommender.keras
           models/tfidf_vectorizer.pkl
           models/mlb_encoder.pkl
           models/top_skills.pkl

Run      : python train_skill_model.py
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing            import MultiLabelBinarizer
from sklearn.model_selection          import train_test_split
from sklearn.neural_network           import MLPClassifier
from sklearn.metrics                  import f1_score, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
JD_DATASET_PATH     = "/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/job_title_des.csv"       # Job Title + Job Description
RESUME_DATASET_PATH = "/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/UpdatedResume.csv"        # Category + Resume
MODEL_PATH          = "models/skill_recommender.keras"
TFIDF_PATH          = "models/tfidf_vectorizer.pkl"
MLB_PATH            = "models/mlb_encoder.pkl"
TOP_SKILLS_PATH     = "models/top_skills.pkl"

MAX_TFIDF_FEATURES  = 5000
TOP_N_SKILLS        = 60
TEST_SIZE           = 0.2
RANDOM_STATE        = 42
MIN_SKILLS_PER_ROW  = 2     # rows with fewer skills will be skipped

os.makedirs("models", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# SKILL LIBRARY  —  pure regex matching, no hardcoded role→skill map
# ──────────────────────────────────────────────────────────────────────────────
SKILL_LIBRARY = [
    # Programming
    "python","java","c++","c#","javascript","typescript","kotlin","swift",
    "go","rust","r","scala","php","ruby","matlab","bash","shell","perl",
    # Web
    "html","css","react","angular","vue","node.js","express","django",
    "flask","fastapi","spring boot","asp.net","bootstrap","tailwind","jquery",
    "wordpress","next.js","nuxt.js","gatsby",
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
    "android","ios","flutter","react native","xamarin",
    # Design / UI-UX
    "figma","adobe xd","sketch","canva","wireframing","prototyping",
    # Testing
    "selenium","pytest","junit","jest","cypress","postman","rest api","graphql",
    # Tools
    "excel","jira","jupyter","maven","gradle","webpack","git",
    # Networking
    "networking","tcp/ip","dns","firewall","vpn","network security","wireshark",
    # Soft Skills
    "communication","leadership","teamwork","problem solving",
    "time management","project management","agile","scrum","critical thinking",
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — pure regex skill extractor (no hardcoding)
# ──────────────────────────────────────────────────────────────────────────────
def extract_skills(text: str) -> list:
    """Raw text se skills extract karo — sirf SKILL_LIBRARY regex use hoga."""
    text_lower = str(text).lower()
    found = []
    for skill in SKILL_LIBRARY:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found.append(skill)
    return list(set(found))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1A — Load job_title_des.csv
# ──────────────────────────────────────────────────────────────────────────────
def load_jd_dataset() -> pd.DataFrame:
    if not os.path.exists(JD_DATASET_PATH):
        print(f"[Load] '{JD_DATASET_PATH}' not found — skipping JD dataset.")
        return pd.DataFrame(columns=["text","source"])

    df = pd.read_csv(JD_DATASET_PATH)
    df = df.rename(columns={"Job Description": "text", "Job Title": "title"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    df["source"] = "jd"

    # Labels: sirf JD text se extract karo
    df["skills"] = df["text"].apply(extract_skills)
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)

    print(f"[Load] JD dataset   : {len(df)} valid rows "
          f"| avg {df['skills'].map(len).mean():.1f} skills/row")
    return df[["text","skills","source"]]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1B — Load UpdatedResume.csv
# ──────────────────────────────────────────────────────────────────────────────
def load_resume_dataset() -> pd.DataFrame:
    if not os.path.exists(RESUME_DATASET_PATH):
        print(f"[Load] '{RESUME_DATASET_PATH}' not found — skipping Resume dataset.")
        return pd.DataFrame(columns=["text","source"])

    df = pd.read_csv(RESUME_DATASET_PATH)

    # Column name flexible handling
    text_col = next((c for c in df.columns if "resume" in c.lower()), None)
    if text_col is None:
        text_col = df.columns[1]

    df = df.rename(columns={text_col: "text"})
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    df["source"] = "resume"

    # Labels: sirf Resume text se extract karo
    df["skills"] = df["text"].apply(extract_skills)
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)

    print(f"[Load] Resume dataset: {len(df)} valid rows "
          f"| avg {df['skills'].map(len).mean():.1f} skills/row")
    return df[["text","skills","source"]]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Combine both datasets
# ──────────────────────────────────────────────────────────────────────────────
def load_combined() -> pd.DataFrame:
    print("\n[Data] Loading datasets ...\n")
    df_jd     = load_jd_dataset()
    df_resume = load_resume_dataset()

    if df_jd.empty and df_resume.empty:
        raise ValueError("Dono datasets missing hain!")

    df = pd.concat([df_jd, df_resume], ignore_index=True)
    df = df.dropna(subset=["text","skills"])
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)

    print(f"\n[Data] Combined total : {len(df)} rows")
    print(f"         JD rows      : {(df['source']=='jd').sum()}")
    print(f"         Resume rows  : {(df['source']=='resume').sum()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — TF-IDF features + MultiLabel targets
# ──────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    # Top N skills by frequency across combined dataset
    all_skills   = [s for row in df["skills"] for s in row]
    skill_counts = Counter(all_skills)
    top_skills   = [sk for sk, _ in skill_counts.most_common(TOP_N_SKILLS)]

    print(f"\n[Features] Top {TOP_N_SKILLS} skills (from actual data — no hardcoding):")
    for i, (sk, cnt) in enumerate(skill_counts.most_common(TOP_N_SKILLS), 1):
        print(f"  {i:>2}. {sk:<35} seen in {cnt} rows")

    # Keep only top skills in labels
    df = df.copy()
    df["skills"] = df["skills"].apply(lambda s: [x for x in s if x in top_skills])
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_ROW].reset_index(drop=True)
    print(f"\n[Features] Rows after top-skill filter: {len(df)}")

    # TF-IDF Vectorizer
    print(f"[Features] Fitting TF-IDF (max_features={MAX_TFIDF_FEATURES}) ...")
    tfidf = TfidfVectorizer(
        max_features = MAX_TFIDF_FEATURES,
        ngram_range  = (1, 2),
        stop_words   = "english",
        sublinear_tf = True,
        min_df       = 2,
    )
    X = tfidf.fit_transform(df["text"]).toarray().astype(np.float32)

    # MultiLabel Binarizer
    mlb = MultiLabelBinarizer(classes=top_skills)
    y   = mlb.fit_transform(df["skills"]).astype(np.float32)

    print(f"[Features] X: {X.shape}  |  y: {y.shape}")
    return X, y, tfidf, mlb, top_skills


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — MLPClassifier  (3 hidden layers, no TensorFlow)
# ──────────────────────────────────────────────────────────────────────────────
def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes  = (512, 256, 128),
        activation          = "relu",
        solver              = "adam",
        alpha               = 1e-4,
        learning_rate_init  = 1e-3,
        max_iter            = 100,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 8,
        random_state        = RANDOM_STATE,
        verbose             = True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train + Evaluate + Save
# ──────────────────────────────────────────────────────────────────────────────
def train():
    print("\n════════ RecruiterVision AI — Skill Model Training ════════")
    print("   Labels: extracted from raw text (zero hardcoding)\n")

    df                           = load_combined()
    X, y, tfidf, mlb, top_skills = build_features(df)

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"\n[Split]  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # Train
    clf = build_model()
    print(f"[Model]  MLPClassifier: 512 → 256 → 128 → {y_train.shape[1]}\n")
    clf.fit(X_train, y_train)

    # Evaluate
    print("\n[Eval]")
    y_pred     = clf.predict(X_test)
    subset_acc = accuracy_score(y_test, y_pred)
    print(f"  Subset Accuracy : {subset_acc:.4f}")
    print(f"\n  Per-skill F1 (top 15 skills):")
    for i, sk in enumerate(mlb.classes_[:15]):
        if y_test[:, i].sum() > 0:
            f1  = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            sup = int(y_test[:, i].sum())
            print(f"    {sk:<35} F1: {f1:.3f}  (test support: {sup})")

    # Save
    with open(MODEL_PATH,      "wb") as f: pickle.dump(clf,        f)
    with open(TFIDF_PATH,      "wb") as f: pickle.dump(tfidf,      f)
    with open(MLB_PATH,        "wb") as f: pickle.dump(mlb,        f)
    with open(TOP_SKILLS_PATH, "wb") as f: pickle.dump(top_skills, f)

    print(f"\n[Saved]  {MODEL_PATH}")
    print(f"[Saved]  {TFIDF_PATH}")
    print(f"[Saved]  {MLB_PATH}")
    print(f"[Saved]  {TOP_SKILLS_PATH}")
    print("\n════════ Training Complete — Ab app.py run karo ════════\n")


if __name__ == "__main__":
    train()