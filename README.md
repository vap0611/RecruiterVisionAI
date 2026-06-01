🚀 RecruiterVisionAI

An AI-powered Resume Analysis and Skill Recommendation System designed to help candidates improve their ATS compatibility, identify missing skills, and optimize their resumes according to industry requirements.

📌 Overview
RecruiterVisionAI helps job seekers understand why their resumes may be rejected by Applicant Tracking Systems (ATS) and provides actionable recommendations to improve their chances of getting shortlisted.
The system analyzes uploaded resumes, compares them with job descriptions, calculates ATS scores, and recommends missing skills using Natural Language Processing (NLP) and Machine Learning techniques.
🎯 Problem Statement

Studies suggest that nearly 70–75% of resumes are filtered out by ATS software before reaching recruiters.

Common reasons include:

* Missing role-specific keywords
* Poor ATS-friendly formatting
* Incomplete skill representation
* Weak alignment with job descriptions
* Lack of resume optimization

As a result, many qualified candidates lose opportunities despite possessing relevant technical skills.

RecruiterVisionAI addresses this challenge by helping candidates identify gaps and improve their resumes before applying.

✨ Features
* Resume Upload (PDF/DOCX)
* Resume Parsing and Text Extraction
* Skill Extraction
* ATS Compatibility Score
* Job Description Matching
* Missing Skill Detection
* AI-Based Skill Recommendation
* Interactive Dashboard
* Resume Optimization Insights

🏗️ System Architecture

User Resume + Job Description

↓

Text Extraction

↓

Text Preprocessing

↓

Skill Extraction

↓

Word2Vec Embeddings

↓

Random Forest Classification

↓

Skill Recommendation

↓

ATS Score Generation

↓

Dashboard Output

🤖 Machine Learning Models
Model 1: TF-IDF + Cosine Similarity

Purpose:
* Resume-Job Matching
* ATS Score Calculation

Techniques:
* TF-IDF Vectorization
* Cosine Similarity

Output:
* ATS Compatibility Score
* Resume Match Percentage

Model 2: Word2Vec + Random Forest

Purpose:
* Skill Recommendation
* Missing Skill Detection

Techniques:
* Word2Vec Embeddings
* Random Forest Classifier
* Multi-Label Classification

Output:
* Recommended Skills
* Related Skills
* Skill Gap Analysis

🧠 NLP Pipeline
1. Data Collection
2. Text Cleaning
3. Tokenization
4. Skill Extraction
5. Word Embedding Generation
6. Feature Engineering
7. Multi-Label Classification
8. Recommendation Generation

🛠️ Technologies Used

Frontend
* HTML
* CSS
* JavaScript

Backend
* Flask
* Python

Machine Learning
* Scikit-Learn
* Gensim
* Word2Vec
* Random Forest
* TF-IDF
* Cosine Similarity

Data Processing
* Pandas
* NumPy
* Regex

📊 Evaluation Metrics
The model is evaluated using:
* Accuracy Score
* Hamming Loss
* Micro F1 Score
* Macro F1 Score
* Weighted F1 Score

📂 Project Structure
RecruiterVisionAI/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   ├── job_title_des.csv
│   └── UpdatedResumeDataSet.csv
│
├── models/
│   ├── word2vec.model
│   ├── rf_skill_model.pkl
│   └── mlb.pkl
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── login.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
└── README.md



🚧 Limitations
* Depends on dataset quality
* Limited semantic understanding
* Resume formatting variations may affect extraction
* Performance may vary across domains
* Large datasets require more processing time

🔮 Future Scope
* BERT/RoBERTa Integration
* LinkedIn Profile Analysis
* Multilingual Resume Support
* Cloud Deployment
* AI Interview Question Generator
* Real-Time Recruitment Analytics

👨‍💻 Developed By
Vidhi Patel
Computer Engineering Student
LDRP Institute of Technology and Research

📜 License
This project is developed for academic and educational purpose
