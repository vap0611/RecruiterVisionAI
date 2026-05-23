import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def build_and_train_custom_model():
    print("--> Reading Resume.csv dataset for training...")
    if not os.path.exists("/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/Resume.csv"):
        print("❌ Error: Resume.csv dataset file missing in directory!")
        return

    # Data load karna
    df = pd.read_csv("/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/Resume.csv")
    
    # Missing fields clean karna
    df = df.dropna(subset=['Resume_str', 'Category'])
    
    X = df['Resume_str']  # Input feature (Resume text)
    y = df['Category']    # Target output (Job role labels)

    # 1. Train-Test Split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Text preprocessing aur Feature extraction using TF-IDF Matrix
    print("--> Vectorizing text dataset using TF-IDF algorithm...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 3. Training Custom Logistic Regression Classifier Model
    print("--> Training Custom Machine Learning Classifier Model...")
    custom_clf_model = LogisticRegression(max_iter=300)
    custom_clf_model.fit(X_train_tfidf, y_train)

    # 4. Accuracy evaluation Check
    predictions = custom_clf_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"🚀 TRAINING SUCCESS! Custom Model Validation Accuracy: {round(accuracy * 100, 2)}%")

    # 5. Trained weights ko local files (.pkl) mein save karna
    print("--> Saving trained model components locally...")
    with open("custom_ats_model.pkl", "wb") as model_file:
        pickle.dump(custom_clf_model, model_file)
        
    with open("tfidf_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
        
    print("🎉 System saved: 'custom_ats_model.pkl' and 'tfidf_vectorizer.pkl' are ready!")

if __name__ == "__main__":
    build_and_train_custom_model()