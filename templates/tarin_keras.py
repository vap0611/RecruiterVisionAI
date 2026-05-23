import pandas as pd
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_keras_deep_learning_model():
    print("==================================================")
    print("🧠 KERAS DEEP LEARNING MODEL TRAINING PIPELINE 🧠")
    print("==================================================")
    
    csv_path = "/home/vidhi/Desktop/Python PGMS/RecruiterVisionAI/dataset/job_title_des.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset file not found at {csv_path}")
        return

    # 1. Load Dataset
    df = pd.read_csv(csv_path).dropna(subset=['Job Description'])
    descriptions = df['Job Description'].tolist()

    # ---- MACHINE LEARNING TRAIN-TEST SPLIT (80-20 Rule) ----
    train_data, test_data = train_test_split(descriptions, test_size=0.2, random_state=42)
    print(f"--> Split Done! Train size: {len(train_data)}, Test size: {len(test_data)}")

    # ---- TF-IDF VECTORIZATION (Input Features for Neural Network) ----
    # Neural network ko numerical matrix chahiye hota hai, isliye TF-IDF fit kar rahe hain
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(train_data).toarray()
    X_test_tfidf = vectorizer.transform(test_data).toarray()

    input_dim = X_train_tfidf.shape[1] # Matrix size (2000 dimensions)

    # ---- 2. KERAS NEURAL NETWORK ARCHITECTURE ----
    print("--> Building Keras Sequential Neural Network...")
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'), # Hidden Layer 1
        Dropout(0.3),                                       # Overfitting rokne ke liye
        Dense(64, activation='relu'),                       # Hidden Layer 2
        Dense(input_dim, activation='sigmoid')              # Output Layer (Multi-label skill representation)
    ])

    # Model Compile karna
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ---- 3. TRAINING MODE ----
    print("--> Training Deep Learning Model (TensorFlow Backend)...")
    # Epochs=5 rakha hai taaki training fast ho jaye, batch_size=32 rows ek sath process karega
    model.fit(X_train_tfidf, X_train_tfidf, epochs=5, batch_size=32, validation_data=(X_test_tfidf, X_test_tfidf))

    # ---- 4. SAVE MODEL ARTIFACTS ----
    print("\n--> Saving trained Keras model and vectorizer layers locally...")
    
    # Keras Neural Network native format mein save karna
    model.save("keras_skill_model.keras")
    
    # Vectorizer vocabulary ko save karna taaki app.py mein dimensions match ho sakein
    with open("keras_vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("🎉 SUCCESS! 'keras_skill_model.keras' and 'keras_vectorizer.pkl' are saved and ready!")

if __name__ == "__main__":
    train_keras_deep_learning_model()