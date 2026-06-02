# 1. Data Manipulation
import pandas as pd
import numpy as np

# 2. Text Cleaning & Natural Language Processing (NLP)
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 3. Machine Learning Tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download text cleaning packages from NLTK
nltk.download('stopwords')
nltk.download('punkt')

print("🚀 Step 2 Complete: All libraries imported successfully!")

# Load the raw file
file_path = "train_data.txt"

# Read the file and name the columns
train_data = pd.read_csv(
    file_path, 
    sep=":::", 
    engine="python", 
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Clean up any hidden spaces around words
train_data['GENRE'] = train_data['GENRE'].str.strip()
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].str.strip()

# Print confirmation
print(f"🎉 Step 3 Complete! Total movies loaded: {train_data.shape[0]}")

# Preview the first 3 rows of data
print(train_data[['TITLE', 'GENRE', 'DESCRIPTION']].head(3))

# Load the official list of English stopwords we downloaded earlier
stop_words = set(stopwords.words('english'))

def clean_movie_plot(text):
    # 1. Convert text to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers, keeping only alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Split the text into individual words
    words = text.split()
    
    # 4. Filter out stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # 5. Join the cleaned words back into a single string
    return " ".join(filtered_words)

print("Starting to clean the text data... Please hold on a moment!")

# Apply the cleaning function to the DESCRIPTION column
train_data['CLEAN_DESCRIPTION'] = train_data['DESCRIPTION'].apply(clean_movie_plot)

print("🧼 Step 5 Complete: Text cleaning finished successfully!")

# Compare original vs cleaned text for the first movie
print("--- ORIGINAL DESCRIPTION ---")
print(train_data['DESCRIPTION'].iloc[0][:150] + "...")

print("\n--- CLEANED DESCRIPTION ---")
print(train_data['CLEAN_DESCRIPTION'].iloc[0][:150] + "...")

# Initialize the TF-IDF Vectorizer
# max_features=10000 tells it to only keep the top 10,000 most frequent words
tfidf = TfidfVectorizer(max_features=10000)

print("Converting text data into numbers (TF-IDF columns)...")

# Transform the cleaned descriptions into numbers (X matrix)
X = tfidf.fit_transform(train_data['CLEAN_DESCRIPTION'])

# Extract the genres, which are the targets our AI will try to guess (y vector)
y = train_data['GENRE']

print(f"🎉 Step 6 Complete! X shape: {X.shape}")
print("Your text has successfully been converted into a massive math matrix!")

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("🎉 Step 7 Complete!")
print(f"Movies used for teaching the AI: {X_train.shape[0]}")
print(f"Movies held back for testing the AI: {X_val.shape[0]}")

# 1. Initialize the AI brain
classifier = MultinomialNB()

print("🧠 Training the Machine Learning model... This should take just a few seconds!")

# 2. Teach the model using our training math matrix and the correct genres
classifier.fit(X_train, y_train)

print("🎉 Step 8 Complete: Model training is finished!")

# 1. Make predictions on the validation set
y_pred = classifier.predict(X_val)

# 2. Calculate total accuracy
accuracy = accuracy_score(y_val, y_pred)

print("📊 --- MODEL EVALUATION REPORT ---")
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Performance Report per Genre:")
print(classification_report(y_val, y_pred, zero_division=0))

def predict_movie_genre(custom_description):
    # 1. Clean the user's custom input text using our pipeline
    cleaned_input = clean_movie_plot(custom_description)
    
    # 2. Transform the cleaned text into numbers using our fitted TF-IDF translator
    vectorized_input = tfidf.transform([cleaned_input])
    
    # 3. Use the trained AI model to make a prediction
    prediction = classifier.predict(vectorized_input)
    
    print("\n🎬 --- AI MOVIE PREDICTOR ---")
    print(f"Movie Plot Input: \"{custom_description}\"")
    print(f"🤖 Predicted Genre: **{prediction[0].upper()}**")

# ==========================================
# 🚀 Live Interactive Test
# ==========================================

# 1. Ask the user to type their movie idea directly into a text box
user_plot = input("✍️ Enter your custom movie plot summary here: ")

# 2. Feed whatever the user typed straight into our AI brain!
if user_plot.strip():
    predict_movie_genre(user_plot)
else:
    print("❌ You didn't enter any text! Please run the cell again and type a plot.")
