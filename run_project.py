import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

print("HR Culture Insight Analyzer Starting...")

# -------------------------------

# Load Sentiment Model

# -------------------------------

sentiment_model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

print("Sentiment Model Loaded")

# -------------------------------

# Load Emotion Model

# -------------------------------

emotion_model = load_model("emotion_model.keras")

print("Emotion Model Loaded")

# -------------------------------

# Load HR Dataset

# -------------------------------

hr = pd.read_csv("Uncleaned_employees_final_dataset.csv")

print("HR Dataset Loaded")

# -------------------------------

# Calculate HR Performance Score

# -------------------------------

hr["performance_score"] = (
0.35 * (hr["previous_year_rating"] / hr["previous_year_rating"].max()) +
0.25 * hr["KPIs_met_more_than_80"] +
0.20 * (hr["avg_training_score"] / hr["avg_training_score"].max()) +
0.10 * (hr["no_of_trainings"] / hr["no_of_trainings"].max()) +
0.10 * hr["awards_won"]
)

performance_score = hr["performance_score"].mean()

print("Performance Score:", performance_score)

# -------------------------------

# Sentiment Prediction

# -------------------------------

feedback = ["Management communication is very poor"]

feedback_vec = vectorizer.transform(feedback)

sentiment_pred = sentiment_model.predict(feedback_vec)

sentiment_score = float(sentiment_pred[0])

print("Sentiment Score:", sentiment_score)

# -------------------------------

# Emotion Score (Demo Value)

# -------------------------------

emotion_score = 0.7

print("Emotion Score:", emotion_score)

# -------------------------------

# Final Culture Score

# -------------------------------

culture_score = (
0.35 * emotion_score +
0.35 * sentiment_score +
0.30 * performance_score
)

print("Final Culture Score:", culture_score)

# -------------------------------

# Culture Interpretation

# -------------------------------

if culture_score >= 0.7:
culture_status = "Excellent Work Culture"
elif culture_score >= 0.5:
culture_status = "Moderate Work Culture"
else:
culture_status = "Poor Work Culture"

print("Culture Status:", culture_status)
