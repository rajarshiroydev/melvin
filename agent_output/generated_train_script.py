import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Dataset location
DATASET_LOCATION = "/Users/rajarshiroy/Library/Caches/mle-bench/data/spooky-author-identification/prepared/public"

# Construct file paths
train_file_path = Path(DATASET_LOCATION) / "train.csv"
test_file_path = Path(DATASET_LOCATION) / "test.csv"
sample_submission_file_path = Path(DATASET_LOCATION) / "sample_submission.csv"

# Load data
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
sample_submission_df = pd.read_csv(sample_submission_file_path)

# Define target and text columns
target_col = "author"
text_col = "text"
classes = ['EAP', 'HPL', 'MWS']

# Prepare data for TF-IDF
X_train_text = train_df[text_col]
X_test_text = test_df[text_col]
y_train = train_df[target_col]

# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

# Transform test data
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Initialize and train Logistic Regression model
model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train_encoded)

# Generate predictions (probabilities)
test_predictions_proba = model.predict_proba(X_test_tfidf)

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id']})

# Ensure column order matches sample_submission.csv
# Map encoded classes back to original class names for column assignment
predicted_class_names = label_encoder.inverse_transform(model.classes_)
for i, class_name in enumerate(predicted_class_names):
    submission_df[class_name] = test_predictions_proba[:, i]

# Reorder columns to match sample_submission_df
submission_cols = ['id'] + list(sample_submission_df.columns[1:])
submission_df = submission_df[submission_cols]

# Write submission.csv to the current working directory
submission_df.to_csv("submission.csv", index=False)