import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission_df = pd.read_csv("sample_submission.csv")

# Define target column and classes
target_col = "author"
classes = ['EAP', 'HPL', 'MWS']

# Encode target labels
label_encoder = LabelEncoder()
train_df[target_col + '_encoded'] = label_encoder.fit_transform(train_df[target_col])

# Feature Engineering (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['text'])

y_train = train_df[target_col + '_encoded']

# Model Training (Logistic Regression)
model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, C=1.0)
model.fit(X_train_tfidf, y_train)

# Generate predictions
predictions_proba = model.predict_proba(X_test_tfidf)

# Create submission DataFrame
submission_df = pd.DataFrame(predictions_proba, columns=label_encoder.classes_)
submission_df['id'] = test_df['id']

# Reorder columns to match sample_submission.csv
submission_df = submission_df[['id'] + list(sample_submission_df.columns[1:])]

# Save submission file
submission_df.to_csv("submission.csv", index=False)