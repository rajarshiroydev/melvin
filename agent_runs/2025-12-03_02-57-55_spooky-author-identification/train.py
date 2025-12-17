import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset # Hugging Face datasets library for convenience
import os # For managing output directories

# Set random seeds for reproducibility
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For transformers library
    from transformers import set_seed as hf_set_seed
    hf_set_seed(seed)

set_seed(43)

# --- 1. Load Data (FULL DATASET) ---
data_path = "/home/zeus/.cache/mle-bench/data/spooky-author-identification/prepared/public"
train_df = pd.read_csv(f"{data_path}/train.csv")
test_df = pd.read_csv(f"{data_path}/test.csv")
sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")

# --- 2. Preprocessing ---
# Handle missing values for the text column by filling with an empty string.
train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

# Encode target labels (author names) into numerical IDs.
label_encoder = LabelEncoder()
train_df['author_encoded'] = label_encoder.fit_transform(train_df['author'])
num_labels = len(label_encoder.classes_)

# Split data into training and validation sets (e.g., 90/10 holdout).
# This is CRITICAL for validation metrics to prevent overfitting.
# Stratify ensures that the class distribution is maintained in both splits.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(),
    train_df['author_encoded'].tolist(),
    test_size=0.1, # 10% for validation
    random_state=43,
    stratify=train_df['author_encoded']
)

# Prepare test texts for prediction
test_texts = test_df['text'].tolist()

# --- 3. Model and Tokenizer Initialization ---
# Using DistilBERT for a good balance of speed and performance.
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# --- 4. Tokenization ---
# Tokenize texts, ensuring truncation and padding to a fixed maximum length.
max_seq_length = 128

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_seq_length)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=max_seq_length)
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=max_seq_length)

# --- 5. Create Hugging Face Dataset objects ---
# Convert tokenized data and labels into Hugging Face Dataset format.
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})
# Test dataset does not have labels
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask']
})

# --- 6. Define Metrics Function ---
# This function computes the multi-class logarithmic loss, as required by the competition.
def compute_metrics(p):
    predictions = p.predictions # These are raw logits from the model
    labels = p.label_ids

    # Convert logits to probabilities using softmax.
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1, keepdims=True)
    
    # Clip probabilities to avoid log(0) or log(1) issues, as specified in Kaggle's evaluation.
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # Calculate multi-class logarithmic loss using sklearn's implementation.
    return {"log_loss": log_loss(labels, probabilities)}

# --- 7. Configure Training Arguments and Trainer ---
# Create a directory for model checkpoints
output_dir = './model_checkpoints'
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20, # Set a higher number of epochs; EarlyStopping will manage.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5, # A commonly used small learning rate for fine-tuning transformers.
    warmup_steps=50, # Number of steps for the learning rate scheduler warmup.
    weight_decay=0.01, # L2 regularization.
    logging_dir='./logs',
    logging_steps=50, # Log training metrics more frequently.
    evaluation_strategy="epoch", # Evaluate on the validation set at the end of each epoch.
    save_strategy="epoch",       # Save model checkpoint at the end of each epoch.
    load_best_model_at_end=True, # Load the best model (based on `metric_for_best_model`) when training finishes.
    metric_for_best_model="log_loss", # Metric to monitor for early stopping and best model selection.
    greater_is_better=False,     # For log_loss, a lower value is better.
    save_total_limit=1,          # Only save the best model checkpoint to save disk space.
    seed=43, # Random seed for reproducibility.
    report_to="none", # Disable reporting to external services like Weights & Biases for simplicity.
)

# Early Stopping Callback: Stop training if validation log_loss does not improve for 3 consecutive epochs.
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer, # Pass tokenizer to Trainer for potential dynamic padding.
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# --- 8. Train the Model ---
trainer.train()

# --- 9. Evaluate and Print Final Score ---
# Evaluate the best model (loaded at the end of training due to load_best_model_at_end=True)
# on the validation set.
eval_results = trainer.evaluate()
final_log_loss = eval_results["eval_log_loss"]

# Print the final score in the required format.
print(f"FINAL_SCORE: {final_log_loss:.4f}")

# --- 10. Predict on Test Data and Generate Submission ---
# The trainer automatically uses the best model loaded at the end of training
# for prediction, thanks to `load_best_model_at_end=True`.
test_predictions_output = trainer.predict(test_dataset)
test_logits = test_predictions_output.predictions

# Convert raw logits to probabilities using softmax.
test_probabilities = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=-1, keepdims=True)

# Clip probabilities to avoid log(0) or log(1) issues, matching competition requirements.
epsilon = 1e-15
test_probabilities = np.clip(test_probabilities, epsilon, 1 - epsilon)

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id']})

# Get author names in the order they were encoded by LabelEncoder.
# This ensures the columns in the submission file match the expected order.
author_columns = label_encoder.classes_

# Assign probabilities to the corresponding author columns.
submission_df[author_columns] = test_probabilities

# Ensure ID column type matches sample_submission.csv (typically int).
submission_df['id'] = submission_df['id'].astype(sample_submission_df['id'].dtype)

# Save to submission.csv
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' generated successfully.")