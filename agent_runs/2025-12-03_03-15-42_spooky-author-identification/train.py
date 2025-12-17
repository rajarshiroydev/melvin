import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset # Using Hugging Face datasets library for convenience

# --- Boilerplate: Set random seeds ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 44
set_seed(SEED)

# --- Configuration ---
DATASET_PATH = "/home/zeus/.cache/mle-bench/data/spooky-author-identification/prepared/public"
TRAIN_FILE = os.path.join(DATASET_PATH, "train.csv")
TEST_FILE = os.path.join(DATASET_PATH, "test.csv") # Path to the test dataset
SAMPLE_SUBMISSION_FILE = os.path.join(DATASET_PATH, "sample_submission.csv") # Path to sample submission for format
SUBMISSION_FILE = "submission.csv" # Name of the output submission file

TARGET_COLUMN = "author"
TEXT_COLUMN = "text"
ID_COLUMN = "id" # Identifier column in test.csv

# Model choice: DistilBERT is a good balance of performance and speed
MODEL_NAME = "distilbert-base-uncased" 
MAX_SEQ_LENGTH = 384 # Max sequence length for tokenizer (IncreasedSeqLength384 as per winning strategy)
BATCH_SIZE = 16      # Batch size for training and evaluation
LEARNING_RATE = 2e-5 # Learning rate for fine-tuning
NUM_TRAIN_EPOCHS = 10 # Max epochs, EarlyStopping will prevent overtraining
EARLY_STOPPING_PATIENCE = 3 # Patience for EarlyStoppingCallback

# --- Load Data ---
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
except FileNotFoundError as e:
    print(f"Error: Data file not found. Please ensure files are in {DATASET_PATH}. {e}")
    exit()

# --- Subsampling (REMOVED for full dataset training) ---
# The original subsampling logic is removed as per "Load the ENTIRE dataset" instruction.
print(f"Using full training dataset of {len(df_train)} rows.")

# --- Handle Missing Values ---
# For text, fill NaN with an empty string to prevent tokenizer errors
imputer_text = SimpleImputer(strategy='constant', fill_value='')

# Apply to training data
df_train[TEXT_COLUMN] = imputer_text.fit_transform(df_train[[TEXT_COLUMN]]).squeeze()
# Apply to test data (using the same imputer fitted on train)
df_test[TEXT_COLUMN] = imputer_text.transform(df_test[[TEXT_COLUMN]]).squeeze()

# For target, drop rows with missing target values if any (LabelEncoder would fail)
df_train.dropna(subset=[TARGET_COLUMN], inplace=True)

# --- Encode Target Labels ---
label_encoder = LabelEncoder()
df_train['labels'] = label_encoder.fit_transform(df_train[TARGET_COLUMN])
num_labels = len(label_encoder.classes_)
print(f"Detected {num_labels} unique authors: {label_encoder.classes_}")

# --- Split Data (CRITICAL CONSTRAINT: 90/10 Holdout for final training) ---
# Changed from 80/20 to 90/10 for final production training to maximize training data
train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=SEED, stratify=df_train['labels'])
print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """Tokenizes input text using the pre-trained tokenizer."""
    return tokenizer(examples[TEXT_COLUMN], truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH)

# Convert pandas DataFrames to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df[[TEXT_COLUMN, 'labels']])
val_dataset = Dataset.from_pandas(val_df[[TEXT_COLUMN, 'labels']])
test_dataset = Dataset.from_pandas(df_test[[TEXT_COLUMN, ID_COLUMN]]) # Test dataset for prediction

# Apply tokenization to datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove original text column and set format for PyTorch
tokenized_train_dataset = tokenized_train_dataset.remove_columns([TEXT_COLUMN])
tokenized_val_dataset = tokenized_val_dataset.remove_columns([TEXT_COLUMN])
# For test dataset, remove text column but keep ID
tokenized_test_dataset = tokenized_test_dataset.remove_columns([TEXT_COLUMN])

tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch") # Set format for test dataset

# --- Load Model ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# --- Define Metrics (Multi-class Log Loss) ---
def compute_metrics(p):
    """Computes multi-class logarithmic loss for evaluation."""
    # p.predictions can be a tuple if the model outputs more than just logits
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    
    # Apply softmax to logits to get probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    
    # Calculate log loss. The 'labels' argument ensures correct mapping of probabilities.
    logloss = log_loss(labels, probabilities, labels=np.arange(num_labels))
    
    return {"log_loss": logloss}

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results", # Directory to save checkpoints and model outputs
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save model at the end of each epoch
    save_total_limit=1,          # Only keep the best model checkpoint (CRITICAL for robustness)
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs", # Directory for logs
    logging_steps=10,     # Log every 10 steps
    load_best_model_at_end=True, # Load the best model (based on metric_for_best_model) at the end of training
    metric_for_best_model="log_loss", # Monitor custom log_loss for best model selection
    greater_is_better=False, # Lower log_loss is better
    seed=SEED,
    data_seed=SEED,
    report_to="none", # Disable reporting to external services like W&B
)

# --- Trainer Initialization ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer, # Pass tokenizer to Trainer for proper padding/truncation during batching
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] # CRITICAL CONSTRAINT: Early Stopping
)

# --- Train Model ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- Evaluate Model on Validation Set ---
print("Evaluating model on validation set...")
eval_results = trainer.evaluate()
final_log_loss = eval_results["eval_log_loss"]

# --- Output Final Score (CRITICAL CONSTRAINT) ---
print(f"FINAL_SCORE: {final_log_loss:.4f}")

# --- Prediction on Test Data and Submission Generation ---
print("Making predictions on test data using the best model...")
# The trainer automatically loads the best model at the end of training due to load_best_model_at_end=True
predictions = trainer.predict(tokenized_test_dataset)

# Extract logits from predictions
test_logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions

# Apply softmax to get probabilities
test_probabilities = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()

# Create submission DataFrame
submission_df = pd.DataFrame({ID_COLUMN: df_test[ID_COLUMN]})

# Map probabilities to author names based on label_encoder.classes_ order
# The label_encoder.classes_ provides the sorted order of author names (e.g., EAP, HPL, MWS).
# The probabilities in test_probabilities[:, i] correspond to this order.
for i, class_name in enumerate(label_encoder.classes_):
    submission_df[class_name] = test_probabilities[:, i]

# Ensure ID column type matches sample_submission (e.g., int64)
submission_df[ID_COLUMN] = df_test[ID_COLUMN].astype(df_sample_submission[ID_COLUMN].dtype)

# Reorder columns to match sample_submission.csv format (ID, EAP, HPL, MWS)
# Assuming sample_submission columns are ID, then author probabilities in a specific order.
# We extract the author columns from sample_submission to ensure correct order.
author_cols_in_submission = [col for col in df_sample_submission.columns if col != ID_COLUMN]
final_submission_cols = [ID_COLUMN] + author_cols_in_submission
submission_df = submission_df[final_submission_cols]

# Save submission file
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file '{SUBMISSION_FILE}' created successfully.")