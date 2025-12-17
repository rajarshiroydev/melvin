import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. Load Data ---
# Use the specified dataset path.
DATASET_PATH = "/home/zeus/.cache/mle-bench/data/spooky-author-identification/prepared/public"
train_df_path = f"{DATASET_PATH}/train.csv"
test_df_path = f"{DATASET_PATH}/test.csv"
sample_submission_df_path = f"{DATASET_PATH}/sample_submission.csv"

# Load the ENTIRE training dataset.
df_train = pd.read_csv(train_df_path)
# Load the test dataset for final predictions.
df_test = pd.read_csv(test_df_path)
# Load sample submission for format reference.
df_sample_submission = pd.read_csv(sample_submission_df_path)

# --- 2. Preprocessing ---
text_column = "text"
target_column = "author"

# Handle potential missing values in the text column by filling with an empty string.
# This is robust for tokenizers.
df_train[text_column] = df_train[text_column].fillna("")
df_test[text_column] = df_test[text_column].fillna("")

# Encode target labels (author names) into numerical IDs.
label_encoder = LabelEncoder()
df_train["labels"] = label_encoder.fit_transform(df_train[target_column])
num_labels = len(label_encoder.classes_)

# Map author names to their encoded IDs for submission column ordering.
# This ensures the probability columns in the submission match the order of label_encoder.classes_.
author_to_id = {author: id for id, author in enumerate(label_encoder.classes_)}
# Sort authors by their encoded ID to ensure consistent column order in submission.
sorted_authors = sorted(author_to_id, key=author_to_id.get)

# Split data into training and validation sets (90/10 split as required).
# Stratify to maintain the class distribution in both sets.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train[text_column].tolist(),
    df_train["labels"].tolist(),
    test_size=0.1,  # 90% Train, 10% Validation
    random_state=42,
    stratify=df_train["labels"]
)

# --- 3. Model Selection and Tokenization ---
# Choose a pre-trained model. DistilBERT is a good choice for speed and performance.
model_name = "distilbert-base-uncased"

# Load the tokenizer and the model for sequence classification.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define a tokenization function.
# Truncation ensures texts fit the model's max input length.
# Padding ensures all sequences have the same length.
# max_length=128 is chosen for efficiency and consistency with the prototype.
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Create Hugging Face Dataset objects from the split data.
train_dataset_dict = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset_dict = Dataset.from_dict({"text": val_texts, "labels": val_labels})
# Create a dataset for the test data (no labels needed for prediction).
test_dataset_dict = Dataset.from_dict({"text": df_test[text_column].tolist(), "id": df_test["id"].tolist()})

# Apply tokenization to all datasets.
tokenized_train_dataset = train_dataset_dict.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset_dict.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset_dict.map(tokenize_function, batched=True)

# Remove the original 'text' column as it's no longer needed after tokenization.
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text"])
# For test dataset, keep 'id' but remove 'text'.
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])

# Set the format of the datasets to PyTorch tensors.
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")

# --- 4. Training Arguments and Trainer Setup ---
output_dir = "./results_spooky_author_production"
os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,  # Set a higher number of epochs, Early Stopping will manage actual epochs.
    per_device_train_batch_size=16, # Batch size for training. Adjust based on GPU memory.
    per_device_eval_batch_size=16,  # Batch size for evaluation.
    warmup_steps=500,               # Number of steps for the learning rate scheduler warmup.
    weight_decay=0.01,              # L2 regularization.
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",       # Log metrics at the end of each epoch.
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch.
    save_strategy="epoch",          # Save model checkpoint at the end of each epoch.
    load_best_model_at_end=True,    # Load the best model (based on `metric_for_best_model`) at the end of training.
    metric_for_best_model="eval_log_loss", # Monitor custom log loss for best model selection.
    greater_is_better=False,        # For log loss, a lower value is better.
    learning_rate=1e-5,             # "LowerLR" strategy: 1e-5 is a common and effective low LR for fine-tuning.
    seed=42,                        # Random seed for reproducibility.
    fp16=torch.cuda.is_available(), # Enable mixed-precision training if a GPU is available for speed.
    report_to="none",               # Disable reporting to external services like Weights & Biases.
    save_total_limit=1,             # Only save the best model checkpoint to save disk space.
)

# Define the compute_metrics function to calculate multi-class logarithmic loss.
def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids

    # Apply softmax to logits to get probabilities.
    probabilities = torch.softmax(torch.tensor(predictions), dim=-1).numpy()

    # Clip probabilities to avoid log(0) or log(1) issues, as required by log_loss.
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    # Calculate log loss using sklearn's implementation.
    ll = log_loss(labels, probabilities)
    return {"log_loss": ll} # The key will be prefixed with 'eval_' by Trainer (e.g., 'eval_log_loss').

# Initialize the Hugging Face Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # Implement Early Stopping: stop training if validation log loss doesn't improve for 3 epochs.
    # The metric to monitor is specified in TrainingArguments (metric_for_best_model).
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# --- 5. Training ---
trainer.train()

# --- 6. Evaluation ---
# The Trainer automatically loads the best model (based on eval_log_loss) at the end.
# Evaluate this best model on the validation set.
eval_results = trainer.evaluate()

# Extract the final log loss from the evaluation results.
final_log_loss = eval_results["eval_log_loss"]

# Print the final score in the required format.
print(f"FINAL_SCORE: {final_log_loss:.4f}")

# --- 7. Generate Submission ---
print("Generating submission file...")

# Predict probabilities on the test dataset using the best model.
predictions = trainer.predict(tokenized_test_dataset)

# The predictions object contains logits. Apply softmax to get probabilities.
test_probabilities = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Create the submission DataFrame.
submission_df = pd.DataFrame({'id': df_test['id']})

# Add probability columns, ensuring the order matches label_encoder.classes_.
for i, author_name in enumerate(sorted_authors):
    submission_df[author_name] = test_probabilities[:, i]

# Ensure 'id' column type matches sample_submission.csv (usually int).
submission_df['id'] = submission_df['id'].astype(df_sample_submission['id'].dtype)

# Save the submission file.
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")