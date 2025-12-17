import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Start timer for time limit
start_time = time.time()
TIME_LIMIT = 1800  # 60 minutes

# --- CRITICAL: DATASET PATHS ---
TRAIN_PATH = "/home/zeus/.cache/mle-bench/data/text-normalization-challenge-english-language/prepared/public/en_train.csv"
TEST_PATH = "/home/zeus/.cache/mle-bench/data/text-normalization-challenge-english-language/prepared/public/en_test_2.csv"
SAMPLE_SUBMISSION_PATH = "/home/zeus/.cache/mle-bench/data/text-normalization-challenge-english-language/prepared/public/en_sample_submission_2.csv"

# Load entire dataset
print("Loading full dataset...")
df = pd.read_csv(TRAIN_PATH)
print(f"Loaded {len(df)} samples")

# Prepare data
df = df[['before', 'after']].dropna()
df['before'] = df['before'].astype(str)
df['after'] = df['after'].astype(str)

# 80/20 holdout split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Custom Dataset
class TextNormalizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        before = str(self.data.iloc[idx]['before'])
        after = str(self.data.iloc[idx]['after'])
        
        # Tokenize input
        input_encoding = self.tokenizer(
            before,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            after,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

# Initialize model and tokenizer
print("Initializing model...")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Create datasets and dataloaders
train_dataset = TextNormalizationDataset(train_df, tokenizer)
val_dataset = TextNormalizationDataset(val_df, tokenizer)

# Use 8 workers for DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# Training setup
optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 3

# Training loop with time limit check
print("Starting training...")
time_limit_reached = False

for epoch in range(num_epochs):
    if time_limit_reached:
        break
        
    model.train()
    total_train_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Check time limit
        if time.time() - start_time > TIME_LIMIT:
            print(f"\nTime limit reached! Stopping training...")
            time_limit_reached = True
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        
        if batch_idx % 500 == 0: # Reduce print frequency
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    if time_limit_reached:
        break
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation (skip if time limit reached)
    if not time_limit_reached:
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)
        
        torch.save(model.state_dict(), 'best_model.pt')

# Save emergency model if time limit reached
if time_limit_reached:
    print("Saving emergency model...")
    torch.save(model.state_dict(), 'emergency_model.pt')

# Load model for prediction
print("\nLoading model for prediction...")
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt'))
    print("Loaded best_model.pt")
elif os.path.exists('emergency_model.pt'):
    model.load_state_dict(torch.load('emergency_model.pt'))
    print("Loaded emergency_model.pt")
else:
    print("Using current model state")

model.eval()

# Load test data
print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# --- FIX: ROBUST ID CONSTRUCTION ---
# Check if 'id' column exists, if not construct it
if 'id' not in test_df.columns:
    print("'id' column not found in test file. Constructing from sentence_id and token_id...")
    if 'sentence_id' in test_df.columns and 'token_id' in test_df.columns:
        test_df['id'] = test_df['sentence_id'].astype(str) + "_" + test_df['token_id'].astype(str)
    else:
        print("Warning: Could not construct ID. Using index.")
        test_df['id'] = test_df.index
# -----------------------------------

# Ensure columns exist and fill NA
if 'before' not in test_df.columns:
    raise ValueError("Test data missing 'before' column!")
test_df['before'] = test_df['before'].fillna("")

# Prepare test dataset
class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        before = str(self.data.iloc[idx]['before'])
        
        input_encoding = self.tokenizer(
            before,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # We rely on the dataloader maintaining order, but return idx for safety if needed
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'idx': idx 
        }

test_dataset = TestDataset(test_df, tokenizer)
# Reduced num_workers to 0 for prediction to prevent worker crashes/overhead on small batches
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# Generate predictions
print("Generating predictions...")
predictions = []

with torch.no_grad():
    # ADD TQDM HERE TO SEE PROGRESS
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=1 # Greedy search for speed
        )
        
        batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(batch_predictions)


# Create submission dataframe using the Constructed ID from test_df
submission_df = pd.DataFrame({
    'id': test_df['id'], # Use the ID from the test dataframe we fixed
    'after': predictions
})

# Save submission
submission_df.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to submission.csv with {len(submission_df)} predictions")
print(f"Total time: {time.time() - start_time:.2f} seconds")