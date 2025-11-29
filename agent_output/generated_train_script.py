import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATASET_LOCATION = "/Users/rajarshiroy/Library/Caches/mle-bench/data/leaf-classification/prepared/public"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.05  # CRITICAL: Use only 5% of training data for CPU
NUM_EPOCHS = 3         # CRITICAL: Train for a maximum of 3 epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
train_csv_path = os.path.join(DATASET_LOCATION, "train.csv")
test_csv_path = os.path.join(DATASET_LOCATION, "test.csv")
sample_submission_path = os.path.join(DATASET_LOCATION, "sample_submission.csv")
images_dir = os.path.join(DATASET_LOCATION, "images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_submission_df = pd.read_csv(sample_submission_path)

# CRITICAL: Subset training data
print(f"Original training data size: {len(train_df)}")
train_df = train_df.sample(frac=TRAIN_FRACTION, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Subsetting training data to {TRAIN_FRACTION*100}%: {len(train_df)} samples")

# --- Label Encoding ---
label_encoder = LabelEncoder()
train_df['species_encoded'] = label_encoder.fit_transform(train_df['species'])
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# --- Custom Dataset Class ---
class LeafDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_encoder=None, is_test=False):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id']
        # FIX: Cast image ID to integer to prevent float formatting like '123.0.jpg'
        img_name = os.path.join(self.img_dir, f"{int(img_id)}.jpg")
        
        # Ensure image exists and is valid
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found for ID {img_id} at {img_name}. Skipping.")
            # Return a dummy image and label, or handle error appropriately
            # For this script, we'll raise an error to indicate a critical issue
            raise FileNotFoundError(f"Image file not found for ID {img_id} at {img_name}")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}. Skipping.")
            raise e

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_id
        else:
            label = self.dataframe.iloc[idx]['species_encoded']
            return image, label

# --- Image Transforms ---
# Standard transforms for ResNet
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- DataLoaders ---
train_dataset = LeafDataset(dataframe=train_df, img_dir=images_dir, transform=image_transforms, is_test=False)
test_dataset = LeafDataset(dataframe=test_df, img_dir=images_dir, transform=image_transforms, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Model Definition ---
# Use the recommended 'weights' parameter instead of 'pretrained'
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Modify the final fully connected layer for our number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# --- Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) # CRITICAL: Use torch.optim.AdamW

# --- Training Loop ---
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

print("Training complete.")

# --- Prediction ---
print("Making predictions on the test set...")
model.eval()
test_ids = []
all_predictions = []

with torch.no_grad():
    for inputs, ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        
        test_ids.extend(ids.tolist())
        all_predictions.extend(probabilities.cpu().numpy())

# --- Create Submission File ---
submission_df = pd.DataFrame({'id': test_ids})
class_columns = label_encoder.classes_
predictions_df = pd.DataFrame(all_predictions, columns=class_columns)

# Merge predictions with IDs
submission_df = pd.merge(submission_df, predictions_df, left_index=True, right_index=True)

# Ensure all classes from sample_submission are present, fill with 0 if not predicted
# This handles cases where a class might not appear in the 5% subset of training data
for col in sample_submission_df.columns:
    if col not in submission_df.columns and col != 'id':
        submission_df[col] = 0.0

# Reorder columns to match sample_submission_df
submission_df = submission_df[['id'] + list(sample_submission_df.columns[1:])]

# Save submission file to the current working directory
submission_output_path = "submission.csv"
submission_df.to_csv(submission_output_path, index=False)

print(f"Submission file created at: {os.path.abspath(submission_output_path)}")
print(f"Submission file head:\n{submission_df.head()}")