import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# No need for SimpleImputer or LabelEncoder for this image-only ViT strategy.
# If a hybrid model were implemented, these would be used for tabular features.

# --- BOILERPLATE ---
# Set random seeds (42).
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration ---
DATASET_PATH = "/home/zeus/.cache/mle-bench/data/siim-isic-melanoma-classification/prepared/public"
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATASET_PATH, "sample_submission.csv") # Not strictly used, but good for reference
IMAGE_DIR = os.path.join(DATASET_PATH, "jpeg", "train") # Directory for training images
TEST_IMAGE_DIR = os.path.join(DATASET_PATH, "jpeg", "test") # Directory for test images
TARGET_COLUMN = "target"

# Model and training parameters
IMG_SIZE = 224  # Standard input size for many ViT models
BATCH_SIZE = 32
NUM_EPOCHS = 15 # Increased for production, early stopping will manage actual epochs
LEARNING_RATE = 1e-5 # Typically smaller for fine-tuning pre-trained models
PATIENCE = 3 # Early stopping patience
MODEL_SAVE_PATH = "best_model.pth" # Path to save the best model checkpoint

# --- Data Loading ---
# Load the ENTIRE dataset for training and validation
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
# df_sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV) # Not directly used for logic

print(f"Full training dataset size: {len(df_train)}")
print(f"Test dataset size: {len(df_test)}")

# --- Custom Dataset Classes ---
class MelanomaDataset(Dataset):
    """
    Dataset class for training and validation data, including labels.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'image_name'] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[idx, TARGET_COLUMN]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class TestMelanomaDataset(Dataset):
    """
    Dataset class for test data, without labels, returning image_name for submission.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'image_name'] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        # Return image and its original name for submission
        return image, self.dataframe.loc[idx, 'image_name']

# --- Image Transformations ---
# Standard transformations for ViT models, including basic augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test transform is typically the same as validation transform
test_transform = val_transform

# --- Data Split ---
# 90/10 Holdout split for training and validation, stratified by target
train_df, val_df = train_test_split(
    df_train, test_size=0.1, random_state=SEED, stratify=df_train[TARGET_COLUMN]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

train_dataset = MelanomaDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = MelanomaDataset(val_df, IMAGE_DIR, transform=val_transform)
test_dataset = TestMelanomaDataset(df_test, TEST_IMAGE_DIR, transform=test_transform)

# Determine number of workers for DataLoader
num_workers = os.cpu_count() // 2 if os.cpu_count() else 2
if num_workers == 0: # Ensure at least one worker if cpu_count is 1
    num_workers = 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

# --- Model Definition ---
# Leverage models pre-trained on very large datasets (e.g., ImageNet-21k, JFT)
# Using a ViT-Base model pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k
model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
# num_classes=1 for binary classification (outputting a single logit)
model = timm.create_model(model_name, pretrained=True, num_classes=1)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Training Setup ---
criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification with raw logits
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop with Early Stopping and Model Checkpointing ---
print(f"Starting training on {device} for up to {NUM_EPOCHS} epochs with patience {PATIENCE}...")
best_val_auc = -1.0 # Initialize with a value lower than any possible AUC
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        # BCEWithLogitsLoss expects target to be same shape as input (N, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply sigmoid to logits to get probabilities
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    # Calculate AUC ROC score
    val_auc = roc_auc_score(val_labels, val_preds)
    print(f"Validation AUC ROC: {val_auc:.4f}")

    # Check for model checkpointing and early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0 # Reset patience since validation AUC improved
        torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save the best model
        print(f"New best validation AUC ({best_val_auc:.4f}). Model saved to {MODEL_SAVE_PATH}")
    else:
        patience_counter += 1
        print(f"Validation AUC did not improve. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# --- Load Best Model for Prediction ---
print(f"Loading best model from {MODEL_SAVE_PATH} for final predictions...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device) # Ensure model is on the correct device after loading
model.eval() # Set model to evaluation mode

# --- Prediction on Test Data ---
print("Generating predictions on test data...")
test_preds = []
test_image_names = []

with torch.no_grad():
    for inputs, img_names in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        test_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
        test_image_names.extend(img_names)

# --- Create Submission File ---
submission_df = pd.DataFrame({'image_name': test_image_names, 'target': test_preds})

# Ensure ID column types match sample_submission.csv (image_name is string, target is float)
submission_df['target'] = submission_df['target'].astype(float)

# Save to submission.csv
submission_file_path = "submission.csv"
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file created at {submission_file_path}")
# Final score output matching the prototype's format
print(f"FINAL_SCORE: {best_val_auc:.4f}")