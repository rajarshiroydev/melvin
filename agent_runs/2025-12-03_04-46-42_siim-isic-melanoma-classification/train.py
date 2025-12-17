import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# --- Configuration ---
SEED = 43
DATASET_PATH = "/home/zeus/.cache/mle-bench/data/siim-isic-melanoma-classification/prepared/public"

# Full dataset paths
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATASET_PATH, "sample_submission.csv")

IMAGE_DIR_TRAIN = os.path.join(DATASET_PATH, "jpeg", "train")
IMAGE_DIR_TEST = os.path.join(DATASET_PATH, "jpeg", "test")

TARGET_COLUMN = "target"
# Tabular features to use (excluding image_name, patient_id, and target-related columns like diagnosis, benign_malignant)
TABULAR_FEATURES = ["sex", "age_approx", "anatom_site_general_challenge"]

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20 # Increased epochs for full dataset, early stopping will manage
PATIENCE = 3 # Early stopping patience
VALIDATION_SPLIT_RATIO = 0.1 # 90% Train, 10% Validation

MODEL_SAVE_PATH = "best_multimodal_model.pth" # Path to save the best model checkpoint

# --- Set Random Seeds ---
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Data Loading (Full Dataset) ---
print(f"Loading full training data from {TRAIN_CSV}...")
df_full = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df_full)} rows for training.")

# Load test data for later prediction
print(f"Loading test data from {TEST_CSV}...")
test_df_raw = pd.read_csv(TEST_CSV)
print(f"Loaded {len(test_df_raw)} rows for testing.")

# --- 2. Train-Validation Split ---
# Split the original dataframe into train and validation parts
train_df, val_df = train_test_split(
    df_full,
    test_size=VALIDATION_SPLIT_RATIO,
    random_state=SEED,
    stratify=df_full[TARGET_COLUMN] # Stratify to maintain target distribution
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Train dataset size: {len(train_df)}")
print(f"Validation dataset size: {len(val_df)}")

# --- 3. Preprocessing ---

# Identify categorical and numerical features from the full training dataframe
numerical_features = df_full[TABULAR_FEATURES].select_dtypes(include=np.number).columns.tolist()
categorical_features = df_full[TABULAR_FEATURES].select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit preprocessor ONLY on the training data and transform train, validation, and test
X_tabular_train_processed = preprocessor.fit_transform(train_df[TABULAR_FEATURES])
X_tabular_val_processed = preprocessor.transform(val_df[TABULAR_FEATURES])
X_tabular_test_processed = preprocessor.transform(test_df_raw[TABULAR_FEATURES]) # Transform test data

tabular_input_dim = X_tabular_train_processed.shape[1]
print(f"Tabular features preprocessed. Input dimension for MLP: {tabular_input_dim}")

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Standard input size for many CNNs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# --- Custom Dataset ---
class MelanomaDataset(Dataset):
    def __init__(self, dataframe, image_dir, tabular_data_processed, transform=None, is_test=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.tabular_data_processed = tabular_data_processed # Already preprocessed
        self.transform = transform
        self.image_names = dataframe["image_name"].values
        self.is_test = is_test
        if not is_test:
            self.targets = dataframe[TARGET_COLUMN].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Use the already processed tabular data
        tabular_features = torch.tensor(self.tabular_data_processed[idx], dtype=torch.float32)

        if self.is_test:
            return image, tabular_features
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, tabular_features, target

train_dataset = MelanomaDataset(train_df, IMAGE_DIR_TRAIN, X_tabular_train_processed, image_transform)
val_dataset = MelanomaDataset(val_df, IMAGE_DIR_TRAIN, X_tabular_val_processed, image_transform)
test_dataset = MelanomaDataset(test_df_raw, IMAGE_DIR_TEST, X_tabular_test_processed, image_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 2)

print(f"Train DataLoader size: {len(train_loader.dataset)}")
print(f"Validation DataLoader size: {len(val_loader.dataset)}")
print(f"Test DataLoader size: {len(test_loader.dataset)}")

# --- 4. Model Architecture (Multimodal Fusion) ---
class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, num_classes=1):
        super(MultimodalModel, self).__init__()

        # CNN Branch (Pre-trained ResNet18)
        self.cnn_model = models.resnet18(pretrained=True)
        # Freeze all parameters in the CNN backbone
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 as per instruction
        for param in self.cnn_model.layer4.parameters():
            param.requires_grad = True
        # Replace the final classification layer to get features
        num_ftrs = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity() # Remove the final layer to get features

        # MLP Branch for tabular data
        self.mlp_branch = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Freeze MLP branch parameters as per instruction "within these modules only"
        for param in self.mlp_branch.parameters():
            param.requires_grad = False

        # Fusion and Classification Head
        # The output of ResNet18's fc layer (if Identity) is num_ftrs
        # The output of MLP branch is 64
        fusion_input_dim = num_ftrs + 64
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) # Output a single logit for binary classification
        )
        # Parameters of self.classification_head are new and thus requires_grad=True by default.
        # No explicit unfreezing loop is needed here.

    def forward(self, image_input, tabular_input):
        # CNN branch
        cnn_features = self.cnn_model(image_input)

        # MLP branch
        mlp_features = self.mlp_branch(tabular_input)

        # Concatenate features
        fused_features = torch.cat((cnn_features, mlp_features), dim=1)

        # Classification head
        output = self.classification_head(fused_features)
        return output

model = MultimodalModel(tabular_input_dim=tabular_input_dim, num_classes=1).to(DEVICE)

# --- 5. Training Setup ---
criterion = nn.BCEWithLogitsLoss() # For binary classification, handles sigmoid internally
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. Training Loop with Early Stopping and Model Checkpointing ---
best_val_auc = -1.0
patience_counter = 0
final_score = 0.0 # Will store the best validation AUC

print("Starting training...")
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, tabular_data, targets in train_loader:
        images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, tabular_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular_data, targets in val_loader:
            images, tabular_data, targets = images.to(DEVICE), tabular_data.to(DEVICE), targets.to(DEVICE).unsqueeze(1)

            outputs = model(images, tabular_data)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    val_loss = val_loss / len(val_loader.dataset)
    
    # Handle cases where all_targets might be all same class in a small batch/split
    if len(np.unique(all_targets)) > 1:
        val_auc = roc_auc_score(all_targets, all_preds)
    else:
        val_auc = 0.5 # AUC is undefined or 0.5 for single-class predictions
        print("Warning: Validation set contains only one class. AUC set to 0.5.")


    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Early Stopping and Model Checkpointing
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save the best model state
        print(f"Model saved to {MODEL_SAVE_PATH} with improved Val AUC: {best_val_auc:.4f}")
        final_score = best_val_auc # Update final score with the best AUC
    else:
        patience_counter += 1
        print(f"EarlyStopping patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print(f"Training complete. Best Validation AUC: {final_score:.4f}")

# --- 7. Prediction on Test Data ---
print("Loading best model for final predictions...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval() # Set model to evaluation mode

test_preds = []
test_image_names = test_df_raw["image_name"].values

print("Generating predictions on the test set...")
with torch.no_grad():
    for images, tabular_data in test_loader:
        images, tabular_data = images.to(DEVICE), tabular_data.to(DEVICE)
        outputs = model(images, tabular_data)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(preds.flatten())

# --- 8. Create Submission File ---
submission_df = pd.DataFrame({'image_name': test_image_names, 'target': test_preds})

# Ensure ID column types match sample_submission.csv (image_name is string, target is float)
# This is usually handled by pandas automatically, but good to note.
# submission_df['image_name'] = submission_df['image_name'].astype(str) # Already string
# submission_df['target'] = submission_df['target'].astype(float) # Already float

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

# --- Final Score Output (for benchmarking systems) ---
print(f"FINAL_SCORE: {final_score:.4f}")