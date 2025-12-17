import os
import random
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

# Force CPU to avoid potential CUDA/GPU compatibility issues as originally requested
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ==================== DATA PATHS ====================
TRAIN_PATH = "/home/zeus/.cache/mle-bench/data/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/generated_train.csv"
TEST_PATH = "/home/zeus/.cache/mle-bench/data/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/generated_test.csv"
SAMPLE_SUBMISSION_PATH = "/home/zeus/.cache/mle-bench/data/the-icml-2013-whale-challenge-right-whale-redux/prepared/public/sampleSubmission.csv"

# ==================== CUSTOM DATASET ====================
class AudioDataset:
    def __init__(self, filepaths, labels, sr=22050, duration=2.0, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.samples = int(sr * duration)
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(filepath, sr=self.sr, duration=self.duration)
        except Exception:
            # If loading fails (e.g. file missing), return zeros to maintain shape
            audio = np.zeros(self.samples)
            sr = self.sr
        
        # Pad or truncate to fixed length
        if len(audio) > self.samples:
            audio = audio[:self.samples]
        else:
            audio = np.pad(audio, (0, max(0, self.samples - len(audio))))
        
        # Convert to spectrogram (mel-spectrogram)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Add channel dimension for CNN
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        # Data augmentation (only for training)
        if self.augment and np.random.random() > 0.5:
            try:
                rate = np.random.uniform(0.8, 1.2)
                audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
                if len(audio_stretched) > self.samples:
                    audio_stretched = audio_stretched[:self.samples]
                else:
                    audio_stretched = np.pad(audio_stretched, (0, max(0, self.samples - len(audio_stretched))))
                
                mel_spec = librosa.feature.melspectrogram(y=audio_stretched, sr=sr, n_mels=128, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
            except:
                pass
        
        # Handle cases where we just need data, no labels (test time)
        label = self.labels[idx] if self.labels is not None else 0.0
        
        return mel_spec_db, label

# ==================== MODEL ARCHITECTURES ====================
def create_cnn_model_v1(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_cnn_model_v2(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_cnn_model_v3(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. LOAD DATA INFO
    print("Loading data info...")
    df_train = pd.read_csv(TRAIN_PATH)
    
    # Get input shape from a sample
    dummy_dataset = AudioDataset(df_train['filepath'].values[:1], df_train['label'].values[:1])
    dummy_spec, _ = dummy_dataset[0]
    input_shape = dummy_spec.shape
    
    models_list = []
    
    # 2. CHECK FOR CHECKPOINTS (Crash Recovery)
    if os.path.exists('best_model.weights.h5') or os.path.exists('emergency_model.weights.h5'):
        print("Checkpoint found. Loading models...")
        for model_func in [create_cnn_model_v1, create_cnn_model_v2, create_cnn_model_v3]:
            model = model_func(input_shape)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # Load best weights if available, else emergency
            if os.path.exists('best_model.weights.h5'):
                try:
                    model.load_weights('best_model.weights.h5')
                    models_list.append(model)
                except: pass
            elif os.path.exists('emergency_model.weights.h5'):
                try:
                    model.load_weights('emergency_model.weights.h5')
                    models_list.append(model)
                except: pass
        print(f"Restored {len(models_list)} models.")

    # 3. TRAINING LOOP (if no models loaded)
    if not models_list:
        print("No checkpoints. Starting training...")
        
        # Split
        train_files, val_files, train_labels, val_labels = train_test_split(
            df_train['filepath'].values, df_train['label'].values, 
            test_size=0.1, random_state=42, stratify=df_train['label']
        )
        
        train_dataset = AudioDataset(train_files, train_labels, augment=True)
        val_dataset = AudioDataset(val_files, val_labels, augment=False)
        
        # Load Data into Memory (with timeout safety)
        print("Preparing training data...")
        X_train, y_train = [], []
        for i in range(len(train_dataset)):
            if time.time() - start_time > 1800: # 30 min limit for data prep
                print("Time limit reached during data prep.")
                break
            s, l = train_dataset[i]
            X_train.append(s)
            y_train.append(l)
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        X_val, y_val = [], []
        for i in range(len(val_dataset)):
            s, l = val_dataset[i]
            X_val.append(s)
            y_val.append(l)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Train Ensemble
        if len(X_train) > 0:
            best_auc = 0
            for i, model_func in enumerate([create_cnn_model_v1, create_cnn_model_v2, create_cnn_model_v3]):
                if time.time() - start_time > 2400: # 40 min total limit check
                    print("Time limit approaching. Stopping training.")
                    break
                
                print(f"Training Model {i+1}...")
                model = model_func(input_shape)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])
                
                callbacks_list = [
                    callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True),
                    callbacks.ModelCheckpoint(f'model_{i+1}.weights.h5', monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True)
                ]
                
                model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=15, callbacks=callbacks_list, verbose=1)
                models_list.append(model)
                
                # Update best weights
                val_pred = model.predict(X_val, verbose=0)
                auc = roc_auc_score(y_val, val_pred)
                if auc > best_auc:
                    best_auc = auc
                    model.save_weights('best_model.weights.h5')
                model.save_weights('emergency_model.weights.h5')

    # ==================== TEST PREDICTION ====================
    print("\nGenerating predictions for test set...")
    
    # Load Sample Submission to get the authoritative list of clips
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print(f"Sample submission shape (Target): {sample_sub.shape}")
    
    # Infer test directory from generated_test.csv
    df_test_gen = pd.read_csv(TEST_PATH)
    if len(df_test_gen) > 0:
        test_dir = os.path.dirname(df_test_gen['filepath'].iloc[0])
        print(f"Inferred test directory: {test_dir}")
    else:
        # Fallback: assume standard path structure if generated_test is broken
        test_dir = TRAIN_PATH.replace('generated_train.csv', '').replace('train', 'test')
        print(f"Warning: Generated test empty. Guessing directory: {test_dir}")

    # Construct filepaths for ALL clips in sample submission
    test_filepaths = []
    for clip in sample_sub['clip']:
        test_filepaths.append(os.path.join(test_dir, clip))
        
    # Create dataset for ALL clips
    # AudioDataset handles missing files by returning zeros, preserving the row count
    test_dataset = AudioDataset(test_filepaths, np.zeros(len(test_filepaths)), augment=False)
    
    print(f"Processing {len(test_dataset)} test samples...")
    
    # Load test data
    X_test = []
    # Using a simple loop. AudioDataset handles exceptions internally.
    for i in range(len(test_dataset)):
        spec, _ = test_dataset[i]
        X_test.append(spec)
    X_test = np.array(X_test)
    
    # Predict
    if models_list:
        preds = []
        for model in models_list:
            p = model.predict(X_test, verbose=0, batch_size=128)
            preds.append(p)
        final_preds = np.mean(preds, axis=0).flatten()
    else:
        print("WARNING: No models available. Generating random predictions.")
        final_preds = np.random.uniform(0, 1, len(X_test))
    
    # Create submission matching sample_sub exactly
    submission = pd.DataFrame({
        'clip': sample_sub['clip'],
        'probability': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved.")
    print(f"Final Submission Shape: {submission.shape}")
    print(submission.head())