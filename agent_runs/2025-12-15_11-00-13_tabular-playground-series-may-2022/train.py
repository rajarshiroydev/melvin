import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
import os
import joblib

# Set random seed
np.random.seed(42)

# Create directory for model checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Load data - FULL DATASET
print("Loading data...")
df = pd.read_csv('/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/train.csv')
print(f"Loaded {len(df)} rows")

# Separate features and target
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}  # Store label encoders for each column
if len(categorical_cols) > 0:
    print(f"Encoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Handle missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-validation split (90% train, 10% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

# Initialize models with different random seeds for diversity
models = {
    'lgb1': lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    ),
    'lgb2': lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        random_state=123,
        n_jobs=-1
    ),
    'xgb1': xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    ),
    'xgb2': xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        random_state=123,
        n_jobs=-1,
        eval_metric='auc'
    ),
    'cat1': cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=False
    ),
    'cat2': cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        random_state=123,
        verbose=False
    )
}

# 5-fold stratified KFold for ensemble
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store predictions and model weights
val_predictions = np.zeros((len(X_val), len(models)))
model_weights = {}
best_models = {}

print("\nTraining ensemble models with 5-fold CV...")

for model_idx, (model_name, model) in enumerate(models.items()):
    print(f"\nTraining {model_name}...")
    fold_scores = []
    
    # Train with KFold to get CV scores
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if 'lgb' in model_name:
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=False)]
            )
        elif 'xgb' in model_name:
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
        else:  # catboost
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                verbose=False
            )
        
        # Get predictions and calculate AUC
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    # Calculate average CV score and store weight
    avg_cv_score = np.mean(fold_scores)
    model_weights[model_name] = avg_cv_score
    print(f"  Average CV AUC: {avg_cv_score:.5f}")
    
    # Train final model on full training set with early stopping and checkpointing
    print(f"  Training final {model_name} on full training set...")
    
    if 'lgb' in model_name:
        # LightGBM with early stopping and best model saving
        callbacks = [
            lgb.early_stopping(stopping_rounds=3, verbose=False),
            lgb.record_evaluation({})
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        # Save best model
        joblib.dump(model, f'checkpoints/{model_name}_best.pkl')
        best_models[model_name] = joblib.load(f'checkpoints/{model_name}_best.pkl')
        
    elif 'xgb' in model_name:
        # XGBoost with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        # Save best model
        joblib.dump(model, f'checkpoints/{model_name}_best.pkl')
        best_models[model_name] = joblib.load(f'checkpoints/{model_name}_best.pkl')
        
    else:  # catboost
        # CatBoost with early stopping
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        # Save best model
        joblib.dump(model, f'checkpoints/{model_name}_best.pkl')
        best_models[model_name] = joblib.load(f'checkpoints/{model_name}_best.pkl')
    
    # Get predictions on validation set using best model
    val_predictions[:, model_idx] = best_models[model_name].predict_proba(X_val)[:, 1]
    print(f"  Saved best {model_name} model")

# Normalize weights
total_weight = sum(model_weights.values())
model_weights = {k: v/total_weight for k, v in model_weights.items()}

print("\nModel Weights (based on CV performance):")
for model_name, weight in model_weights.items():
    print(f"  {model_name}: {weight:.4f}")

# Weighted average of probabilities
print("\nCalculating weighted ensemble predictions...")
weighted_predictions = np.zeros(len(X_val))
for model_idx, (model_name, weight) in enumerate(model_weights.items()):
    weighted_predictions += val_predictions[:, model_idx] * weight

# Calculate final ensemble AUC
ensemble_auc = roc_auc_score(y_val, weighted_predictions)
print(f"\nEnsemble AUC on validation set: {ensemble_auc:.5f}")

# Print final score in required format
print(f"FINAL_SCORE: {ensemble_auc:.5f}")

# Load test data
print("\nLoading test data...")
test_df = pd.read_csv('/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/test.csv')
test_ids = test_df['id'].copy()

# Preprocess test data
X_test = test_df.drop(['id'], axis=1)

# Handle categorical features
if len(categorical_cols) > 0:
    print(f"Encoding test categorical features...")
    for col in categorical_cols:
        if col in X_test.columns:
            le = label_encoders[col]
            # Handle unseen labels by mapping them to a default value
            unique_train_values = set(le.classes_)
            test_values = X_test[col].astype(str).unique()
            unseen_values = set(test_values) - unique_train_values
            
            if len(unseen_values) > 0:
                print(f"  Found {len(unseen_values)} unseen values in {col}, mapping to default...")
                # Replace unseen values with the most common value from training
                X_test[col] = X_test[col].astype(str).apply(
                    lambda x: x if x in unique_train_values else le.classes_[0]
                )
            
            X_test[col] = le.transform(X_test[col])

# Handle missing values
print("Imputing test missing values...")
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Generate predictions using best models
print("\nGenerating predictions on test set...")
test_predictions = np.zeros((len(X_test), len(best_models)))

for model_idx, (model_name, model) in enumerate(best_models.items()):
    print(f"  Predicting with {model_name}...")
    test_predictions[:, model_idx] = model.predict_proba(X_test)[:, 1]

# Weighted average of probabilities for final predictions
final_predictions = np.zeros(len(X_test))
for model_idx, (model_name, weight) in enumerate(model_weights.items()):
    final_predictions += test_predictions[:, model_idx] * weight

# Create submission file
print("\nCreating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'target': final_predictions
})

# Ensure ID column type matches
submission['id'] = submission['id'].astype(int)

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"Submission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Submission preview:\n{submission.head()}")