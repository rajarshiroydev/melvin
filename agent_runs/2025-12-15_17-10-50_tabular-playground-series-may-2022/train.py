import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(44)

# Load entire dataset
train_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/train.csv"
test_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/test.csv"
sample_sub_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/sample_submission.csv"

print("Loading data...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

# Separate features and target
X = df_train.drop(['id', 'target'], axis=1)
y = df_train['target']
X_test = df_test.drop(['id'], axis=1)

# Handle categorical column f_27 - Fit on combined data to avoid unseen labels
le = LabelEncoder()
combined_f27 = pd.concat([X['f_27'], X_test['f_27']], axis=0).astype(str)
le.fit(combined_f27)

X['f_27'] = le.transform(X['f_27'].astype(str))
X_test['f_27'] = le.transform(X_test['f_27'].astype(str))

# Split data (90/10 holdout for validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=44, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Initialize imputers
imputers = {
    'knn': KNNImputer(n_neighbors=5),
    'iterative': IterativeImputer(max_iter=10, random_state=44),
    'median': SimpleImputer(strategy='median')
}

# Create imputation meta-features for training
imputation_features_train = pd.DataFrame()
imputation_features_val = pd.DataFrame()
imputation_features_test = pd.DataFrame()

for name, imputer in imputers.items():
    # Fit on training data
    X_imputed_train = imputer.fit_transform(X_train)
    X_imputed_val = imputer.transform(X_val)
    X_imputed_test = imputer.transform(X_test)
    
    # Calculate statistical properties for each sample
    imputation_features_train[f'{name}_mean'] = np.mean(X_imputed_train, axis=1)
    imputation_features_train[f'{name}_std'] = np.std(X_imputed_train, axis=1)
    imputation_features_train[f'{name}_skew'] = pd.DataFrame(X_imputed_train).skew(axis=1).values
    
    imputation_features_val[f'{name}_mean'] = np.mean(X_imputed_val, axis=1)
    imputation_features_val[f'{name}_std'] = np.std(X_imputed_val, axis=1)
    imputation_features_val[f'{name}_skew'] = pd.DataFrame(X_imputed_val).skew(axis=1).values
    
    imputation_features_test[f'{name}_mean'] = np.mean(X_imputed_test, axis=1)
    imputation_features_test[f'{name}_std'] = np.std(X_imputed_test, axis=1)
    imputation_features_test[f'{name}_skew'] = pd.DataFrame(X_imputed_test).skew(axis=1).values

# Use median imputer for base features (simple and fast)
base_imputer = SimpleImputer(strategy='median')
X_train_base = base_imputer.fit_transform(X_train)
X_val_base = base_imputer.transform(X_val)
X_test_base = base_imputer.transform(X_test)

# Scale the base features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_base)
X_val_scaled = scaler.transform(X_val_base)
X_test_scaled = scaler.transform(X_test_base)

# Combine base features with imputation meta-features
X_train_combined = np.hstack([X_train_scaled, imputation_features_train.values])
X_val_combined = np.hstack([X_val_scaled, imputation_features_val.values])
X_test_combined = np.hstack([X_test_scaled, imputation_features_test.values])

# Save preprocessing objects
os.makedirs('models', exist_ok=True)
joblib.dump(base_imputer, 'models/base_imputer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Train LightGBM model with early stopping and checkpointing
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 44
}

print("\nTraining LightGBM model...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# Save best LightGBM model
lgb_model.save_model('models/lgbm_best_model.pkl')

# Train Logistic Regression on combined features
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=44,
    solver='lbfgs'
)
lr_model.fit(X_train_combined, y_train)

# Save Logistic Regression model
joblib.dump(lr_model, 'models/logistic_regression.pkl')

# Validate on validation set
print("\nEvaluating on validation set...")
lgb_preds_val = lgb_model.predict(X_val_scaled)
lr_preds_val = lr_model.predict_proba(X_val_combined)[:, 1]
blended_preds_val = (0.7 * lgb_preds_val + 0.3 * lr_preds_val)

val_score = roc_auc_score(y_val, blended_preds_val)
print(f"Validation Score: {val_score:.4f}")

# Generate predictions on test set
print("\nGenerating test predictions...")
lgb_preds_test = lgb_model.predict(X_test_scaled)
lr_preds_test = lr_model.predict_proba(X_test_combined)[:, 1]
blended_preds_test = (0.7 * lgb_preds_test + 0.3 * lr_preds_test)

# Create submission file
submission = pd.DataFrame({
    'id': df_test['id'],
    'target': blended_preds_test
})

# Ensure ID column types match sample submission
submission['id'] = submission['id'].astype(sample_sub['id'].dtype)

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to 'submission.csv'")
print(f"Submission shape: {submission.shape}")
print(f"Submission head:\n{submission.head()}")