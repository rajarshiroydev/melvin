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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(44)

# Load only first 5000 rows for speed
train_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/train.csv"
df = pd.read_csv(train_path, nrows=5000)

# Separate features and target
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# Handle categorical column f_27
le = LabelEncoder()
X['f_27'] = le.fit_transform(X['f_27'].astype(str))

# Split data (80/20 holdout)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=44, stratify=y
)

# Initialize imputers
imputers = {
    'knn': KNNImputer(n_neighbors=5),
    'iterative': IterativeImputer(max_iter=10, random_state=44),
    'median': SimpleImputer(strategy='median')
}

# Create imputation meta-features
imputation_features_train = pd.DataFrame()
imputation_features_val = pd.DataFrame()

for name, imputer in imputers.items():
    # Fit on training data
    X_imputed_train = imputer.fit_transform(X_train)
    X_imputed_val = imputer.transform(X_val)
    
    # Calculate statistical properties for each sample
    imputation_features_train[f'{name}_mean'] = np.mean(X_imputed_train, axis=1)
    imputation_features_train[f'{name}_std'] = np.std(X_imputed_train, axis=1)
    imputation_features_train[f'{name}_skew'] = pd.DataFrame(X_imputed_train).skew(axis=1).values
    
    imputation_features_val[f'{name}_mean'] = np.mean(X_imputed_val, axis=1)
    imputation_features_val[f'{name}_std'] = np.std(X_imputed_val, axis=1)
    imputation_features_val[f'{name}_skew'] = pd.DataFrame(X_imputed_val).skew(axis=1).values

# Use median imputer for base features (simple and fast)
base_imputer = SimpleImputer(strategy='median')
X_train_base = base_imputer.fit_transform(X_train)
X_val_base = base_imputer.transform(X_val)

# Scale the base features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_base)
X_val_scaled = scaler.transform(X_val_base)

# Combine base features with imputation meta-features
X_train_combined = np.hstack([X_train_scaled, imputation_features_train.values])
X_val_combined = np.hstack([X_val_scaled, imputation_features_val.values])

# Train LightGBM model
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

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=200,
    callbacks=[
        lgb.early_stopping(stopping_rounds=5, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

# Train Logistic Regression on combined features
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=44,
    solver='lbfgs'
)
lr_model.fit(X_train_combined, y_train)

# Get predictions
lgb_preds = lgb_model.predict(X_val_scaled)
lr_preds = lr_model.predict_proba(X_val_combined)[:, 1]

# Simple mean blend (as suggested in strategy)
blended_preds = (0.7 * lgb_preds + 0.3 * lr_preds)

# Calculate final score
score = roc_auc_score(y_val, blended_preds)
print(f"FINAL_SCORE: {score:.4f}")