import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
import os

# Suppress specific LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Set random seeds for reproducibility
np.random.seed(42)

# --- 1. Load Data ---
# CRITICAL CONSTRAINT: Load the ENTIRE dataset.
train_data_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/train.csv"
test_data_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/test.csv"
sample_submission_path = "/home/zeus/.cache/mle-bench/data/tabular-playground-series-may-2022/prepared/public/sample_submission.csv"

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
sample_submission_df = pd.read_csv(sample_submission_path)

TARGET = 'target'
ID_COL = 'id'

# Separate target from features in training data
X_full = train_df.drop(columns=[TARGET, ID_COL])
y_full = train_df[TARGET]

# Store test IDs for submission
test_ids = test_df[ID_COL]
X_test_raw = test_df.drop(columns=[ID_COL])

# --- 2. Global Transformers and Feature Engineering Setup ---

# Global LabelEncoders for f_27 characters (pre-fit for robustness)
# This ensures all possible characters ('A' through 'J') are encoded consistently
# across training and test sets, preventing issues with unseen categories.
global_label_encoders = {}
if 'f_27' in X_full.columns:
    for i in range(10): # f_27 has 10 characters
        char_col_name = f'f_27_char_{i}'
        # Collect all unique characters for this position from both train and test
        all_chars_for_pos = pd.concat([X_full['f_27'].str[i], X_test_raw['f_27'].str[i]]).unique()
        le = LabelEncoder()
        le.fit(all_chars_for_pos)
        global_label_encoders[char_col_name] = le

# Global Imputer for numerical features
# This will be fitted on the full training data and then used to transform both
# training and test data, preventing data leakage.
imputer_numerical = SimpleImputer(strategy='mean')

# List of original integer columns (before f_27 processing)
# This helps in identifying categorical features for LightGBM later.
original_integer_cols = [f for f in X_full.columns if X_full[f].dtype == 'int64']

def apply_feature_engineering(df_input, is_training=True, df_for_agg_fit=None):
    """
    Applies the defined feature engineering steps to a DataFrame.
    
    Args:
        df_input (pd.DataFrame): The DataFrame to apply feature engineering to.
        is_training (bool): True if processing training data, False for test data.
                            Controls fitting of transformers.
        df_for_agg_fit (pd.DataFrame, optional): DataFrame to use for fitting
                                                 group-based aggregations (e.g., the full
                                                 processed training set for test data).
                                                 Defaults to None, in which case df_input is used.
    Returns:
        pd.DataFrame: The DataFrame with applied feature engineering.
    """
    df = df_input.copy()

    # Handle 'f_27' (object type) - string feature engineering
    if 'f_27' in df.columns:
        df['f_27_len'] = df['f_27'].apply(len)
        df['f_27_unique_chars'] = df['f_27'].apply(lambda x: len(set(x)))
        
        for i in range(10):
            char_col_name = f'f_27_char_{i}'
            # FIX: First, extract the character into the new column
            df[char_col_name] = df['f_27'].str[i] 

            if char_col_name in global_label_encoders:
                le = global_label_encoders[char_col_name]
                # Now, apply the label encoding to the newly created column
                # Pre-fitting on combined train/test chars should ensure all values are known.
                # If an unseen char appears (highly unlikely here), it would raise an error.
                df[char_col_name] = df[char_col_name].map(lambda s: le.transform([s])[0])
            else:
                raise ValueError(f"Global LabelEncoder for {char_col_name} not found. Ensure it's pre-fitted.")
        
        df = df.drop('f_27', axis=1)

    # --- Extensive Feature Engineering ---
    # Polynomial Features (degree 2 for a selection of float features)
    poly_features_base = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 
                          'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
    poly_features_base = [col for col in poly_features_base if col in df.columns]

    for col in poly_features_base:
        df[f'{col}_sq'] = df[col]**2

    # Interaction Features (products, sums, and differences of selected pairs)
    interaction_pairs = [
        ('f_00', 'f_01'), ('f_00', 'f_02'), ('f_01', 'f_02'),
        ('f_03', 'f_04'), ('f_05', 'f_06'),
        ('f_19', 'f_20'), ('f_21', 'f_22'),
        ('f_00', 'f_07'), ('f_01', 'f_08'), ('f_02', 'f_09'),
        ('f_28', 'f_29'), ('f_28', 'f_30'),
        ('f_00', 'f_28'), ('f_01', 'f_29')
    ]
    for col1, col2 in interaction_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    # Statistical Aggregations (mean, std, min, max) across different groups
    agg_groups = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_29', 'f_30']
    agg_groups = [col for col in agg_groups if col in df.columns]

    agg_features = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 
                    'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
    agg_features = [col for col in agg_features if col in df.columns]

    # Determine the source DataFrame for calculating aggregations.
    # For training data, aggregations are calculated on the current df.
    # For test data, aggregations are calculated on the full processed training data (df_for_agg_fit)
    # to prevent data leakage.
    source_df_for_agg = df_for_agg_fit if df_for_agg_fit is not None else df

    for group_col in agg_groups:
        for agg_col in agg_features:
            # Calculate aggregations on the source_df_for_agg
            agg_stats = source_df_for_agg.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max'])
            agg_stats.columns = [f'{agg_col}_{stat}_by_{group_col}' for stat in ['mean', 'std', 'min', 'max']]
            
            # Merge these aggregations into the current df
            df = df.merge(agg_stats, on=group_col, how='left')

    # Impute any remaining NaNs (especially from aggregations for unseen groups in test)
    # This step is now performed *after* all feature engineering, ensuring the imputer
    # is fitted/transformed on the complete set of features.
    final_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if is_training:
        # Fit and transform on training data for all numerical columns
        df[final_numerical_cols] = imputer_numerical.fit_transform(df[final_numerical_cols])
    else:
        # Transform test data using imputer fitted on training data for all numerical columns
        df[final_numerical_cols] = imputer_numerical.transform(df[final_numerical_cols])
        
    return df

# --- Apply Feature Engineering to Full Training and Test Data ---
# First, process the full training data. This fits the imputer and provides the
# base for group aggregations for the test set.
X_full_processed = apply_feature_engineering(X_full, is_training=True)

# Then, process the raw test data. Crucially, pass X_full_processed as df_for_agg_fit
# to ensure group aggregations are derived solely from the training data.
X_test_processed = apply_feature_engineering(X_test_raw, is_training=False, df_for_agg_fit=X_full_processed)

# Align columns between processed training and test sets.
# This is critical to ensure the model receives consistent input features.
train_cols = X_full_processed.columns
test_cols = X_test_processed.columns

# Add columns missing in test_processed, filling with 0 (or a more appropriate default)
missing_in_test = list(set(train_cols) - set(test_cols))
for col in missing_in_test:
    X_test_processed[col] = 0 

# Remove columns present in test_processed but not in train_processed
missing_in_train = list(set(test_cols) - set(train_cols))
if missing_in_train:
    warnings.warn(f"Columns {missing_in_train} present in test but not in train. Dropping from test.")
    X_test_processed = X_test_processed.drop(columns=missing_in_train)

# Reorder test columns to match train columns exactly
X_test_processed = X_test_processed[train_cols]


# --- 3. Train-Validation Split ---
# CRITICAL CONSTRAINT: Use a simple 90/10 Holdout split.
X_train, X_val, y_train, y_val = train_test_split(X_full_processed, y_full, test_size=0.1, random_state=42, stratify=y_full)

# --- 4. LightGBM Model Training ---

# Identify categorical features for LightGBM's native handling.
# This includes original integer columns and the new f_27_char_i features.
categorical_features_for_lgbm = original_integer_cols.copy()
for i in range(10):
    char_col_name = f'f_27_char_{i}'
    if char_col_name in X_full_processed.columns and char_col_name not in categorical_features_for_lgbm:
        categorical_features_for_lgbm.append(char_col_name)

# Ensure these columns are explicitly cast to 'category' dtype for LightGBM.
for col in categorical_features_for_lgbm:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
    if col in X_val.columns:
        X_val[col] = X_val[col].astype('category')
    if col in X_test_processed.columns:
        X_test_processed[col] = X_test_processed[col].astype('category')

# Initialize LightGBM Classifier with recommended hyperparameters.
# n_estimators is set high, but early stopping will determine the actual number of rounds.
lgbm = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          n_estimators=2000, # Max iterations, early stopping will cut this short
                          learning_rate=0.03, # Slightly reduced learning rate for more robust training
                          num_leaves=31,      # Controls complexity of individual trees
                          max_depth=-1,       # No limit on tree depth
                          reg_alpha=0.1,      # L1 regularization term on weights
                          reg_lambda=0.1,     # L2 regularization term on weights
                          colsample_bytree=0.8, # Fraction of features to consider at each split
                          subsample=0.8,      # Fraction of data to sample for each tree
                          random_state=42,
                          n_jobs=-1,          # Use all available cores
                          verbose=-1          # Suppress verbose output during training
                         )

# CRITICAL CONSTRAINT: Implement Early Stopping (patience=3)
# LightGBM's early_stopping callback automatically saves the best model (best iteration)
# and uses it for prediction if num_iteration is specified.
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         eval_metric='auc',
         callbacks=[lgb.early_stopping(3, verbose=False)], # patience=3
         categorical_feature=categorical_features_for_lgbm
        )

# --- 5. Prediction and Evaluation on Validation Set ---
# Use the best iteration found by early stopping for validation prediction.
y_pred_proba_val = lgbm.predict_proba(X_val, num_iteration=lgbm.best_iteration_)[:, 1]
auc_score_val = roc_auc_score(y_val, y_pred_proba_val)

# --- 6. Generate Submission ---
# Predict on the preprocessed test data using the best model (best_iteration_).
y_pred_proba_test = lgbm.predict_proba(X_test_processed, num_iteration=lgbm.best_iteration_)[:, 1]

# Create submission DataFrame
submission_df = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: y_pred_proba_test
})

# Ensure ID column type matches sample_submission.csv (usually int)
submission_df[ID_COL] = submission_df[ID_COL].astype(sample_submission_df[ID_COL].dtype)

# Save to submission.csv
submission_output_path = "submission.csv"
submission_df.to_csv(submission_output_path, index=False)

# --- 7. Output Final Score ---
# CRITICAL CONSTRAINT: Print the final score on the LAST LINE in the specified format.
# This is the validation AUC score, as requested for the "FINAL_SCORE".
print(f"FINAL_SCORE: {auc_score_val:.4f}")