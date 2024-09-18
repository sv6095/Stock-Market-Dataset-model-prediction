import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.svm import SVC, LinearSVC

# Data Reading and Exploration
data = pd.read_csv(r'C:\Users\shant\Downloads\Second Year\nifty_500.csv')#include your file path

# Convert columns to numeric, handling non-numeric values (including '-')
for col in ['Change', 'Percentage Change', '365 Day Percentage Change', '30 Day Percentage Change']:
    # Replace '-' with '-1' before conversion
    data[col] = data[col].astype(str).str.replace('-', '-1', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Check unique values in categorical columns
print("\nUnique Industries:\n", data['Industry'].unique())
print("\nUnique Series:\n", data['Series'].unique())

# Data Preprocessing

# Handle missing values (improved handling)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])  # Mode for categorical
    else:
        data[col] = data[col].fillna(data[col].median())  # Median for numerical

# Encode categorical variables
data['Industry'] = LabelEncoder().fit_transform(data['Industry'])
data['Series'] = LabelEncoder().fit_transform(data['Series'])

# Regression Model Building and Evaluation

# Define features and target
X_reg = data.drop(['Last Traded Price', 'Company Name', 'Symbol'], axis=1)
y_reg = data['Last Traded Price']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train RandomForest model
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_regressor.predict(X_test_reg)

# Evaluate
print('\nRegression Evaluation:')
print(f'MAE: {mean_absolute_error(y_test_reg, y_pred_reg)}')
print(f'R-squared: {r2_score(y_test_reg, y_pred_reg)}')

# Create Target for Classification
data['Target'] = (data['Percentage Change'] > 0).astype(int)

# Classification Model Building and Evaluation

# Define features and target
X_clf = data.drop(['Target', 'Company Name', 'Symbol', 'Percentage Change'], axis=1)
y_clf = data['Target']

# Split data (stratified) for initial classification models
X_train_clf_initial, X_test_clf_initial, y_train_clf_initial, y_test_clf_initial = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Initialize and train classifiers (directly on imbalanced data)
svm_classifier = SVC(random_state=42)
lsvm_classifier = LinearSVC(random_state=42, max_iter=50000) 
svm_classifier.fit(X_train_clf_initial, y_train_clf_initial) 
lsvm_classifier.fit(X_train_clf_initial, y_train_clf_initial) 

# Make predictions
y_pred_svm = svm_classifier.predict(X_test_clf_initial)
y_pred_lsvm = lsvm_classifier.predict(X_test_clf_initial)

# Evaluation function (reusable)
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nEvaluation Metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=1):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

# Evaluate models
evaluate_model(y_test_clf_initial, y_pred_svm, "SVM")
evaluate_model(y_test_clf_initial, y_pred_lsvm, "Linear SVM")

# Random Forest Models

# Random Forest for Regression 
# Split data for Random Forest regressor (Use a different random_state)
X_train_reg_rf, X_test_reg_rf, y_train_reg_rf, y_test_reg_rf = train_test_split(X_reg, y_reg, test_size=0.2, random_state=123) 

rf_regressor_new = RandomForestRegressor(random_state=42) 
rf_regressor_new.fit(X_train_reg_rf, y_train_reg_rf)
y_pred_reg_rf = rf_regressor_new.predict(X_test_reg_rf)

print('\nEvaluation Metrics for Random Forest Regression:')
print(f'MAE: {mean_absolute_error(y_test_reg_rf, y_pred_reg_rf)}')
print(f'R-squared: {r2_score(y_test_reg_rf, y_pred_reg_rf)}')

# Random Forest for Classification

# Define features and target 
X_clf_rf = data.drop(['Target', 'Company Name', 'Symbol', 'Percentage Change'], axis=1)
y_clf_rf = data['Target']

# Split data (stratified) for Random Forest Classifier
X_train_clf_rf, X_test_clf_rf, y_train_clf_rf, y_test_clf_rf = train_test_split(
    X_clf_rf, y_clf_rf, test_size=0.2, random_state=123, stratify=y_clf_rf 
)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_clf_rf, y_train_clf_rf) 
y_pred_clf_rf = rf_classifier.predict(X_test_clf_rf)

evaluate_model(y_test_clf_rf, y_pred_clf_rf, "Random Forest Classifier")
