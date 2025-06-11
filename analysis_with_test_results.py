import os
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# Load dataset
DATA_PATH = os.path.join('Datasets', 'healthcare_dataset.csv')

df = pd.read_csv(DATA_PATH)

# Use 'Test Results' as target variable
if 'Test Results' not in df.columns:
    raise ValueError('The dataset does not contain a "Test Results" column.')

df['target'] = df['Test Results']

# Convert date columns to ordinal
for col in ['Date of Admission', 'Discharge Date']:
    df[col] = pd.to_datetime(df[col])
    df[col] = df[col].map(datetime.toordinal)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Exclude target from numeric columns if present
numeric_cols = [col for col in numeric_cols if col != 'target']

categorical_cols = [col for col in df.columns if col not in numeric_cols + ['target']]

# Encode categorical columns with LabelEncoder
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Feature matrix and target vector
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Feature importance dataframe
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
importances.sort_values('Importance', ascending=False, inplace=True)

plt.figure(figsize=(12, 8))
sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')

# SelectKBest on numeric features
selector = SelectKBest(score_func=f_classif, k=min(10, len(numeric_cols)))
X_numeric = df[numeric_cols]
selector.fit(X_numeric, y)
selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]

scores_df = pd.DataFrame({'Feature': numeric_cols, 'Score': selector.scores_})
scores_df.sort_values('Score', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=scores_df, x='Score', y='Feature', palette='rocket')
plt.title('SelectKBest Feature Scores')
plt.tight_layout()
plt.savefig('select_kbest_scores.png')

# Correlation heatmap for numeric features
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# Random Forest with selected features
rf_kbest = RandomForestClassifier(random_state=42)
rf_kbest.fit(X_train[selected_features], y_train)
y_pred_rf_kbest = rf_kbest.predict(X_test[selected_features])
acc_rf_kbest = accuracy_score(y_test, y_pred_rf_kbest)
f1_rf_kbest = f1_score(y_test, y_pred_rf_kbest, average='weighted')

# Try XGBoost if available
acc_xgb = f1_xgb = None
if HAS_XGBOOST:
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

# Compile results
results = [
    {'Model': 'Logistic Regression', 'Feature Selection': 'All', 'Accuracy': acc_lr, 'Weighted F1 Score': f1_lr},
    {'Model': 'Random Forest', 'Feature Selection': 'All', 'Accuracy': acc_rf, 'Weighted F1 Score': f1_rf},
    {'Model': 'Random Forest', 'Feature Selection': 'SelectKBest(10)', 'Accuracy': acc_rf_kbest, 'Weighted F1 Score': f1_rf_kbest},
]
if HAS_XGBOOST:
    results.append({'Model': 'XGBoost', 'Feature Selection': 'All', 'Accuracy': acc_xgb, 'Weighted F1 Score': f1_xgb})

results_df = pd.DataFrame(results)
print(results_df)
print('\nClassification Report (Random Forest):')
print(classification_report(y_test, y_pred_rf))
print('Confusion Matrix (Random Forest):')
print(confusion_matrix(y_test, y_pred_rf))
