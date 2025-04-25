import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration
DATA_PKL = 'preprocessed/processed_data_v3.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load data
with open(DATA_PKL, 'rb') as f:
    trials = pickle.load(f)

# Flatten epochs across all trials
X_list, y_list = [], []
for trial in trials:
    for seg, label in zip(trial['X'], trial['y']):
        X_list.append(seg.flatten())
        y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Feature scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)

# Train linear SVM with balanced class weights
clf = SVC(kernel='linear', class_weight='balanced', random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Save model and scaler
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/svm_p300.pkl')
joblib.dump(scaler, 'models/svm_scaler.pkl')
print("Saved SVM model and scaler to models/") 
