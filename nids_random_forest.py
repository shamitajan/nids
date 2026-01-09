"""
AI-Based Network Intrusion Detection System
Dataset: NSL-KDD
Model: Random Forest Classifier
Task: Binary Classification (Normal vs Attack)
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -----------------------------
# Step 1: Load Dataset
# -----------------------------

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"
]

url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
data = pd.read_csv(url, names=columns)

# Drop difficulty level column
data.drop("difficulty_level", axis=1, inplace=True)


# -----------------------------
# Step 2: Label Encoding
# -----------------------------

# Binary classification
# normal -> 0
# attack -> 1
data["label"] = data["attack_type"].apply(lambda x: 0 if x == "normal" else 1)
data.drop("attack_type", axis=1, inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = label_encoder.fit_transform(data[col])


# -----------------------------
# Step 3: Feature Scaling
# -----------------------------

X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# Step 4: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# -----------------------------
# Step 5: Model Training
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------
# Step 6: Evaluation
# -----------------------------

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
