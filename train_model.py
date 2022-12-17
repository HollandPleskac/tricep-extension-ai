import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
import pickle

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict


landmarks_csv = pd.read_csv("landmarks.csv")

# Split into X & y
X = landmarks_csv.drop("class", axis=1)
y = landmarks_csv["class"]

# Proprocessing
le = preprocessing.LabelEncoder()
le.fit(["up", "down"])
y = le.transform(y)

# print( X)
print(X)

# Split into train & test
np.random.seed(42)  # seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# use X_train.values to not give labels to the model

# Make & fit baseline model
clf = RandomForestClassifier()
clf.fit(X_train.values, y_train)

# Make baseline predictions
y_preds = clf.predict(X_test.values)

# Evaluate the classifier on validation set
baseline_metrics = evaluate_preds(y_test, y_preds)

# Save the model
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)