import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('processed.cleveland.data')

datas=data.rename(columns={'num\t':'target'})

datas.replace('?', np.nan, inplace=True)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
datas[['ca', 'thal']] = imputer.fit_transform(datas[['ca', 'thal']])

datas["ca"] = datas["ca"].astype(float).astype(int)
datas["chol"] = datas["chol"].astype(float).astype(int)
datas["age"] = datas["age"].astype(float).astype(int)
datas["sex"] = datas["sex"].astype(float).astype(int)
datas["trestbps"] = datas["trestbps"].astype(float).astype(int)
datas["cp"] = datas["cp"].astype(float).astype(int)
datas["fbs"] = datas["fbs"].astype(float).astype(int)
datas["thal"] = datas["thal"].astype(float).astype(int)
datas["restecg"] = datas["restecg"].astype(float).astype(int)
datas["thalach"] = datas["thalach"].astype(float).astype(int)
datas["exang"] = datas["exang"].astype(float).astype(int)
datas["slope"] = datas["slope"].astype(float).astype(int)
datas["target"] = datas["target"].astype(float).astype(int)

datas['target'] = np.where(datas['target'] == 0, 0, 1)

X = datas.drop("target",axis=1)
Y = datas["target"]



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.20, stratify=Y,random_state=42)
X_test_copy = X_test.copy()

# Train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200,max_depth=6, min_samples_split=5, random_state=42)
model=rf_model.fit(X_train, Y_train)

# Make predictions on the test set
rf_predictions= rf_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

rf_Acc = accuracy_score(Y_test, rf_predictions)
precision = precision_score(Y_test, rf_predictions, average='weighted')
recall = recall_score(Y_test, rf_predictions, average='weighted')
f1 = f1_score(Y_test, rf_predictions, average='weighted')
conf_matrix = confusion_matrix(Y_test, rf_predictions)

print(f"rf_Acc: {rf_Acc:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)



import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    
    ("model", RandomForestClassifier())
])

# Train model
pipeline.fit(X_train, Y_train)

# Save using pickle
with open("heart_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully!")


