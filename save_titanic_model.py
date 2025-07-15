# ✅ Ready to copy

# Step 1: Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import shap
import pickle
import matplotlib.pyplot as plt

# Step 2: Load data
titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Step 3: Preprocessing
titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
titanic["Embarked"].fillna(titanic["Embarked"].mode()[0], inplace=True)

# Encode Sex and Embarked
le_sex = LabelEncoder()
titanic["Sex"] = le_sex.fit_transform(titanic["Sex"])

le_embarked = LabelEncoder()
titanic["Embarked"] = le_embarked.fit_transform(titanic["Embarked"])

# Select features
X = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = titanic["Survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Hyperparameter tuning
params = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
}

grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Best model
model = grid.best_estimator_

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: SHAP explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
plt.show()

# Step 7: Save model
pickle.dump(model, open("final_titanic_model.pkl", "wb"))
print("✅ Model saved as final_titanic_model.pkl")
