import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.svm import SVC

print("\n Loading data...")

df=pd.read_csv('data/Churn_data.csv')
print(df.head())

#Analysing data and removing unwanted columns
df=df.drop("customerID",axis=1)
df['Churn'].value_counts()
print(df.info())

print("\n Preprocessing data...")

#Converting 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Filling NaN values in 'TotalCharges' (e.g., with 0 or the mean/median). Using 0 as a simple imputation.
df['TotalCharges'].fillna(0, inplace=True)

print("\n Creating preprocessing pipeline...")

X = df.drop(['Churn'], axis=1)
y = df["Churn"].map({"Yes":1,"No":0})

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])
print("\n Training models...")
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
models = {
    'Logistic Regression': make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    'Random Forest': make_pipeline(
        preprocessor,
        RandomForestClassifier(n_estimators=500, random_state=42)
    ),
    'SVM': make_pipeline(
        preprocessor,
        SVC(kernel='rbf', probability=True, random_state=42)
    )
}
 
results = {}
trained_models = {}
 
for name, pipeline in models.items():
    print(f"\n  Training {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    trained_models[name] = pipeline
    
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

print("\n Selecting best model...")

best_model_name = max(results, key=lambda x: results[x]['f1'])
best_pipeline = trained_models[best_model_name]
 
print(f"\nBest Model: {best_model_name}")
print(f"\nPerformance Comparison:")
print("-" * 70)
results_df = pd.DataFrame(results).T
print(results_df.round(4))

joblib.dump(best_pipeline, 'models/churn_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print(f"\n Model saved to 'models/churn_model.pkl'")
print(f" Preprocessor saved to 'models/preprocessor.pkl'")
 
print("\n Detailed evaluation on test set...")
 
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
 
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=100)
print("Confusion matrix plot saved to 'models/confusion_matrix.png'")
 
# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=100)
print("ROC curve plot saved to 'models/roc_curve.png'")