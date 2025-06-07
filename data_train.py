# model_comparison.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# 1. Load & preprocess
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)

# 2. Encode target & one-hot categoricals
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Select only the most predictive features
features = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'Contract_One year',
    'Contract_Two year',
    'InternetService_Fiber optic',
    'PaymentMethod_Electronic check',
    'PaperlessBilling_Yes'
]
X = df[features]
y = df['Churn']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. Build a ColumnTransformer that:
#    • scales 'tenure'
#    • applies PCA on ['MonthlyCharges','TotalCharges'] → 1 component
#    • passes through the five dummy variables
preprocessor = ColumnTransformer(
    transformers=[
        ('scale_tenure', StandardScaler(), ['tenure']),
        ('pca_charge', 
            ImbPipeline([
                ('scale', StandardScaler()),
                ('pca', PCA(n_components=1))
            ]),
            ['MonthlyCharges','TotalCharges']
        ),
        ('pass_cat', 'passthrough', [
            'Contract_One year',
            'Contract_Two year',
            'InternetService_Fiber optic',
            'PaymentMethod_Electronic check',
            'PaperlessBilling_Yes'
        ])
    ],
    remainder='drop'
)

# 6. Define our nine models
models = {
    'Logistic_L1': LogisticRegression(
        penalty='l1', solver='saga',
        max_iter=5000, random_state=42
    ),
    'Logistic_L2': LogisticRegression(
        penalty='l2', solver='lbfgs',
        max_iter=2000, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    'DecisionTree': DecisionTreeClassifier(
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    )
}

# 7. Evaluate each in an imbalanced-learning pipeline
with open('model_comparison.txt', 'w', encoding='utf-8') as log:
    for name, clf in models.items():
        pipeline = ImbPipeline([
            ('preproc', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        # some models support probability estimates
        y_score = (pipeline.predict_proba(X_test)[:,1]
                   if hasattr(pipeline, 'predict_proba') else None)

        # compute metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_score) if y_score is not None else None
        cm   = confusion_matrix(y_test, y_pred)

        # log results
        log.write(f"=== {name} ===\n")
        log.write(f"Accuracy:  {acc:.3f}\n")
        log.write(f"Precision: {prec:.3f}\n")
        log.write(f"Recall:    {rec:.3f}\n")
        log.write(f"F1 Score:  {f1:.3f}\n")
        if roc is not None:
            log.write(f"ROC AUC:   {roc:.3f}\n")
        log.write("Confusion Matrix:\n")
        log.write(np.array2string(cm))
        log.write("\nClassification Report:\n")
        log.write(classification_report(y_test, y_pred))
        # for logistic models, log coefficient importance
        if name.startswith('Logistic'):
            coefs = pipeline.named_steps['clf'].coef_[0]
            feat_names = pipeline.named_steps['preproc']\
                                  .get_feature_names_out()
            log.write("Coefficients:\n")
            for fn, c in zip(feat_names, coefs):
                log.write(f"  {fn}: {c:.3f}\n")
        log.write("\n\n")

print("Done. Results logged in model_comparison.txt")
# This code is a continuation of the previous analysis script.
# It builds and evaluates multiple machine learning models on the Telco Customer Churn dataset.